# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 312.0          # hard wall-clock cap (evaluator kills at 330 s)
_START = time.time()
_MIN_D = 0.35              # validity threshold: no two atoms closer than this
_GAP = 0.46                # packing gap (>= 0.35) used to build valid starts
_RPACK = 0.5               # packing radius for the near-ideal compact start
_FCC_L = 0.72              # FCC lattice constant (a/2 = 0.36 > 0.35 -> valid)
_HCP_L = 0.72              # HCP lattice constant


def _valid(pos):
    """True if finite and every pair is at least _MIN_D apart."""
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    for i in range(n):
        d = pos[i + 1:] - pos[i]
        if np.any(np.einsum('ij,ij->i', d, d) < _MIN_D * _MIN_D):
            return False
    return True


def _energy(pos):
    try:
        e, _ = sutton_chen_energy_gradient(pos, 'sutton_chen')
        return float(e)
    except Exception:
        return np.inf


def _relax_bounded(pos, budget):
    """Run relax() but stop if it overruns the per-start wall-time budget."""
    t0 = time.time()
    while True:
        try:
            e, p = relax(pos, 'sutton_chen', 'sutton_chen')
            e = float(e)
        except Exception:
            return None
        if np.isfinite(e) and _valid(p):
            return e, p
        if time.time() - t0 > budget:
            return None
        pos = pos + 0.01 * np.random.default_rng(int(time.time() * 1e6) & 0xFFFF).standard_normal(pos.shape)


def _compact_random(rng, n, radius, gap):
    """Poisson-style random packing inside a sphere of the given radius."""
    pts = []
    tries = 0
    maxtries = 60 * n
    while len(pts) < n and tries < maxtries:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        ok = True
        for q in pts:
            if np.linalg.norm(p - q) < gap:
                ok = False
                break
        if ok:
            pts.append(p)
    if len(pts) < n:
        return None
    return np.array(pts)


def _compact_ideal(n):
    """Densest-ish packing at radius _RPACK; falls back to a random packing."""
    for gap in (_RPACK, 0.5, 0.48, 0.46, 0.44):
        radius = gap * n ** (1.0 / 3.0) * 1.15
        rng = np.random.default_rng(12345 + n)
        for _ in range(15):
            pts = _compact_random(rng, n, radius, gap)
            if pts is not None:
                return pts
    # last resort: random with a small gap, then validate
    rng = np.random.default_rng(777 + n)
    for gap in (0.4, 0.38, 0.36):
        pts = _compact_random(rng, n, gap * n ** (1.0 / 3.0) * 1.4, gap)
        if pts is not None:
            return pts
    return None


def _fcc_chunk(n, L):
    """First n atoms of an FCC lattice (valid: min distance L/sqrt(2))."""
    pts = []
    a = L / 2.0
    for x in range(-3, 4):
        for y in range(-3, 4):
            for z in range(-3, 4):
                for sx, sy, sz in [(0, 0, 0), (a, a, 0), (a, 0, a), (0, a, a)]:
                    pts.append([x * L + sx, y * L + sy, z * L + sz])
                    if len(pts) >= n:
                        return np.array(pts[:n])
    return None


def _hcp_chunk(n, L):
    """First n atoms of an HCP lattice (valid: min distance L/sqrt(2))."""
    a = L
    c = L * (8.0 / 3.0) ** 0.5
    pts = []
    for layer in range(-4, 5):
        z = layer * c
        offset = (a / 2.0, a / 2.0 / np.sqrt(3.0), 0.0) if layer % 2 else (0.0, 0.0, 0.0)
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * a + offset[0]
                y = (j * np.sqrt(3.0) / 2.0) * a + offset[1]
                pts.append([x, y, z])
                if len(pts) >= n:
                    return np.array(pts[:n])
    return None


def construct(instance):
    global _DEADLINE
    n = int(instance["n"])
    kind = instance.get("kind")

    # Assemble a diverse pool of valid starting configurations.
    starts = []
    seen = set()

    def _add(arr):
        if arr is None:
            return
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (n, 3) or not _valid(arr):
            return
        key = tuple(np.round(arr, 4).flatten())
        if key in seen:
            return
        seen.add(key)
        starts.append(arr)

    _add(_compact_ideal(n))
    rng = np.random.default_rng(n * 131 + 7)
    for gap in (0.46, 0.5, 0.55, 0.6):
        radius = gap * n ** (1.0 / 3.0) * 1.2
        for _ in range(6):
            _add(_compact_random(rng, n, radius, gap))
    for L in (_FCC_L, 0.78, 0.85):
        _add(_fcc_chunk(n, L))
    for L in (_HCP_L, 0.8):
        _add(_hcp_chunk(n, L))

    # A couple of small random perturbations of the ideal start for basin hopping.
    base = starts[0] if starts else None
    if base is not None:
        for s in (0.05, 0.1, 0.15):
            _add(base + s * rng.standard_normal((n, 3)))

    if not starts:
        # Fallback: guaranteed-valid random with a small gap.
        for gap in (0.4, 0.38):
            arr = _compact_random(rng, n, gap * n ** (1.0 / 3.0) * 1.5, gap)
            if arr is not None:
                starts.append(arr)
                break
        if not starts:
            arr = np.zeros((n, 3))
            for i in range(n):
                arr[i] = [i * 0.4, (i % 3) * 0.4, (i // 3) * 0.4]
            starts.append(arr)

    best_e = np.inf
    best_p = None

    # Per-start budget: enough to make progress, but never blow the deadline.
    n_starts = len(starts)
    per_budget = max(0.4, min(4.0, (250.0 - (time.time() - _START)) / max(n_starts, 1)))

    for idx, s in enumerate(starts):
        if time.time() > _DEADLINE - 8.0:
            break
        res = _relax_bounded(s, per_budget)
        if res is not None:
            e, p = res
            if e < best_e:
                best_e = e
                best_p = p

    if best_p is None:
        # Nothing relaxed cleanly; return the best valid start we have.
        for s in starts:
            e = _energy(s)
            if np.isfinite(e) and (best_p is None or e < best_e):
                best_e, best_p = e, s
    if best_p is None:
        best_p = starts[0]
    return best_p
# EVOLVE-BLOCK-END
