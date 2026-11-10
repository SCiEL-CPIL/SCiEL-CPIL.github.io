"""Divide-and-conquer assembly: build 38-60 atom Dzugutov clusters from
small polytetrahedral sub-units (icosahedra / decahedra / fragments), then
optimize the relative placement of the units and finally do a full local
relaxation.  This exploits the hierarchical polytetrahedral order of the
Dzugutov model to reduce the effective search dimensionality."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


# ---------------------------------------------------------------------------
# Dzugutov potential helpers (recomputed exactly, matching the evaluator)
# ---------------------------------------------------------------------------
_A = 5.82
_B = 1.28
_a = 1.87
_b = 1.94
_c = 1.10
_d = 0.27
_m = 16.0


def _V(r):
    """Pair potential V(r) with the Dzugutov form."""
    out = np.zeros_like(r, dtype=float)
    m1 = r < _a
    m2 = (r >= _a) & (r < _b)
    r1 = r[m1]
    r2 = r[m2]
    out[m1] = _A * (r1 ** -_m - _B) * np.exp(_c / (r1 - _a))
    out[m2] = _B * np.exp(_d / (r2 - _b))
    return out


def _pair_energies(pos):
    """Total pair energy of a configuration (pos: (n,3))."""
    n = pos.shape[0]
    if n < 2:
        return 0.0
    d = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=2)
    iu = np.triu_indices(n, 1)
    return float(np.sum(_V(r[iu])))


def _energy_and_grad(pos):
    e = _pair_energies(pos)
    g = dzugutov_energy_gradient(pos)
    return e, g


# ---------------------------------------------------------------------------
# Small polytetrahedral building blocks
# ---------------------------------------------------------------------------
def _icosahedron(center, radius=1.0):
    """13-atom icosahedron: 1 central + 12 vertices of a regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of a regular icosahedron (circumscribed radius = 1)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    verts *= radius
    pts = np.vstack([center[None, :], center[None, :] + verts])
    return pts


def _decahedron(center, radius=1.0):
    """13-atom Mackay decahedron: 1 central + 12 vertices of a regular
    decahedron (vertices at the icosahedron's edge-midpoint directions)."""
    # 12 directions: permutations of (0, +-1, +-phi) are icosahedron vertices.
    # Decahedron vertices are at the 12 edge-centre directions.  A convenient
    # set: the 12 vertices of a regular icosahedron rotated so that faces
    # become 5-fold...  Simpler: use the 12 directions given by the icosahedron
    # but with the decahedron's characteristic shape (two pentagonal caps +
    # belt).  We use the standard Mackay decahedron vertex set.
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices: top pentagon, bottom pentagon, and a middle belt of 12?
    # Mackay decahedron = 12 vertices.  Use the set of 12 vectors:
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [1, phi, 0], [-1, phi, 0],
        [phi, 0, 1], [phi, 0, -1], [0, 1, -phi], [0, -1, -phi],
        [1, -phi, 0], [-1, -phi, 0], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    # Actually that's the same icosahedron set.  For a decahedron we want the
    # 12 vertices of a regular decahedron.  Use the 12 directions:
    # (0, +-1, +-phi) and cyclic -> that's icosahedron.  The decahedron's 12
    # vertices are at the icosahedron's *face* normals? No.
    # To keep it simple and robust, fall back to an icosahedron-like 13-atom
    # unit but slightly squashed along one axis (approximates a decahedron).
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    verts *= radius
    verts[:, 2] *= 0.85  # squash -> decahedral-ish
    pts = np.vstack([center[None, :], center[None, :] + verts])
    return pts


def _random_fragments(n_atoms, rng):
    """Return a list of small (<=13 atom) fragment coordinate arrays whose
    sizes sum to n_atoms.  Fragments are icosahedra / decahedra / small
    random tight clusters."""
    frags = []
    remaining = n_atoms
    # Prefer 13-atom units, then fill with 5-atom fragments.
    while remaining > 0:
        if remaining >= 13 and rng.random() < 0.7:
            kind = rng.random()
            if kind < 0.5:
                frags.append(_icosahedron(np.zeros(3), radius=1.0))
            else:
                frags.append(_decahedron(np.zeros(3), radius=1.0))
            remaining -= 13
        else:
            size = min(remaining, int(rng.integers(4, 8)))
            pts = _random_tight_cluster(size, rng)
            frags.append(pts)
            remaining -= size
    return frags


def _random_tight_cluster(size, rng, spacing=1.0):
    """A small, dense, non-overlapping cluster used as a generic fragment."""
    pts = [np.zeros(3)]
    for _ in range(size - 1):
        for _try in range(200):
            p = rng.normal(0, spacing, 3)
            if all(np.linalg.norm(p - q) >= 0.95 * spacing for q in pts):
                pts.append(p)
                break
        else:
            p = rng.normal(0, spacing * 1.3, 3)
            pts.append(p)
    return np.array(pts)


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------
def _rot_matrix(rng):
    """Random 3D rotation matrix via QR of a Gaussian matrix."""
    M = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(M)
    d = np.sign(np.diag(R))
    Q = Q * d
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _place_fragment(frag, center, rot):
    """Place a fragment (centred at origin) at `center` with rotation `rot`."""
    return frag @ rot.T + center[None, :]


# ---------------------------------------------------------------------------
# Assembly: optimise fragment placements
# ---------------------------------------------------------------------------
def _assemble(fragments, rng, n_iter_optim=60, n_restarts=3):
    """Optimise the relative positions (and orientations) of the fragments.
    Parameters are per-fragment: 3 translation + 3 orientation (Euler-ish via
    a 3-vector mapped to a rotation).  We optimise with a few restarts."""
    n_frag = len(fragments)
    sizes = [f.shape[0] for f in fragments]

    def params_to_config(params):
        pos = []
        for i in range(n_frag):
            t = params[3 * i: 3 * i + 3]
            o = params[3 * i + 3: 3 * i + 6]
            rot = _euler_to_rot(o)
            pos.append(_place_fragment(fragments[i], t, rot))
        return np.vstack(pos)

    def objective(params):
        cfg = params_to_config(params)
        # penalty for any pair closer than 0.5
        d = cfg[:, None, :] - cfg[None, :, :]
        r = np.linalg.norm(d, axis=2)
        iu = np.triu_indices(cfg.shape[0], 1)
        if r[iu].min() < 0.5:
            return 1e6 + (0.5 - r[iu].min()) ** 2
        return _pair_energies(cfg)

    best_e = np.inf
    best_cfg = None
    dim = 6 * n_frag

    for restart in range(n_restarts):
        # initial guess: place fragments in a rough ring / random
        init = np.zeros(dim)
        for i in range(n_frag):
            ang = 2 * np.pi * i / max(n_frag, 1)
            R = 1.5 * np.sqrt(sizes[i])
            init[3 * i] = R * np.cos(ang)
            init[3 * i + 1] = R * np.sin(ang)
            init[3 * i + 2] = rng.normal(0, 0.5)
            init[3 * i + 3: 3 * i + 6] = rng.normal(0, 0.5, 3)
        if restart > 0:
            init = init + rng.normal(0, 0.8, dim)

        res = minimize(objective, init, method="L-BFGS-B",
                       options={"maxiter": n_iter_optim, "gtol": 1e-4})
        if res.fun < best_e:
            best_e = res.fun
            best_cfg = params_to_config(res.x)

    return best_cfg, best_e


def _euler_to_rot(v):
    """Map a 3-vector to a rotation matrix (Rodrigues from unit vector)."""
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.eye(3)
    axis = v / n
    angle = n  # use magnitude as angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# Full local relaxation (used at the end and as a fallback)
# ---------------------------------------------------------------------------
def _relax_full(pos, potential="dzugutov", parameter=None):
    try:
        energy, positions = relax(pos, potential, parameter)
        return energy, positions
    except Exception:
        return _pair_energies(pos), pos


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(hash((size, potential)) % (2 ** 31))

    candidates = []

    # --- Strategy A: divide-and-conquer assembly from polytetrahedral units ---
    for attempt in range(2):
        fragments = _random_fragments(size, rng)
        cfg, e = _assemble(fragments, rng, n_iter_optim=80, n_restarts=4)
        # final full relaxation
        e2, pos2 = _relax_full(cfg, potential, parameter)
        candidates.append((e2, pos2))

    # --- Strategy B: assemble from a few large icosahedra + fillers ---
    # Build as many 13-atom icosahedra as possible, then a filler fragment.
    n_ico = size // 13
    rem = size - 13 * n_ico
    fragments = []
    for _ in range(n_ico):
        fragments.append(_icosahedron(np.zeros(3), radius=1.0))
    if rem > 0:
        fragments.append(_random_tight_cluster(rem, rng))
    cfg, e = _assemble(fragments, rng, n_iter_optim=100, n_restarts=5)
    e2, pos2 = _relax_full(cfg, potential, parameter)
    candidates.append((e2, pos2))

    # --- Strategy C: keep the original random compact seed (safety net) ---
    spacing = 1.1
    radius = spacing * size ** (1.0 / 3.0) * 0.85
    for _ in range(3):
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(
                    np.linalg.norm(p - q) >= 0.9 * spacing for q in points):
                points.append(p)
        e3, pos3 = _relax_full(np.array(points), potential, parameter)
        candidates.append((e3, pos3))

    # pick the best
    candidates.sort(key=lambda x: x[0])
    best_e, best_pos = candidates[0]

    # final safety: ensure minimum separation of 0.5
    best_pos = _enforce_min_distance(best_pos, 0.5, rng)
    return best_pos


def _enforce_min_distance(pos, dmin, rng, max_iter=2000):
    """Nudge atoms apart until all pairs are >= dmin (best effort)."""
    pos = pos.copy()
    n = pos.shape[0]
    for _ in range(max_iter):
        moved = False
        for i, j in combinations(range(n), 2):
            d = pos[j] - pos[i]
            r = np.linalg.norm(d)
            if r < dmin:
                if r < 1e-8:
                    d = rng.normal(0, 0.1, 3)
                    r = np.linalg.norm(d)
                shift = (dmin - r) / r * d
                pos[i] -= shift / 2
                pos[j] += shift / 2
                moved = True
        if not moved:
            break
    return pos
# EVOLVE-BLOCK-END
