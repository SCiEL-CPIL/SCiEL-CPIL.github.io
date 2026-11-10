"""
Strategy: Multi-start Basin Hopping with Swap Perturbations and Diverse Seeds
- Seeds: deformed FCC, molten (high-T random), icosahedral, random compact.
- Basin Hopping uses BOTH Gaussian noise AND atom-position swaps to escape
  local minima more effectively than noise-only perturbation.
- Staged time budget: Phase 1 (diverse seeds, short BH), Phase 2 (longer BH
  with swaps from best), Phase 3 (extra diverse starts if time remains).
- Deadline-guarded at 325 s for safety under 330 s kill.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_T0 = time.time()
_DEADLINE = 325.0

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _icosahedron_vertices():
    phi = (1 + 5 ** 0.5) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _generate_icosahedral_seed(n, rng):
    """Mackay-style icosahedral seed: core + shells of random surface points."""
    if n <= 1:
        return np.zeros((n, 3))
    pts = [np.array([0.0, 0.0, 0.0])]
    if n >= 13:
        v = _icosahedron_vertices()
        pts.extend([v[i].copy() for i in range(12)])
    # Add remaining atoms on a sphere of radius ~ 1.5
    remaining = n - len(pts)
    for _ in range(remaining):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        pts.append(d * 1.5)
    return np.array(pts[:n])

def _generate_deformed_fcc_seed(n, rng):
    """
    Generate a deformed FCC-like seed.
    Place atoms on FCC lattice points within a sphere, then add small random
    deformation to break symmetry and help escape symmetric local minima.
    """
    if n <= 1:
        return np.zeros((n, 3))
    a = 1.0
    fcc_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                fcc_pts.append(np.array([i * a, j * a, k * a]))
                fcc_pts.append(np.array([(i + 0.5) * a, (j + 0.5) * a, k * a]))
                fcc_pts.append(np.array([(i + 0.5) * a, j * a, (k + 0.5) * a]))
                fcc_pts.append(np.array([i * a, (j + 0.5) * a, (k + 0.5) * a]))
    fcc_pts = np.array(fcc_pts)
    fcc_pts -= fcc_pts.mean(axis=0)
    # Sort by distance from origin
    dists = np.linalg.norm(fcc_pts, axis=1)
    idx = np.argsort(dists)
    fcc_pts = fcc_pts[idx[:n]]
    # Apply small random deformation (deformed FCC)
    deformation = rng.normal(0, 0.15, fcc_pts.shape)
    fcc_pts += deformation
    return fcc_pts

def _generate_molten_seed(n, rng):
    """
    Generate a 'molten' cluster: random points in a sphere with radius chosen
    so density is high enough for compact packing but with enough disorder.
    """
    if n <= 1:
        return np.zeros((n, 3))
    # Radius for roughly unit density packing
    r = 1.0 * n ** (1.0 / 3.0) * 1.2
    pts = []
    for _ in range(n):
        # Uniform in sphere
        while True:
            p = rng.uniform(-1, 1, 3)
            if np.linalg.norm(p) <= 1:
                break
        pts.append(p * r)
    return np.array(pts)

def _generate_random_compact(n, rng):
    """Random compact seed with minimum-distance enforcement."""
    if n <= 1:
        return np.zeros((n, 3))
    r = 0.8 * n ** (1.0 / 3.0) * 1.5
    pts = []
    for _ in range(n):
        for _attempt in range(200):
            p = rng.uniform(-r, r, 3)
            if not pts:
                break
            d = np.linalg.norm(np.array(pts) - p, axis=1)
            if np.min(d) >= 0.5:
                break
        pts.append(p)
    return np.array(pts)

def _bh_with_swaps(positions, potential, kind, rng, max_iters=30, temp=0.5):
    """
    Basin Hopping with BOTH Gaussian noise AND atom-exchange swaps.
    At each step, randomly choose between:
      - Gaussian perturbation (70% probability)
      - Swap two random atom positions (30% probability)
    This helps escape local minima that are trapped by atom-identity barriers.
    """
    n = len(positions)
    try:
        best_e, best_p = relax(positions, potential, kind)
    except:
        return float('inf'), positions

    cur_e, cur_p = best_e, best_p.copy()
    t = temp

    for i in range(max_iters):
        if _time_left() < 5.0:
            break

        # Adaptive step size
        step = 0.3 * (1.0 + 0.5 * np.cos(np.pi * i / max_iters))

        # Choose perturbation type
        if rng.random() < 0.3 and n >= 2:
            # Swap perturbation: exchange two random atoms' positions
            perturbed = cur_p.copy()
            i1, i2 = rng.choice(n, size=2, replace=False)
            perturbed[i1], perturbed[i2] = cur_p[i2].copy(), cur_p[i1].copy()
            # Also add small noise to avoid exact degeneracy
            perturbed += rng.normal(0, step * 0.3, perturbed.shape)
        else:
            # Gaussian noise perturbation
            perturbed = cur_p + rng.normal(0, step, cur_p.shape)

        try:
            new_e, new_p = relax(perturbed, potential, kind)
        except:
            continue

        de = new_e - cur_e
        if de < 0 or rng.random() < np.exp(-de / t):
            cur_e, cur_p = new_e, new_p
            if new_e < best_e:
                best_e, best_p = new_e, new_p.copy()
        t *= 0.95

    return best_e, best_p

def _validate(pos, n):
    """Check validity: finite, shape, min distance."""
    if pos is None or len(pos) != n:
        return False
    if not np.all(np.isfinite(pos)):
        return False
    if n < 2:
        return True
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return np.min(d) >= 0.35

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]

    best_energy = float('inf')
    best_pos = None

    # Phase 1: Diverse seeds, short BH (5-10 iters each)
    # Allocate ~40% of time to Phase 1
    phase1_budget = 0.4 * _DEADLINE
    phase1_deadline = _T0 + phase1_budget

    seeds = []
    rng_seeds = [42, 123, 456, 789, 1000, 2000, 3000]
    for s in rng_seeds:
        rng = np.random.default_rng(s)
        try:
            seeds.append(_generate_icosahedral_seed(n, rng))
        except:
            pass
        try:
            seeds.append(_generate_deformed_fcc_seed(n, rng))
        except:
            pass
        try:
            seeds.append(_generate_molten_seed(n, rng))
        except:
            pass
        try:
            seeds.append(_generate_random_compact(n, rng))
        except:
            pass

    # Deduplicate-ish: keep first few unique-ish
    used = []
    unique_seeds = []
    for s in seeds:
        if s is not None and len(s) == n and np.all(np.isfinite(s)):
            # Simple uniqueness check: compare first few atoms
            is_dup = False
            for u in used:
                if np.allclose(s[:3], u[:3], atol=0.1):
                    is_dup = True
                    break
            if not is_dup:
                unique_seeds.append(s)
                used.append(s)

    # Run short BH on each unique seed
    for i, seed in enumerate(unique_seeds):
        if time.time() > phase1_deadline:
            break
        if _time_left() < 10.0:
            break
        bh_rng = np.random.default_rng(5000 + i * 100)
        try:
            e, pos = _bh_with_swaps(seed, potential, kind, bh_rng, max_iters=8, temp=0.5)
            if e < best_energy:
                best_energy = e
                best_pos = pos
        except:
            try:
                e, pos = relax(seed, potential, kind)
                if e < best_energy:
                    best_energy = e
                    best_pos = pos
            except:
                pass

    # Phase 2: Longer BH with swaps from the best found
    if best_pos is not None and _time_left() > 15.0:
        # Use remaining time for longer BH
        remaining_iters = max(5, int((_time_left() - 10.0) / 2.0))
        bh_rng = np.random.default_rng(9999)
        try:
            e, pos = _bh_with_swaps(best_pos, potential, kind, bh_rng,
                                    max_iters=remaining_iters, temp=0.4)
            if e < best_energy:
                best_energy = e
                best_pos = pos
        except:
            pass

    # Phase 3: If time remains, try a few more diverse starts with medium BH
    if _time_left() > 20.0:
        for i in range(3):
            if _time_left() < 8.0:
                break
            rng = np.random.default_rng(7000 + i * 500)
            try:
                seed = _generate_molten_seed(n, rng)
                e, pos = _bh_with_swaps(seed, potential, kind, rng, max_iters=10, temp=0.5)
                if e < best_energy:
                    best_energy = e
                    best_pos = pos
            except:
                pass

    # Fallback
    if best_pos is None or not _validate(best_pos, n):
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    return best_pos
# EVOLVE-BLOCK-END
