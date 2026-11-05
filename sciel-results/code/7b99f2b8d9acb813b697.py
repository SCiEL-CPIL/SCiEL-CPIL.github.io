"""Rewritten: topology-informed (polytetrahedral) search with barrier-augmented L-BFGS for Dzugutov clusters."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# ---- Dzugutov potential (recomputed exactly for our own scoring/barrier) ----
_A, _B, _a, _b, _c, _d, _m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16

def _dzugutov_energy(pos):
    """Exact Dzugutov energy, vectorized."""
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    r = r[iu]
    e = np.zeros_like(r)
    mask_a = r < _a
    mask_b = (r < _b) & ~mask_a
    e[mask_a] = _A * (r[mask_a] ** (-_m) - _B) * np.exp(_c / (r[mask_a] - _a))
    e[mask_b] = _B * np.exp(_d / (r[mask_b] - _b))
    return float(np.sum(e))

# Minimum of the inner branch r^-m - B  ->  r_min = (m*A/(m*A*B))**(1/m) = (1/B)^(1/m)
_RMIN = (1.0 / _B) ** (1.0 / _m)

def _valid(pos, n):
    """Check minimum separation 0.5 and finiteness."""
    if pos.shape != (n, 3):
        return False
    if not np.all(np.isfinite(pos)):
        return False
    d = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=2)
    iu = np.triu_indices(n, 1)
    return bool(np.all(r[iu] >= 0.5 - 1e-9))

# ---- Icosahedral geometry helpers ----
_PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _icosahedron_unit():
    """12 vertices of a regular icosahedron on the unit sphere."""
    v = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1],
    ], dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v

def _icosahedron_edge(v):
    """Edge length of the icosahedron with given vertices (distance between nearest neighbours)."""
    d = v[:, None, :] - v[None, :, :]
    r = np.linalg.norm(d, axis=2)
    iu = np.triu_indices(len(v), 1)
    return float(np.min(r[iu]))

def _icosahedron_scaled(scale):
    """Icosahedron vertices scaled so nearest-neighbour distance == scale."""
    v = _icosahedron_unit()
    e = _icosahedron_edge(v)
    return v * (scale / e)

def _mackay_icosahedron(n_target):
    """Build a Mackay icosahedral cluster with as many atoms as possible <= n_target,
    using shells: 1 (center), 12 (1st shell), 42 (2nd shell). Returns positions (n,3)."""
    # Shell radii in units of edge a: r1 = a, r2 = a*sqrt( (3+sqrt(5))/2 )? use known ratios.
    # Known: 1st shell atoms at distance a from center; 2nd shell at distance a * (1+sqrt(5))/2? 
    # Use geometric construction: place 2nd shell on the icosahedron faces (12 faces * 3 = 36) + edges? 
    # Simpler robust approach: generate candidate points on a sphere and keep those matching shell distances.
    # We instead build shells by known coordinates.
    scale = _RMIN  # nearest-neighbour distance
    v1 = _icosahedron_scaled(scale)  # 12 atoms at distance scale from center
    center = np.zeros(3)
    # 2nd shell: Mackay 42 atoms. Coordinates = all points at distance a2 = scale * (1 + 1/PHI) from center
    # lying on the icosahedron's symmetry. Use the 30 edge-midpoint-type + 12 face-type.
    a2 = scale * (1.0 + 1.0 / _PHI)
    # face centers: normalize each face normal * a2
    # get faces (triangles) of icosahedron by finding triplets at mutual distance scale
    d = v1[:, None, :] - v1[None, :, :]
    r = np.linalg.norm(d, axis=2)
    faces = []
    iu = np.triu_indices(12, 1)
    edge_set = set()
    for i, j in zip(iu[0], iu[1]):
        if abs(r[i, j] - scale) < 1e-6:
            edge_set.add((min(i, j), max(i, j)))
    for i in range(12):
        for j in range(i + 1, 12):
            if (i, j) in edge_set:
                for k in range(j + 1, 12):
                    if (j, k) in edge_set and (i, k) in edge_set:
                        faces.append((i, j, k))
    face_centers = []
    for (i, j, k) in faces:
        nrm = np.cross(v1[j] - v1[i], v1[k] - v1[i])
        nrm = nrm / np.linalg.norm(nrm)
        face_centers.append(nrm * a2)
    # edge midpoints scaled to a2: 30 edges -> 30 points at distance a2 along edge directions? 
    # Mackay 2nd shell = 12 face centers + 30 edge-related? Actually 42 = 12 + 30.
    edge_points = []
    for (i, j) in edge_set:
        mid_dir = (v1[i] + v1[j])
        mid_dir = mid_dir / np.linalg.norm(mid_dir)
        edge_points.append(mid_dir * a2)
    shell2 = np.array(face_centers + edge_points)
    # dedupe
    shell2 = np.unique(np.round(shell2, 9), axis=0)
    positions = np.vstack([center[None, :], v1, shell2])
    # If we have more than n_target, trim the 2nd shell from the end; if fewer, we just use what we have.
    if positions.shape[0] > n_target:
        # keep center + 12 first shell always, then fill with as many 2nd-shell as fit
        keep = n_target - 13
        if keep < 0:
            keep = 0
        positions = np.vstack([center[None, :], v1, shell2[:max(0, keep)]])
    return positions

def _random_compact(n, rng, scale):
    """Random points in a ball with min separation ~ scale, best-effort."""
    radius = scale * n ** (1.0 / 3.0) * 0.9
    pts = []
    tries = 0
    while len(pts) < n and tries < 200 * n:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= 0.6 * scale for q in pts):
            pts.append(p)
    # fallback: fill with a small grid jitter if we couldn't get enough
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)

# ---- Barrier-augmented objective (keeps bonds near r_min) ----
def _barrier_objective(pos, n, lam):
    """Energy + lambda * sum over close pairs of (r - r_min)^2 for r < 0.9*r_min (too close) 
    and (r_min - r)^2 for r in (0.9*r_min, 1.2*r_min) to encourage bond length."""
    if not np.all(np.isfinite(pos)):
        return 1e6
    # separation guard
    d = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=2)
    iu = np.triu_indices(n, 1)
    rp = r[iu]
    if np.any(rp < 0.5):
        return 1e6 + 1e4 * np.sum(np.maximum(0.5 - rp, 0) ** 2)
    e = _dzugutov_energy(pos)
    # barrier: penalize deviation from r_min for pairs that are "bonds" (r < 1.3*r_min)
    bond = rp < 1.3 * _RMIN
    dev = rp[bond] - _RMIN
    barrier = lam * np.sum(dev ** 2)
    return e + barrier

def _barrier_grad(pos, n, lam):
    if not np.all(np.isfinite(pos)):
        return np.ones_like(pos)
    d = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=2)
    iu = np.triu_indices(n, 1)
    rp = r[iu]
    if np.any(rp < 0.5):
        # simple repulsive gradient
        g = np.zeros_like(pos)
        for (i, j) in zip(iu[0], iu[1]):
            if rp[i] < 0.5:
                direction = (pos[i] - pos[j])
                direction = direction / (np.linalg.norm(direction) + 1e-9)
                g[i] += 1e4 * (0.5 - rp[i]) * direction
                g[j] -= 1e4 * (0.5 - rp[i]) * direction
        return g
    ge = dzugutov_energy_gradient(pos)
    # barrier gradient
    bond = rp < 1.3 * _RMIN
    gb = np.zeros_like(pos)
    for (i, j) in zip(iu[0], iu[1]):
        if bond[i]:
            direction = (pos[i] - pos[j]) / (rp[i] + 1e-9)
            coef = 2.0 * lam * (rp[i] - _RMIN)
            gb[i] += coef * direction
            gb[j] -= coef * direction
    return ge + gb

def _lbfgs_refine(pos, n, lam, time_left):
    """Run L-BFGS on the barrier-augmented objective; return refined positions and energy."""
    start = time.time()
    def obj(x):
        if time.time() - start > time_left - 1.0:
            return 1e7
        return _barrier_objective(x.reshape(n, 3), n, lam)
    def grad(x):
        if time.time() - start > time_left - 1.0:
            return np.ones_like(x)
        return _barrier_grad(x.reshape(n, 3), n, lam).reshape(-1)
    x0 = pos.reshape(-1)
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-4, 'maxls': 20})
        p = res.x.reshape(n, 3)
        if _valid(p, n):
            return p
    except Exception:
        pass
    return pos

def construct(instance):
    size = int(instance["n"])
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    t_start = time.time()
    DEADLINE = 320.0  # safety margin under 330s kill

    rng = np.random.default_rng(size)

    # Candidate topologies / starts
    starts = []
    # 1) Mackay icosahedral (polytetrahedral) topology
    try:
        starts.append(_mackay_icosahedron(size))
    except Exception:
        pass
    # 2) Random compact starts (several)
    for _ in range(3):
        starts.append(_random_compact(size, rng, _RMIN))

    best_pos = None
    best_e = np.inf

    def consider(pos):
        nonlocal best_pos, best_e
        if not _valid(pos, size):
            return
        e = _dzugutov_energy(pos)
        if e < best_e:
            best_e = e
            best_pos = pos.copy()

    # Process each start: local relax + barrier L-BFGS + re-relax
    for idx, s in enumerate(starts):
        if time.time() - t_start > DEADLINE - 5.0:
            break
        # First, a cheap local relax to settle
        try:
            e0, p0 = relax(np.array(s, dtype=float), potential, parameter)
            if _valid(p0, size):
                consider(p0)
            work = p0 if _valid(p0, size) else np.array(s, dtype=float)
        except Exception:
            work = np.array(s, dtype=float)

        # Barrier-augmented L-BFGS with a couple of lambda values
        for lam in (0.5, 0.05):
            if time.time() - t_start > DEADLINE - 5.0:
                break
            remaining = DEADLINE - (time.time() - t_start)
            if remaining < 3.0:
                break
            try:
                p = _lbfgs_refine(work, size, lam, remaining)
                if _valid(p, size):
                    consider(p)
                    work = p
            except Exception:
                pass

        # Final local relax on the best-so-far of this branch
        if time.time() - t_start > DEADLINE - 3.0:
            break
        try:
            e1, p1 = relax(work, potential, parameter)
            if _valid(p1, size):
                consider(p1)
        except Exception:
            pass

    # If we still have time, do a few random-perturbation + relax rounds from the best
    if best_pos is not None:
        for _ in range(6):
            if time.time() - t_start > DEADLINE - 3.0:
                break
            pert = best_pos + rng.normal(0, 0.05, best_pos.shape)
            if not _valid(pert, size):
                continue
            try:
                e2, p2 = relax(pert, potential, parameter)
                if _valid(p2, size):
                    consider(p2)
            except Exception:
                pass

    # Fallback: ensure we always return a valid configuration
    if best_pos is None or not _valid(best_pos, size):
        # build a guaranteed-valid ring
        ang = 2 * np.pi * np.arange(size) / size
        R = max(1.0, 0.5 * size / (2 * np.pi) * 1.05)
        ring = np.column_stack([R * np.cos(ang), R * np.sin(ang), np.zeros(size)])
        # ensure separation >= 0.5
        sep = 2 * R * np.sin(np.pi / size)
        if sep < 0.5:
            R = 0.5 / (2 * np.sin(np.pi / size) + 1e-9) * 1.05
            ring = np.column_stack([R * np.cos(ang), R * np.sin(ang), np.zeros(size)])
        best_pos = ring
        best_e = _dzugutov_energy(best_pos)

    return best_pos
# EVOLVE-BLOCK-END
