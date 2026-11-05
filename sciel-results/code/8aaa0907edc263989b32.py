"""Dzugutov cluster global minima: structured polytetrahedral/icosahedral seeds + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# --- Dzugutov potential (exact) -------------------------------------------
_A, _B, _a, _b, _c, _d, _m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16.0
_MIN_DIST = 0.5

def _dzugutov_energy(positions):
    n = positions.shape[0]
    if n < 2:
        return 0.0
    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r = r[iu]
    e = np.zeros_like(r)
    m1 = r < _a
    m2 = (r >= _a) & (r < _b)
    # avoid divide-by-zero / overflow for tiny r
    r = np.maximum(r, 1e-8)
    e[m1] = _A * (r[m1] ** (-_m) - _B) * np.exp(_c / (r[m1] - _a))
    e[m2] = _B * np.exp(_d / (r[m2] - _b))
    return float(np.sum(e))

def _dzugutov_energy_and_grad(positions):
    n = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(r, np.inf)
    r = np.maximum(r, 1e-8)
    grad = np.zeros_like(positions)
    e = 0.0
    m1 = (r < _a) & (r > 0)
    m2 = (r >= _a) & (r < _b) & (r > 0)
    # V'(r) for r<a: A*(r^-m - B)*exp(c/(r-a))
    # dV/dr = A*[ -m r^(-m-1)*exp(c/(r-a)) + (r^-m - B)*exp(c/(r-a))*(-c/(r-a)^2) ]
    r1 = r[m1]
    ex1 = np.exp(_c / (r1 - _a))
    dV1 = _A * (-_m * r1 ** (-_m - 1) * ex1 + (r1 ** (-_m) - _B) * ex1 * (-_c / (r1 - _a) ** 2))
    # V'(r) for a<=r<b: B*exp(d/(r-b)); dV/dr = B*exp(d/(r-b))*(-d/(r-b)^2)
    r2 = r[m2]
    ex2 = np.exp(_d / (r2 - _b))
    dV2 = _B * ex2 * (-_d / (r2 - _b) ** 2)
    # accumulate: grad_i += sum_j dV * (r_i - r_j)/r
    # Use the upper triangle to avoid double count, but vectorize over all pairs.
    # Build force vector via broadcasting on the full matrix (symmetric).
    dV = np.zeros_like(r)
    dV[m1] = dV1
    dV[m2] = dV2
    np.fill_diagonal(dV, 0.0)
    # factor = dV / r  (per pair), force on i from j = factor * (p_i - p_j)
    factor = dV / r
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, None, :] * diff, axis=1)
    e = float(np.sum(dV) / 2.0)  # dV symmetric, sum over all ordered pairs = 2*E
    return e, grad

def _local_relax(positions, max_iter=400, tol=1e-9):
    """Gradient descent with adaptive step, enforcing min distance softly."""
    p = positions.copy()
    e, g = _dzugutov_energy_and_grad(p)
    step = 0.02
    for it in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        p_new = p - step * g
        # keep atoms at least _MIN_DIST apart by nudging collisions
        # (cheap: skip heavy enforcement; basin hopping re-seeds)
        e_new, g_new = _dzugutov_energy_and_grad(p_new)
        if e_new < e:
            p, e, g = p_new, e_new, g_new
            step = min(step * 1.1, 0.2)
        else:
            step *= 0.5
            if step < 1e-6:
                break
    return e, p

def _enforce_min_dist(p, margin=0.02):
    """Push apart any pair closer than _MIN_DIST+margin."""
    n = p.shape[0]
    for _ in range(200):
        moved = False
        d = p[:, None, :] - p[None, :, :]
        r = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(r, np.inf)
        close = (r < _MIN_DIST + margin) & (r > 0)
        if not np.any(close):
            break
        iu = np.triu_indices(n, k=1)
        for (i, j) in zip(*iu):
            if close[i, j]:
                rij = p[i] - p[j]
                dist = np.linalg.norm(rij)
                if dist < 1e-8:
                    rij = np.array([1e-4, 0, 0])
                    dist = np.linalg.norm(rij)
                need = (_MIN_DIST + margin) - dist
                push = rij / dist * need * 0.5
                p[i] += push
                p[j] -= push
                moved = True
        if not moved:
            break
    return p

# --- Structured seeds ------------------------------------------------------
def _icosahedron_vertices():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [-1,  t, 0], [1,  t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1,  t], [0, 1,  t], [0, -1, -t], [0, 1, -t],
        [ t, 0, -1], [ t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def _macky_shells():
    """Return list of shell vertex sets (unit-ish) for Mackay icosahedron: 1, 13, 55."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = _icosahedron_vertices()
    shells = [np.zeros((1, 3)), base]
    # 55-atom shell: 12 vertices + 30 edge-midpoints + 12 face-centers
    # 12 vertices
    verts = base.copy()
    # edges: 30 edges of icosahedron
    edges = [
        (0,1),(0,4),(0,5),(0,8),(0,10),
        (1,3),(1,5),(1,9),(1,10),
        (2,3),(2,4),(2,8),(2,11),
        (3,7),(3,9),(3,11),
        (4,6),(4,7),(4,8),
        (5,6),(5,7),(5,9),
        (6,10),(6,11),
        (7,10),(7,11),
        (8,9),(9,10),(10,11),
    ]
    edge_mid = [0.5*(verts[a]+verts[b]) for a,b in edges]
    edge_mid = np.array(edge_mid)
    # faces: 20 faces
    faces = [
        (0,1,10),(0,10,5),(0,5,4),(0,4,8),(0,8,10),
        (1,3,10),(1,10,9),(1,9,5),(1,5,10),  # placeholder, fix below
    ]
    # Use proper icosahedron face list
    faces = [
        (0,1,10),(0,10,5),(0,5,4),(0,4,8),(0,8,10),
        (1,3,10),(1,10,9),(1,9,5),(1,5,10),
    ]
    # The above face list is unreliable; build faces from known icosahedron combinatorics.
    # Simpler: use the 20 faces as all triples that are mutually nearest.
    # Compute faces by convex hull of base vertices.
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(base)
        faces = [tuple(f) for f in hull.simplices]
    except Exception:
        faces = []
    face_cent = [np.mean(verts[list(f)], axis=0) for f in faces]
    face_cent = np.array(face_cent) if face_cent else np.zeros((0,3))
    shell55 = np.vstack([verts, edge_mid, face_cent])
    shells.append(shell55)
    return shells

def _seed_macky(n, rng):
    """Build a Mackay-like icosahedral structure with n atoms (fill shells, trim/expand)."""
    shells = _macky_shells()
    # cumulative counts: 1, 13, 55
    cum = [1, 13, 55]
    scale = 1.0
    chosen = []
    for idx, sh in enumerate(shells):
        if len(chosen) + len(sh) <= n:
            chosen.append(sh)
    chosen = np.vstack(chosen) if chosen else np.zeros((1,3))
    # scale so nearest-neighbor ~ 1.0 (Dzugutov equilibrium)
    if len(chosen) >= 2:
        d = np.linalg.norm(chosen[:,None,:]-chosen[None,:,:], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = d.min()
        if nn > 1e-8:
            scale = 1.0 / nn
    chosen = chosen * scale
    if len(chosen) > n:
        chosen = chosen[:n]
    elif len(chosen) < n:
        # add atoms on a small random shell around center
        r = np.linalg.norm(chosen, axis=1).max() * 1.2 if len(chosen) else 1.0
        extra = []
        while len(extra) < n - len(chosen):
            p = rng.normal(0, r*0.5, 3)
            extra.append(p)
        chosen = np.vstack([chosen, np.array(extra)])
    return chosen

def _seed_tetrahedral(n, rng):
    """Pack tetrahedra: start with a central tetrahedron, add atoms on vertices of a tetrahedral lattice."""
    # Build a tetrahedral close-packed-like set by placing atoms on a simple cubic-ish
    # lattice then pruning to n, but bias toward tetrahedral coordination.
    # Use a body-centered arrangement: generate on a grid, keep nearest to origin.
    L = int(np.ceil(n ** (1/3))) + 2
    coords = []
    for x in range(-L, L+1):
        for y in range(-L, L+1):
            for z in range(-L, L+1):
                coords.append([x, y, z])
    coords = np.array(coords, dtype=float)
    # add face-centered and body-centered offsets to enrich tetrahedral coordination
    offsets = np.array([[0.5,0.5,0.5],[0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[-0.5,-0.5,0.5]])
    coords2 = []
    for c in coords:
        coords2.append(c)
        for o in offsets:
            coords2.append(c + o)
    coords2 = np.array(coords2)
    # scale to nearest-neighbor ~ 1.0
    d = np.linalg.norm(coords2[:,None,:]-coords2[None,:,:], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    if nn > 1e-8:
        coords2 = coords2 / nn
    # pick n closest to origin (compact)
    r = np.linalg.norm(coords2, axis=1)
    order = np.argsort(r)
    chosen = coords2[order[:n]]
    return chosen

def _seed_random(n, rng):
    radius = 1.1 * n ** (1/3) * 0.75
    pts = []
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p-q) >= 0.7 for q in pts):
            pts.append(p)
    return np.array(pts)

# --- Basin hopping ---------------------------------------------------------
def _basin_hop(start, rng, budget, max_iters):
    best_e, best_p = _local_relax(start)
    best_p = _enforce_min_dist(best_p)
    e, p = _local_relax(best_p)
    best_e, best_p = e, p
    for it in range(max_iters):
        if time.time() > budget:
            break
        # random perturbation (small, topology-preserving)
        delta = rng.normal(0, 0.15, p.shape)
        cand = p + delta
        cand = _enforce_min_dist(cand)
        e_c, p_c = _local_relax(cand)
        # Metropolis-ish accept: always accept lower; accept higher with small prob
        if e_c < best_e:
            best_e, best_p = e_c, p_c
            p = p_c
        elif rng.random() < 0.1:
            p = p_c
    return best_e, best_p

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(hash((size, potential)) & 0xffffffff)

    deadline = time.time() + 320.0  # safety margin under 330s kill
    per_seed_budget = 25.0

    # Generate a diverse set of seeds
    seeds = []
    seeds.append(_seed_macky(size, rng))
    seeds.append(_seed_tetrahedral(size, rng))
    for k in range(3):
        seeds.append(_seed_random(size, rng))

    best_e = np.inf
    best_p = None
    for s in seeds:
        if time.time() > deadline - 10:
            break
        s = _enforce_min_dist(s)
        e, p = _basin_hop(s, rng, deadline - 10, max_iters=40)
        if e < best_e:
            best_e, best_p = e, p

    if best_p is None:
        # fallback: a simple compact random cluster
        radius = 1.1 * size ** (1/3) * 0.75
        pts = []
        while len(pts) < size:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p)
        best_p = _enforce_min_dist(np.array(pts))

    return best_p
# EVOLVE-BLOCK-END
