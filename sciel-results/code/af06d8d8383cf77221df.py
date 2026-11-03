"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius(points):
    """Compute covering radius in radians via convex-hull facet circumcenters."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: sample the sphere and take max angular distance to nearest point
        from scipy.spatial import cKDTree
        rng = np.random.RandomState(0)
        tree = cKDTree(points)
        n_samples = 20000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return float(np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1))))

    max_r = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    return float(max_r)

def _fast_facets(points):
    """Return list of (radius, circumcenter, facet_indices) for triangular facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    out = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        out.append((r, v, simplex))
    return out

def _facet_driven_local_search(points, time_limit=None, start_time=None, max_iters=500, step_size=0.05):
    """
    Local search that moves vertices of the largest facet towards its circumcenter.
    This is a direct geometric optimization.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    best = points.copy()
    best_cr = _covering_radius(points)
    
    if start_time is None:
        start_time = time.time()

    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        facets = _fast_facets(points)
        if not facets:
            break
            
        # Find the facet with the largest radius
        max_idx = 0
        max_r = facets[0][0]
        for i in range(1, len(facets)):
            if facets[i][0] > max_r:
                max_r = facets[i][0]
                max_idx = i
                
        r, center, simplex = facets[max_idx]
        
        # Move the 3 vertices of this facet towards the circumcenter
        cand = points.copy()
        for idx in simplex:
            p = cand[idx]
            # Move towards center
            move = center - p
            cand[idx] = p + step_size * move
            cand[idx] /= np.linalg.norm(cand[idx])
            
        cand_cr = _covering_radius(cand)
        if cand_cr < best_cr:
            best = cand.copy()
            best_cr = cand_cr
            points = cand
        else:
            # If no improvement, try a smaller step or break
            step_size *= 0.95
            if step_size < 1e-5:
                break

    return best, best_cr

def _spherical_lloyd(points, max_iters=200, time_limit=None, start_time=None):
    """
    Lloyd-style: each point moves to the average circumcenter of its adjacent facets.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        facets = _fast_facets(points)
        if not facets:
            break
        # Build point -> list of circumcenters
        centers_per_point = [[] for _ in range(len(points))]
        for r, center, simplex in facets:
            for idx in simplex:
                centers_per_point[idx].append(center)
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            if not centers_per_point[i]:
                new_points[i] = points[i]
            else:
                c = np.mean(centers_per_point[i], axis=0)
                cn = np.linalg.norm(c)
                if cn < 1e-12:
                    new_points[i] = points[i]
                else:
                    new_points[i] = c / cn
        # Damped update
        alpha = 0.5
        blended = (1 - alpha) * points + alpha * new_points
        blended /= np.linalg.norm(blended, axis=1, keepdims=True)
        cr = _covering_radius(blended)
        if cr < best_cr:
            points = blended
            best = points.copy()
            best_cr = cr
        else:
            alpha = 0.2
            blended = (1 - alpha) * points + alpha * new_points
            blended /= np.linalg.norm(blended, axis=1, keepdims=True)
            cr = _covering_radius(blended)
            if cr < best_cr:
                points = blended
                best = points.copy()
                best_cr = cr
            else:
                break
    return best, best_cr

def _simulated_annealing(points, time_limit=None, start_time=None, seed=42,
                         T_init=0.05, T_min=1e-6, cooling=0.995):
    """
    Simulated annealing with stress-based perturbation.
    Points in large facets are perturbed more.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    best = points.copy()
    best_cr = _covering_radius(points)
    rng = np.random.RandomState(seed)
    T = T_init
    cur = points.copy()
    cur_cr = best_cr

    if start_time is None:
        start_time = time.time()

    while T > T_min:
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        facets = _fast_facets(cur)
        if not facets:
            break
            
        # Calculate stress for each point: sum of radii of adjacent facets
        stress = np.zeros(N)
        for r, center, simplex in facets:
            for idx in simplex:
                stress[idx] += r
                
        # Select points to perturb, weighted by stress
        # Normalize stress to probabilities
        total_stress = np.sum(stress)
        if total_stress > 0:
            probs = stress / total_stress
            # Perturb 1-3 points
            n_pert = rng.randint(1, 4)
            idxs = rng.choice(N, size=n_pert, replace=False, p=probs)
        else:
            idxs = rng.choice(N, size=rng.randint(1, 4), replace=False)
            
        cand = cur.copy()
        for i in idxs:
            noise = rng.randn(3) * T
            cand[i] = cur[i] + noise
            cand[i] /= np.linalg.norm(cand[i])
            
        cand_cr = _covering_radius(cand)
        delta = cand_cr - cur_cr
        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-12)):
            cur = cand
            cur_cr = cand_cr
            if cand_cr < best_cr:
                best = cand.copy()
                best_cr = cand_cr
        T *= cooling
    return best, best_cr

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _icosahedron_based_points(n):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    pts = base.copy()
    rng = np.random.RandomState(123)
    while len(pts) < n:
        i, j = rng.randint(0, len(pts), 2)
        if i == j:
            continue
        mid = (pts[i] + pts[j]) / 2.0
        mn = np.linalg.norm(mid)
        if mn > 1e-10:
            mid /= mn
            dists = 1.0 - np.sum(mid * pts, axis=1)
            if np.min(dists) > 0.01:
                pts = np.vstack([pts, mid])
    return pts[:n]

def _refine(points, time_limit=None, start_time=None):
    """
    Multi-stage refinement: Lloyd -> Facet-Driven Local Search -> SA -> Lloyd.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    if start_time is None:
        start_time = time.time()

    # Stage 1: Lloyd (Global smoothing)
    rem = time_limit - (time.time() - start_time) if time_limit else 10.0
    if rem > 2:
        t1 = min(rem * 0.2, 5.0)
        points, cr = _spherical_lloyd(points, max_iters=100, time_limit=t1, start_time=start_time)
    else:
        cr = _covering_radius(points)

    # Stage 2: Facet-Driven Local Search (Targeted optimization)
    rem = time_limit - (time.time() - start_time) if time_limit else 10.0
    if rem > 2:
        t2 = min(rem * 0.4, 10.0)
        points, cr = _facet_driven_local_search(points, time_limit=t2, start_time=start_time)

    # Stage 3: Simulated Annealing (Escape local minima)
    rem = time_limit - (time.time() - start_time) if time_limit else 10.0
    if rem > 2:
        t3 = min(rem * 0.3, 10.0)
        points, cr = _simulated_annealing(points, time_limit=t3, start_time=start_time, seed=123)

    # Stage 4: Final Lloyd polish
    rem = time_limit - (time.time() - start_time) if time_limit else 10.0
    if rem > 1:
        t4 = min(rem * 0.5, 5.0)
        points, cr = _spherical_lloyd(points, max_iters=50, time_limit=t4, start_time=start_time)

    return points, cr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    initial_configs = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=7),
        _random_points(N, seed=13),
        _icosahedron_based_points(N),
        _random_points(N, seed=99),
    ]

    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)

    # Phase 1: Refine each initial config
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        time_per = min(remaining / (len(initial_configs) - i), 20.0)
        refined, cr = _refine(init_pts, time_limit=time_per, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()

    # Phase 2: More aggressive SA from best
    remaining = deadline - time.time()
    if remaining > 15:
        sa_time = min(remaining * 0.4, 25.0)
        sa_pts, sa_cr = _simulated_annealing(best_points, time_limit=sa_time,
                                             start_time=start_time, seed=777,
                                             T_init=0.08)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_pts.copy()

    # Phase 3: Final refinement
    remaining = deadline - time.time()
    if remaining > 10:
        fin_time = min(remaining * 0.5, 20.0)
        fin_pts, fin_cr = _refine(best_points, time_limit=fin_time, start_time=start_time)
        if fin_cr < best_cr:
            best_cr = fin_cr
            best_points = fin_pts.copy()

    # Final Facet-Driven polish
    remaining = deadline - time.time()
    if remaining > 5:
        fd_time = max(remaining * 0.7, 3.0)
        fd_pts, fd_cr = _facet_driven_local_search(best_points, time_limit=fd_time,
                                                   start_time=start_time, step_size=0.02)
        if fd_cr < best_cr:
            best_cr = fd_cr
            best_points = fd_pts.copy()

    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
# EVOLVE-BLOCK-END
