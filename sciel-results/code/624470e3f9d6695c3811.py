"""Suite task: cover unit sphere with 130 equal caps (minimize covering radius)."""
import numpy as np
import time

N = 130
DEADLINE = 145.0  # hard wall-clock limit with safety margin


def _normalize(P):
    P = P - P.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    P = P / norms
    return P


def _facet_center(a, b, c):
    """Compute the circumcenter (on unit sphere) of triangle (a,b,c) on the sphere.
    Returns the outward-pointing unit normal of the great-circle triangle."""
    # Try all orderings to find a consistent orientation
    for perm in ((a, b, c), (b, a, c), (a, c, b)):
        x, y, z = perm
        n = np.cross(x, y)
        nn = np.linalg.norm(n)
        if nn > 1e-12:
            n = n / nn
            # Check that z is on the same side as n (interior)
            if np.dot(n, z) > 0:
                return n
    # Fallback: use centroid
    ctr = a + b + c
    nc = np.linalg.norm(ctr)
    if nc < 1e-12:
        return None
    return ctr / nc


def _cover_radius(P):
    """Compute the covering radius in radians via convex hull facet circumradii."""
    from scipy.spatial import ConvexHull
    P = _normalize(P.copy())
    if np.any(~np.isfinite(P)):
        return np.inf
    try:
        h = ConvexHull(P)
    except Exception:
        return np.inf
    
    max_r = 0.0
    for f in h.simplices:
        a, b, c = P[f[0]], P[f[1]], P[f[2]]
        ctr = _facet_center(a, b, c)
        if ctr is None:
            continue
        r = max(np.arccos(np.clip(np.dot(ctr, v), -1, 1)) for v in (a, b, c))
        max_r = max(max_r, r)
    return max_r


def _find_hot_facet(P):
    """Find the facet with maximum circumradius. Returns (indices, center, radius)."""
    from scipy.spatial import ConvexHull
    P = _normalize(P.copy())
    try:
        h = ConvexHull(P)
    except Exception:
        return None, None, -1.0
    
    worst = -1.0
    hot = None
    hot_ctr = None
    for f in h.simplices:
        a, b, c = P[f[0]], P[f[1]], P[f[2]]
        ctr = _facet_center(a, b, c)
        if ctr is None:
            continue
        r = max(np.arccos(np.clip(np.dot(ctr, v), -1, 1)) for v in (a, b, c))
        if r > worst:
            worst = r
            hot = f
            hot_ctr = ctr
    return hot, hot_ctr, worst


def _gradient_step(P, hot, ctr, step_size):
    """Move hot facet vertices away from the facet center to reduce covering radius."""
    Pnew = P.copy()
    for idx in hot:
        # Move point away from center: gradient of arccos(dot(ctr, p)) w.r.t. p
        # d/dp arccos(ctr . p) = -ctr / sqrt(1 - (ctr.p)^2)
        dot_val = np.dot(ctr, P[idx])
        denom = np.sqrt(max(1e-15, 1.0 - dot_val * dot_val))
        grad = -ctr / denom
        Pnew[idx] = Pnew[idx] + step_size * grad
    return Pnew


def _optimize(P_init, time_budget, verbose=False):
    """Local optimization via steepest descent on the hot facet."""
    t0 = time.time()
    P = _normalize(P_init.copy())
    best = _cover_radius(P)
    bestP = P.copy()
    
    step = 0.01
    momentum = np.zeros_like(P)
    no_improve = 0
    max_no_improve = 300
    
    while time.time() - t0 < time_budget:
        hot, ctr, worst = _find_hot_facet(P)
        if hot is None:
            break
        
        # Compute gradient
        g = np.zeros_like(P)
        for idx in hot:
            dot_val = np.dot(ctr, P[idx])
            denom = np.sqrt(max(1e-15, 1.0 - dot_val * dot_val))
            g[idx] = -ctr / denom
        
        # Momentum
        momentum = 0.5 * momentum + 0.5 * g
        gm = np.linalg.norm(momentum, axis=1, keepdims=True)
        gm = np.where(gm > 1e-15, gm, 1.0)
        momentum = momentum / gm
        
        # Try step
        Pnew = _normalize(P + step * momentum)
        val = _cover_radius(Pnew)
        
        if val < best - 1e-12:
            P = Pnew
            best = val
            bestP = Pnew.copy()
            no_improve = 0
            # Increase step slightly if we made good progress
            step = min(step * 1.05, 0.05)
        else:
            no_improve += 1
            step *= 0.8
            if step < 1e-6:
                step = 0.01
                momentum *= 0.1
            if no_improve > max_no_improve:
                break
    
    return bestP, best


def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng, n):
    """Generate n random points uniformly on the unit sphere."""
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-1, 1, n)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def _perturbed_spiral(rng, noise_scale):
    P = _fibonacci_spiral()
    P = P + noise_scale * rng.standard_normal((N, 3))
    return _normalize(P)


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    bestP = None
    best = np.inf
    
    # Phase 1: Try multiple diverse initializations with short optimizations
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_spiral())
    
    # Random points
    initializations.append(_random_sphere(rng, N))
    
    # Perturbed spirals with different noise levels
    for noise in [0.05, 0.1, 0.2]:
        initializations.append(_perturbed_spiral(rng, noise))
    
    # Grid-based initialization (cubic lattice projected to sphere)
    grid = np.array([[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)])
    grid = grid / np.linalg.norm(grid, axis=1, keepdims=True)
    # Subsample to get N points
    idx = rng.choice(len(grid), N, replace=False)
    initializations.append(grid[idx])
    
    # Optimize each initialization with a time budget
    n_init = len(initializations)
    per_init_budget = (DEADLINE - 30) / max(1, n_init)  # Reserve 30s for final polish
    
    for i, P_init in enumerate(initializations):
        elapsed = time.time() - t_start
        remaining = DEADLINE - 30 - elapsed
        if remaining < 2:
            break
        budget = min(per_init_budget, remaining)
        if budget < 1:
            break
        Pc, vc = _optimize(P_init, budget)
        if vc < best:
            best = vc
            bestP = Pc.copy()
    
    # Phase 2: Final polish on the best configuration with remaining time
    elapsed = time.time() - t_start
    remaining = DEADLINE - 10 - elapsed
    if remaining > 2 and bestP is not None:
        Pc, vc = _optimize(bestP, remaining)
        if vc < best:
            best = vc
            bestP = Pc.copy()
    
    # Safety: ensure valid output
    if bestP is None:
        bestP = _fibonacci_spiral()
    
    bestP = _normalize(np.asarray(bestP, dtype=float))
    return bestP
