"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150 s)


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n


def _covering_radius_hull(pts):
    """Exact covering radius (radians) via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    best = 0.0
    for f in hull.simplices:
        tri = pts[f]
        a, b, c = tri[0], tri[1], tri[2]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        cc = n / nn
        cosv = np.clip(np.dot(cc, a), -1.0, 1.0)
        r = np.arccos(cosv)
        if r > best:
            best = r
    if best == 0.0:
        G = np.clip(pts @ pts.T, -1.0, 1.0)
        np.fill_diagonal(G, -1.0)
        best = np.arccos(G.min())
    return best


def _covering_radius_fast(pts, n_grid=150):
    """Fast approximate covering radius using a grid on the sphere."""
    pts = _normalize(pts)
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid)
    theta, phi = np.meshgrid(theta, phi)
    grid_x = np.sin(theta) * np.cos(phi)
    grid_y = np.sin(theta) * np.sin(phi)
    grid_z = np.cos(theta)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid = _normalize(grid)
    
    G = grid @ pts.T
    G = np.clip(G, -1.0, 1.0)
    max_cos = G.max(axis=1)
    angles = np.arccos(max_cos)
    return angles.max()


def _build_hss50():
    """Build a configuration based on truncated icosahedron structure."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    ico = _normalize(ico)
    
    D = 1.0 - ico @ ico.T
    min_d = D.min()
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_d) < 1e-8:
                edges.append((i, j))
    
    verts = []
    for (i, j) in edges:
        verts.append(ico[i] * (2.0 / 3.0) + ico[j] * (1.0 / 3.0))
        verts.append(ico[i] * (1.0 / 3.0) + ico[j] * (2.0 / 3.0))
    verts = np.array(verts)
    verts = _normalize(verts)
    
    # Remove 10 vertices to get 50
    vmap = []
    for (i, j) in edges:
        vmap.append((i, j))
        vmap.append((i, j))
    
    G = ico @ ico.T
    np.fill_diagonal(G, -2.0)
    pair_i, pair_j = np.unravel_index(np.argmin(G), G.shape)
    
    remove = set()
    for idx, (i, j) in enumerate(vmap):
        if i == pair_i or j == pair_i or i == pair_j or j == pair_j:
            remove.add(idx)
    
    keep = [k for k in range(len(verts)) if k not in remove]
    out = verts[keep]
    if len(out) != 50:
        # If we don't have exactly 50, take first 50
        out = out[:50]
    return _normalize(out)


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _bfgs_optimize(pts, time_budget):
    """BFGS optimization on the sum of squared circumradii of hull facets.
    
    Uses a penalty method to enforce the sphere constraint.
    """
    pts = _normalize(pts).copy()
    start = time.time()
    
    def objective(x):
        if time.time() - start > time_budget:
            return np.inf
        p = x.reshape(N, 3)
        p = _normalize(p)
        
        # Compute covering radius via hull
        try:
            hull = ConvexHull(p)
        except Exception:
            return 1.0
        
        radii = []
        for f in hull.simplices:
            tri = p[f]
            a, b, c = tri[0], tri[1], tri[2]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            cc = n / nn
            cosv = np.clip(np.dot(cc, a), -1.0, 1.0)
            r = np.arccos(cosv)
            radii.append(r)
        
        if not radii:
            return 1.0
        
        radii = np.array(radii)
        # Sum of squared radii (smooth objective)
        return np.sum(radii**2)
    
    x0 = pts.ravel()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                         options={'maxiter': 500, 'gtol': 1e-8, 'maxls': 50})
        
        if result.fun < objective(x0):
            best_pts = _normalize(result.x.reshape(N, 3))
        else:
            best_pts = pts
    except Exception:
        best_pts = pts
    
    return best_pts


def _local_polish(pts, time_budget):
    """Coordinate-wise local optimization with adaptive steps."""
    pts = _normalize(pts).copy()
    best_r = _covering_radius_hull(pts)
    start = time.time()
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    current_step = 0
    
    while time.time() - start < time_budget:
        improved = False
        for i in range(N):
            for axis in range(3):
                for sgn in (1.0, -1.0):
                    for ss in step_sizes[current_step:current_step+2]:
                        cand = pts.copy()
                        cand[i, axis] += sgn * ss
                        norm = np.linalg.norm(cand[i])
                        if norm < 1e-10:
                            continue
                        cand[i] /= norm
                        r = _covering_radius_hull(cand)
                        if r < best_r - 1e-12:
                            pts = cand
                            best_r = r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            current_step += 1
            if current_step >= len(step_sizes):
                break
    
    return pts, best_r


def construct():
    t0 = time.time()
    
    # Generate multiple initial configurations
    configs = []
    
    # 1. Hardin-Sloane-Smith based
    try:
        hss = _build_hss50()
        if len(hss) == 50 and np.all(np.isfinite(hss)):
            configs.append(hss)
    except Exception:
        pass
    
    # 2. Fibonacci spiral
    fib = _fibonacci_spiral()
    configs.append(fib)
    
    # 3. Random configurations with icosahedral seed
    rng = np.random.RandomState(42)
    for _ in range(2):
        ico = np.array([
            [0, 1, (1+np.sqrt(5))/2], [0, -1, (1+np.sqrt(5))/2],
            [0, 1, -(1+np.sqrt(5))/2], [0, -1, -(1+np.sqrt(5))/2],
            [1, (1+np.sqrt(5))/2, 0], [-1, (1+np.sqrt(5))/2, 0],
            [1, -(1+np.sqrt(5))/2, 0], [-1, -(1+np.sqrt(5))/2, 0],
            [(1+np.sqrt(5))/2, 0, 1], [-(1+np.sqrt(5))/2, 0, 1],
            [(1+np.sqrt(5))/2, 0, -1], [-(1+np.sqrt(5))/2, 0, -1],
        ], dtype=float)
        ico = _normalize(ico)
        
        rand_pts = rng.randn(38, 3)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
        
        all_pts = np.vstack([ico, rand_pts])
        configs.append(all_pts)
    
    # Evaluate all configs and pick the best
    best_pts = None
    best_r = np.inf
    
    for cfg in configs:
        cfg = _normalize(cfg)
        r = _covering_radius_fast(cfg)
        if r < best_r:
            best_r = r
            best_pts = cfg
    
    if best_pts is None:
        best_pts = fib
        best_r = _covering_radius_fast(best_pts)
    
    # BFGS optimization (primary strategy)
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 30.0:
        try:
            bfgs_pts = _bfgs_optimize(best_pts, min(remaining - 30.0, 60.0))
            bfgs_r = _covering_radius_hull(bfgs_pts)
            if bfgs_r < best_r:
                best_pts, best_r = bfgs_pts, bfgs_r
        except Exception:
            pass
    
    # Local polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15.0:
        try:
            polish_pts, polish_r = _local_polish(best_pts, min(remaining - 15.0, 30.0))
            if polish_r < best_r:
                best_pts, best_r = polish_pts, polish_r
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)) or len(best_pts) != N:
        best_pts = fib
    
    return best_pts
# EVOLVE-BLOCK-END
