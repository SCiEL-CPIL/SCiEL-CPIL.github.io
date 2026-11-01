import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import itertools

N = 14
D = 3
DEADLINE = 115.0  # Safety margin from 120s

def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    d = cdist(pts, pts)
    mask = ~np.eye(N, dtype=bool)
    return d[mask]

def _true_score(pts):
    """Calculate the actual score: (max/min)^2."""
    d = _pairwise_distances(pts)
    if len(d) == 0:
        return np.inf
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _soft_log_ratio(dist, tau):
    """
    Smooth approximation of log(max_d) - log(min_d) using soft min/max.
    """
    mask = ~np.eye(N, dtype=bool)
    d = dist[mask]
    
    # Stable soft max
    max_d = np.max(d)
    soft_max = max_d + tau * np.log(np.sum(np.exp((d - max_d) / tau)))
    
    # Stable soft min
    min_d = np.min(d)
    soft_min = min_d - tau * np.log(np.sum(np.exp((min_d - d) / tau)))
    
    return soft_max - soft_min

def _objective(x, tau):
    """
    Objective function to minimize.
    """
    pts = x.reshape(N, D)
    # Center the points
    pts = pts - np.mean(pts, axis=0)
    
    d = _pairwise_distances(pts)
    
    if np.any(d < 1e-12):
        return 1e10
        
    return _soft_log_ratio(d, tau)

def _generate_fib_sphere_points(seed=0):
    """Generate points on a sphere using Fibonacci distribution."""
    rng = np.random.default_rng(seed)
    i = np.arange(N)
    # Add some randomness to the standard fib sphere to break symmetry if needed
    # But fib sphere is a good start.
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def _generate_icosahedron_plus_2():
    """
    Generate a configuration based on an icosahedron (12 vertices) plus 2 points.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Add 2 points on the poles
    extra = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    return np.vstack([verts, extra])

def _generate_symmetric_rings():
    """
    Generate points in two rings plus poles.
    6 top, 6 bottom, 2 poles.
    """
    angles = 2 * np.pi * np.arange(6) / 6
    r_top = 0.8
    h_top = 0.6
    top_pts = np.column_stack([r_top * np.cos(angles), r_top * np.sin(angles), np.full(6, h_top)])
    
    angles_bot = angles + np.pi / 6
    r_bot = 0.8
    h_bot = -0.6
    bot_pts = np.column_stack([r_bot * np.cos(angles_bot), r_bot * np.sin(angles_bot), np.full(6, h_bot)])
    
    poles = np.array([[0, 0, 1.2], [0, 0, -1.2]])
    
    return np.vstack([top_pts, bot_pts, poles])

def _generate_lattice_points():
    """
    Generate points from an FCC lattice within a sphere.
    """
    # FCC lattice vectors
    # Just generate a grid and filter
    grid_size = 4
    coords = []
    for x in range(-grid_size, grid_size + 1):
        for y in range(-grid_size, grid_size + 1):
            for z in range(-grid_size, grid_size + 1):
                # FCC condition: x+y+z is even? Or just cubic?
                # Let's use simple cubic for now, it's dense enough.
                if x**2 + y**2 + z**2 <= 10: # Radius ~3.16
                    coords.append([x, y, z])
    
    coords = np.array(coords)
    # Center
    coords = coords - np.mean(coords, axis=0)
    # Normalize to unit sphere roughly
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    coords = coords / norms
    
    # Select 14 points that are well distributed?
    # Just take the first 14? Or random 14?
    # Random selection from a dense set might be good.
    idx = np.random.choice(len(coords), N, replace=False)
    return coords[idx]

def construct():
    t_start = time.time()
    
    best_score = np.inf
    best_pts = None
    
    # Initial candidates
    candidates = []
    candidates.append(_generate_fib_sphere_points(0))
    candidates.append(_generate_fib_sphere_points(1))
    candidates.append(_generate_icosahedron_plus_2())
    candidates.append(_generate_symmetric_rings())
    
    # Add some random sphere points
    for i in range(5):
        rng = np.random.default_rng(i)
        pts = rng.normal(0, 1, size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        candidates.append(pts)
        
    # Add lattice points
    candidates.append(_generate_lattice_points())
    
    # Two-stage optimization: coarse (large tau) then fine (small tau)
    tau_stages = [0.3, 0.1, 0.05, 0.02]
    
    refined_candidates = []
    
    for guess in candidates:
        if time.time() - t_start > DEADLINE - 10:
            break
            
        x0 = guess.flatten()
        
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(tau_stages[0],),
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.fun < 1e5:
                refined_candidates.append(res.x)
        except Exception:
            continue

    # Keep top candidates by objective value
    if refined_candidates:
        cand_scores = []
        for cx in refined_candidates:
            pts = cx.reshape(N, D)
            pts = pts - np.mean(pts, axis=0)
            
            d = _pairwise_distances(pts)
            if d.min() < 1e-6:
                continue
                
            score = _true_score(pts)
            cand_scores.append((score, cx))
        
        cand_scores.sort(key=lambda x: x[0])
        # Keep top 6 for fine-tuning
        top_candidates = [c[1] for c in cand_scores[:6]]
    else:
        top_candidates = [g.flatten() for g in candidates[:3]]
    
    # Second pass: fine-tune top candidates with decreasing tau
    for cx in top_candidates:
        if time.time() - t_start > DEADLINE - 5:
            break
        
        x_curr = cx.copy()
        
        for tau in tau_stages[1:]:
            if time.time() - t_start > DEADLINE - 2:
                break
            
            try:
                res = minimize(
                    _objective,
                    x_curr,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # Only update if improved
                if res.fun < _objective(x_curr, tau) + 1e-6:
                    x_curr = res.x
            except Exception:
                continue
        
        # Evaluate final score
        pts = x_curr.reshape(N, D)
        pts = pts - np.mean(pts, axis=0)
        
        d = _pairwise_distances(pts)
        if d.min() < 1e-6:
            continue
            
        score = _true_score(pts)
        if score < best_score:
            best_score = score
            best_pts = pts
    
    # If we have time, try a few more random starts with fine tau
    if time.time() - t_start < DEADLINE - 10:
        rng = np.random.default_rng(123)
        for _ in range(10):
            if time.time() - t_start > DEADLINE - 5:
                break
            
            x0 = rng.normal(0, 1, size=(N, D)).flatten()
            
            try:
                # Quick coarse
                res1 = minimize(
                    _objective, x0, args=(0.2,),
                    method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-9}
                )
                if res1.fun > 1e5:
                    continue
                
                # Fine
                res2 = minimize(
                    _objective, res1.x, args=(0.05,),
                    method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-11}
                )
                
                pts = res2.x.reshape(N, D)
                pts = pts - np.mean(pts, axis=0)
                
                d = _pairwise_distances(pts)
                if d.min() < 1e-6:
                    continue
                    
                score = _true_score(pts)
                if score < best_score:
                    best_score = score
                    best_pts = pts
                    
            except Exception:
                continue

    # Fallback
    if best_pts is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        
    return best_pts
