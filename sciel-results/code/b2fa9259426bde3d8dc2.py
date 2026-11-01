import numpy as np
import time
from scipy.optimize import minimize
from functools import partial

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute the true (non-smooth) score: max/min squared distance."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def objective_smooth(x, tau=10.0, lambda_reg=1e-4):
    """
    Smooth log-ratio objective using soft-max and soft-min.
    x is flattened array of shape (N*D,)
    Includes a small regularization term to fix scale.
    """
    points = x.reshape(N, D)
    
    # Regularization to prevent scale drift
    reg_term = lambda_reg * np.sum(points**2)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max and soft min via log-sum-exp
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Numerical stability
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    soft_min = np.maximum(soft_min, 1e-15)
    
    log_ratio = np.log(soft_max) - np.log(soft_min)
    
    return log_ratio + reg_term

def fibonacci_sphere(n_points, radius=1.0):
    """Generate n_points on a sphere using Fibonacci lattice."""
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        r = np.sqrt(1.0 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x * radius, y * radius, z * radius])
    return np.array(points)

def rhombic_dodecahedron():
    """Generate vertices of a rhombic dodecahedron."""
    # 8 cube vertices
    pts = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                pts.append([i, j, k])
    # 6 octahedron vertices (scaled)
    # To make it a proper rhombic dodecahedron, the ratio of coordinates matters.
    # Standard rhombic dodecahedron vertices: (0, ±1, ±1), (±1, 0, ±1), (±1, ±1, 0)
    # This is actually the dual of the cuboctahedron.
    # Let's use the standard definition:
    # Vertices are (0, ±1, ±1), (±1, 0, ±1), (±1, ±1, 0)
    # This gives 12 vertices. Wait.
    # A rhombic dodecahedron has 14 vertices.
    # Vertices: (±1, ±1, ±1) and (0, ±2, 0), (±2, 0, 0), (0, 0, ±2) is one realization.
    # Another realization: (0, ±1, ±1), (±1, 0, ±1), (±1, ±1, 0) is 12 vertices (Cuboctahedron dual? No, that's the set of points).
    # Let's check the number of vertices for (0, ±1, ±1)...
    # x=0, y=±1, z=±1 -> 4
    # x=±1, y=0, z=±1 -> 4
    # x=±1, y=±1, z=0 -> 4
    # Total 12. This is the cuboctahedron vertices? No, cuboctahedron has 12 vertices.
    # Rhombic dodecahedron has 14 vertices.
    # The 14 vertices can be: (±1, ±1, ±1) [8] and (0, ±2, 0), (±2, 0, 0), (0, 0, ±2) [6].
    # Let's use this.
    pts = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                pts.append([i, j, k])
    pts.append([0, 2, 0]); pts.append([0, -2, 0])
    pts.append([2, 0, 0]); pts.append([-2, 0, 0])
    pts.append([0, 0, 2]); pts.append([0, 0, -2])
    return np.array(pts, dtype=float)

def icosahedron():
    """Generate vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    pts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    return np.array(pts, dtype=float)

def normalize_points(points):
    """Center and scale points to unit variance."""
    points = points - np.mean(points, axis=0)
    std = np.std(points)
    if std > 1e-10:
        points = points / std
    return points

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Rhombic Dodecahedron (14 vertices)
    rd = rhombic_dodecahedron()
    rd = normalize_points(rd)
    candidates.append(rd.flatten())
    
    # 2. Fibonacci Sphere
    fib_pts = fibonacci_sphere(N)
    fib_pts = normalize_points(fib_pts)
    candidates.append(fib_pts.flatten())
    
    # 3. Icosahedron + 2 points
    ico = icosahedron()
    ico = normalize_points(ico)
    # Add 2 points at random positions on the sphere, then optimize
    for _ in range(3):
        extra = rng.normal(size=(2, D))
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        pts = np.vstack([ico, extra])
        pts = normalize_points(pts)
        candidates.append(pts.flatten())
        
    # 4. Random points on sphere
    for seed in range(10):
        r = np.random.default_rng(seed)
        pts = r.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        pts = normalize_points(pts)
        candidates.append(pts.flatten())
        
    # 5. Random points in ball
    for seed in range(10):
        r = np.random.default_rng(100 + seed)
        pts = r.normal(size=(N, D))
        pts = normalize_points(pts)
        candidates.append(pts.flatten())
        
    # Optimization loop
    tau_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 10:
            break
        
        x_curr = init_x.copy()
        
        try:
            # Phase 1: L-BFGS-B with increasing tau
            for tau in tau_schedule:
                if time.time() - start_time > timeout - 5:
                    break
                
                obj = partial(objective_smooth, tau=tau)
                res = minimize(
                    obj,
                    x_curr,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if res.x is not None and np.isfinite(res.fun):
                    x_curr = res.x
            
            # Phase 2: SLSQP Polish
            if time.time() - start_time < timeout - 2:
                try:
                    obj_final = partial(objective_smooth, tau=10000.0)
                    res_polish = minimize(
                        obj_final,
                        x_curr,
                        method='SLSQP',
                        options={'maxiter': 200, 'ftol': 1e-14, 'maxls': 50}
                    )
                    if res_polish.x is not None and np.isfinite(res_polish.fun):
                        x_curr = res_polish.x
                except Exception:
                    pass
            
            # Evaluate true score
            pts = x_curr.reshape(N, D)
            true_s = raw_score(pts)
            if np.isfinite(true_s) and true_s < best_score:
                best_score = true_s
                best_points = pts.copy()
                
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = normalize_points(fibonacci_sphere(N))
        best_score = raw_score(best_points)
        
    # Final validation
    sq_dists = get_pairwise_sq_dists(best_points)
    if np.min(sq_dists) < 1e-8:
        best_points = normalize_points(fibonacci_sphere(N))
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
