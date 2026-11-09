# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Uses a multi-start continuous optimization strategy:
    1. Generate diverse initial configurations (symmetric polytopes + random).
    2. Run Nelder-Mead optimization from each start to maximize (d_min/d_max)^2.
    3. Return the best configuration found.
    """
    n = 14
    d = 3
    deadline = time.time() + 580  # Safety margin
    
    def _objective(x):
        points = x.reshape(n, d)
        return -_score(points)
    
    # Generate initial configurations
    starts = []
    
    # 1. Icosahedron-based starts (12 points) + 2 random
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 points to make 14
    for scale in [0.5, 1.0, 1.5]:
        base = ico * scale
        # Add 2 points at random positions
        extra = np.random.randn(2, d) * 0.5
        config = np.vstack([base, extra])
        starts.append(config.flatten())
        
        # Also try with extra points at opposite positions
        extra_opp = -np.random.randn(2, d) * 0.5
        config_opp = np.vstack([base, extra_opp])
        starts.append(config_opp.flatten())
        
    # 2. Cube-based starts
    cube_pts = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
        [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1],
        [0,1,1], [0,1,-1], [0,-1,1], [0,-1,-1]
    ])
    cube_pts = cube_pts / np.linalg.norm(cube_pts, axis=1, keepdims=True)
    
    for scale in [0.5, 1.0, 1.5]:
        base = cube_pts * scale
        extra = np.random.randn(2, d) * 0.5
        config = np.vstack([base, extra])
        starts.append(config.flatten())
        
    # 3. Random uniform on sphere
    for _ in range(10):
        pts = np.random.randn(n, d)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        starts.append(pts.flatten())
        
    # 4. Random uniform in cube
    for _ in range(10):
        pts = np.random.uniform(-1, 1, (n, d))
        starts.append(pts.flatten())
        
    # 5. Torsion-free random
    for _ in range(10):
        pts = np.random.randn(n, d) * np.random.uniform(0.5, 2.0)
        starts.append(pts.flatten())
        
    # Evaluate all starts and keep top 20 by initial score
    scored_starts = []
    for s in starts:
        score = -_objective(s)
        scored_starts.append((score, s))
        
    scored_starts.sort(key=lambda x: x[0], reverse=True)
    top_starts = [s for _, s in scored_starts[:20]]
    
    # Run Nelder-Mead from each top start
    best_points = None
    best_score = -np.inf
    
    for i, x0 in enumerate(top_starts):
        if time.time() > deadline:
            break
            
        try:
            result = minimize(
                _objective, 
                x0, 
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < -best_score:
                best_points = result.x.reshape(n, d)
                best_score = -result.fun
                
        except Exception:
            continue
            
    # Fallback: if optimization failed, use best initial
    if best_points is None:
        best_score, best_points_flat = scored_starts[0]
        best_points = best_points_flat.reshape(n, d)
        
    # Final polish: one more optimization from the best found
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective, 
                best_points.flatten(), 
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.fun < -best_score:
                best_points = result.x.reshape(n, d)
                best_score = -result.fun
        except Exception:
            pass
            
    return best_points

# EVOLVE-BLOCK-END
