import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    def score(points):
        # points: (N, D)
        # Calculate pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dists = dists[mask]
        if len(dists) == 0:
            return 1e10
        min_d = np.min(dists)
        max_d = np.max(dists)
        if min_d < 1e-6:
            return 1e10
        return (max_d / min_d) ** 2

    def objective(x):
        points = x.reshape(N, D)
        return score(points)

    best_score = float('inf')
    best_points = None

    # Generate 24-cell vertices in 4D
    # The 24-cell has 24 vertices: all permutations of (±1, ±1, 0, 0)
    import itertools
    vertices_4d = []
    for signs in itertools.product([1, -1], repeat=2):
        for pos in itertools.permutations(range(4)):
            v = np.zeros(4)
            v[pos[0]] = signs[0]
            v[pos[1]] = signs[1]
            vertices_4d.append(v)
    vertices_4d = np.array(vertices_4d)
    
    # Select 14 vertices from the 24-cell (first 14)
    selected_vertices = vertices_4d[:N]
    
    # Also try other subsets for diversity
    rng = np.random.default_rng(42)
    
    # Projection approach: optimize a 4x3 projection matrix
    # points_3d = vertices_4d @ P where P is 4x3
    def projection_objective(p_flat):
        P = p_flat.reshape(4, 3)
        points = selected_vertices @ P
        return score(points)
    
    # Initializations for projection matrix
    proj_inits = []
    
    # 1. Identity-like projection (drop one dimension)
    P0 = np.eye(4, 3)
    proj_inits.append(P0.flatten())
    
    # 2. Random orthogonal-ish projections
    for _ in range(10):
        Q = rng.normal(size=(4, 3))
        # Orthonormalize columns
        Q, _ = np.linalg.qr(Q)
        proj_inits.append(Q.flatten())
    
    # 3. Scaled random projections
    for _ in range(5):
        P_rand = rng.normal(size=(4, 3)) * 0.5
        proj_inits.append(P_rand.flatten())
    
    # Optimize projection matrix
    for init_p in proj_inits:
        if time.time() > deadline - 15:
            break
            
        try:
            res = minimize(
                projection_objective, 
                init_p, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                P_opt = res.x.reshape(4, 3)
                best_points = selected_vertices @ P_opt
        except Exception as e:
            pass
    
    # Also try direct coordinate optimization as fallback/complement
    inits = []
    for _ in range(5):
        inits.append(rng.random((N, D)))
        
    for _ in range(5):
        pts = rng.normal(size=(N, D))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
        
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective, 
                init.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(N, D)
        except Exception as e:
            pass

    if best_points is None:
        # Fallback to seed
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_points = np.array(grid, dtype=float)
        best_score = score(best_points)

    return best_points
