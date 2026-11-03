import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

# Vertices of equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array of shape (N,2) to Cartesian (N,2)."""
    # w = 1 - u - v
    u = bary[:, 0]
    v = bary[:, 1]
    x = u * V1[0] + v * V2[0]
    y = u * V1[1] + v * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(cart):
    """Convert Cartesian (N,2) to barycentric (u,v) where w=1-u-v."""
    x = cart[:, 0]
    y = cart[:, 1]
    # Solve: x = u*1 + v*0.5, y = u*0 + v*sqrt(3)/2
    v = y / (np.sqrt(3) / 2)
    u = x - 0.5 * v
    return np.column_stack([u, v])

def _all_dets_from_bary(bary):
    """
    Compute all triangle areas (normalized) from barycentric coords.
    bary: (N, 2) array of (u, v) coords
    Returns array of areas for all i<j<k triples.
    """
    n = len(bary)
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_f = i_idx[mask]
    j_f = j_idx[mask]
    k_f = k_idx[mask]
    
    # For barycentric coords, area ratio = |det of barycentric differences|
    # Triangle (i,j,k) area / triangle area = |det([bi; bj; bk])| where bi=(ui, vi, wi)
    # But we can compute using Cartesian or directly.
    # Let's use Cartesian for clarity.
    cart = _bary_to_cart(bary)
    pi = cart[i_f]
    pj = cart[j_f]
    pk = cart[k_f]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _min_area_ratio_from_bary(bary):
    """Return minimum triangle area divided by triangle area."""
    areas = _all_dets_from_bary(bary)
    return np.min(areas) / TRI_AREA

def _objective_smooth(x, bary_bounds=True):
    """
    Objective: negative min area ratio + penalty for invalid barycentric coords.
    x: flat array of length 2*N (u0, v0, u1, v1, ...)
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for out of simplex
    pen = np.maximum(0, -u) + np.maximum(0, -v) + np.maximum(0, -w)
    pen_sq = np.sum(pen ** 2)
    
    if pen_sq > 1e-4:
        return pen_sq
    
    # Clip to valid range for area computation
    u_clipped = np.clip(u, 0, 1)
    v_clipped = np.clip(v, 0, 1)
    w_clipped = 1 - u_clipped - v_clipped
    # Re-normalize if needed
    total = u_clipped + v_clipped + w_clipped
    if total < 1e-10:
        return 100.0
    bary_clipped = np.column_stack([u_clipped / total, v_clipped / total])
    
    area_ratio = _min_area_ratio_from_bary(bary_clipped)
    return -area_ratio

def _structured_init_fps(rng, n_points=N, n_trials=5):
    """
    Generate structured initializations using Farthest Point Sampling.
    Returns list of barycentric coordinate arrays (N, 2).
    """
    inits = []
    
    # FPS on the triangle
    for trial in range(n_trials):
        # Start with a random point
        u0 = rng.random()
        v0 = rng.random()
        while u0 + v0 > 1:
            u0 = rng.random()
            v0 = rng.random()
        
        bary = np.zeros((n_points, 2))
        bary[0] = [u0, v0]
        
        for i in range(1, n_points):
            # Generate many random candidates
            n_cand = 200
            uc = rng.random(n_cand)
            vc = rng.random(n_cand)
            mask = uc + vc <= 1
            uc = uc[mask]
            vc = vc[mask]
            if len(uc) == 0:
                continue
            cand_bary = np.column_stack([uc, vc])
            
            # Compute min distance to existing points
            best_dist = -1
            best_idx = 0
            for ci in range(len(cand_bary)):
                min_d = np.inf
                for ei in range(i):
                    d = np.linalg.norm(cand_bary[ci] - bary[ei])
                    if d < min_d:
                        min_d = d
                if min_d > best_dist:
                    best_dist = min_d
                    best_idx = ci
            bary[i] = cand_bary[best_idx]
        
        inits.append(bary.flatten())
    
    # Add grid-based init
    # Place points on a coarse grid in barycentric space
    grid_u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    grid_v = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    grid_pts = []
    for gu in grid_u:
        for gv in grid_v:
            if gu + gv <= 1 + 1e-9:
                grid_pts.append([gu, gv])
    grid_pts = np.array(grid_pts)
    
    # Select N points from grid via FPS
    bary = np.zeros((N, 2))
    bary[0] = grid_pts[0]
    for i in range(1, N):
        best_dist = -1
        best_idx = 0
        for gi in range(len(grid_pts)):
            min_d = np.inf
            for ei in range(i):
                d = np.linalg.norm(grid_pts[gi] - bary[ei])
                if d < min_d:
                    min_d = d
            if min_d > best_dist:
                best_dist = min_d
                best_idx = gi
        bary[i] = grid_pts[best_idx]
    inits.append(bary.flatten())
    
    # Add random inits
    for _ in range(10):
        bary = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            bary[i] = [u, v]
        inits.append(bary.flatten())
    
    return inits

def _project_to_bary_simplex(bary):
    """Project barycentric coords to the simplex u,v,w >= 0, u+v+w=1."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    w = 1.0 - u - v
    
    # Simple projection: clip negatives and renormalize
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = 1.0 - u - v
    w = np.clip(w, 0, 1)
    
    total = u + v + w
    valid = total > 1e-10
    u[valid] /= total[valid]
    v[valid] /= total[valid]
    # w is determined
    return np.column_stack([u, v])

def _local_optimize_nm(x0, deadline, maxiter=300):
    """Local optimization with Nelder-Mead."""
    try:
        res = minimize(
            _objective_smooth,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if time.time() > deadline:
            return x0
        return res.x
    except Exception:
        return x0

def _local_optimize_lbfgs(x0, deadline, maxiter=200):
    """Local optimization with L-BFGS-B."""
    try:
        # Define bounds: u in [0,1], v in [0,1], u+v <= 1
        # L-BFGS-B doesn't handle inequality u+v<=1 directly, so use penalty
        bounds = [(0, 1)] * (2 * N)
        res = minimize(
            _objective_smooth,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if time.time() > deadline:
            return x0
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, deadline, maxiter=100):
    """Local optimization with SLSQP."""
    try:
        # Constraints: u >= 0, v >= 0, u+v <= 1 for each point
        # Inequality constraints: c(x) >= 0
        def cons_u(x):
            bary = x.reshape(N, 2)
            return bary[:, 0]  # u >= 0
        
        def cons_v(x):
            bary = x.reshape(N, 2)
            return bary[:, 1]  # v >= 0
        
        def cons_w(x):
            bary = x.reshape(N, 2)
            return 1.0 - bary[:, 0] - bary[:, 1]  # w >= 0
        
        constraints = [
            {'type': 'ineq', 'fun': cons_u},
            {'type': 'ineq', 'fun': cons_v},
            {'type': 'ineq', 'fun': cons_w}
        ]
        
        bounds = [(0, 1)] * (2 * N)
        
        res = minimize(
            _objective_smooth,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        if time.time() > deadline:
            return x0
        return res.x
    except Exception:
        return x0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    best_x = None
    best_score = -float('inf')
    
    # Generate initializations
    inits = _structured_init_fps(rng, n_points=N, n_trials=5)
    
    def evaluate(x):
        """Evaluate a configuration, return score and validity."""
        bary = x.reshape(N, 2)
        # Project to simplex
        bary = _project_to_bary_simplex(bary)
        cart = _bary_to_cart(bary)
        
        # Check validity
        valid = True
        for i in range(N):
            xi, yi = cart[i]
            if yi < -1e-9 or yi > np.sqrt(3) * xi + 1e-9 or yi > -np.sqrt(3) * xi + np.sqrt(3) + 1e-9:
                valid = False
                break
        
        if not valid:
            return -float('inf'), False
        
        score = _min_area_ratio_from_bary(bary)
        return score, True
    
    # Phase 1: Global-ish search with Nelder-Mead
    for init in inits:
        if time.time() > deadline - 60:
            break
        
        x_opt = _local_optimize_nm(init, deadline, maxiter=200)
        score, valid = evaluate(x_opt)
        if valid and score > best_score:
            best_score = score
            best_x = x_opt
    
    if time.time() > deadline - 30:
        # Time to return best
        if best_x is not None:
            bary = _project_to_bary_simplex(best_x.reshape(N, 2))
            return _bary_to_cart(bary)
        # Fallback
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.33, 0.33], [0.33, 0.0], [0.0, 0.33],
            [0.66, 0.0], [0.66, 0.33], [0.33, 0.66],
            [0.5, 0.25], [0.25, 0.5]
        ])
        bary = _project_to_bary_simplex(bary)
        return _bary_to_cart(bary)
    
    # Phase 2: Polish with L-BFGS-B and SLSQP
    if best_x is not None:
        # Polish with L-BFGS-B
        x_polished = _local_optimize_lbfgs(best_x, deadline, maxiter=150)
        score, valid = evaluate(x_polished)
        if valid and score > best_score:
            best_score = score
            best_x = x_polished
        
        # Polish with SLSQP
        x_polished = _local_optimize_slsqp(best_x, deadline, maxiter=80)
        score, valid = evaluate(x_polished)
        if valid and score > best_score:
            best_score = score
            best_x = x_polished
    
    # Phase 3: More diverse search with perturbations
    if time.time() < deadline - 20 and best_x is not None:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb best solution
            noise = rng.randn(2 * N) * 0.1
            x_perturbed = best_x + noise
            x_perturbed = _project_to_bary_simplex(x_perturbed.reshape(N, 2)).flatten()
            
            x_opt = _local_optimize_slsqp(x_perturbed, deadline, maxiter=60)
            score, valid = evaluate(x_opt)
            if valid and score > best_score:
                best_score = score
                best_x = x_opt
    
    # Final result
    if best_x is not None:
        bary = _project_to_bary_simplex(best_x.reshape(N, 2))
        return _bary_to_cart(bary)
    
    # Fallback
    bary = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.33, 0.33], [0.33, 0.0], [0.0, 0.33],
        [0.66, 0.0], [0.66, 0.33], [0.33, 0.66],
        [0.5, 0.25], [0.25, 0.5]
    ])
    bary = _project_to_bary_simplex(bary)
    return _bary_to_cart(bary)
