import numpy as np
from scipy.optimize import minimize, differential_evolution
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
    u = bary[:, 0]
    v = bary[:, 1]
    x = u * V1[0] + v * V2[0]
    y = u * V1[1] + v * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric (u,v) where w=1-u-v."""
    x = cart[:, 0]
    y = cart[:, 1]
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
    total = u_clipped + v_clipped + w_clipped
    if total < 1e-10:
        return 100.0
    bary_clipped = np.column_stack([u_clipped / total, v_clipped / total])
    
    area_ratio = _min_area_ratio_from_bary(bary_clipped)
    return -area_ratio

def _structured_init_hexagonal(rng):
    """
    Generate initializations based on hexagonal lattice patterns.
    """
    inits = []
    
    # Hexagonal lattice initialization
    # Place points in a hexagonal pattern within the triangle
    hex_inits = []
    for scale in [0.1, 0.2, 0.3]:
        for offset in [0.0, 0.1, 0.2]:
            bary = np.zeros((N, 2))
            # Place points in a hexagonal grid
            for i in range(N):
                row = i // 3
                col = i % 3
                u = 0.5 + col * scale + offset
                v = 0.5 + row * scale * np.sqrt(3) / 2 + offset
                if u + v <= 1:
                    bary[i] = [u, v]
                else:
                    bary[i] = [0.5, 0.5]
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
    
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = 1.0 - u - v
    w = np.clip(w, 0, 1)
    
    total = u + v + w
    valid = total > 1e-10
    u[valid] /= total[valid]
    v[valid] /= total[valid]
    return np.column_stack([u, v])

def _local_optimize_slsqp(x0, deadline, maxiter=100):
    """Local optimization with SLSQP."""
    try:
        def cons_u(x):
            bary = x.reshape(N, 2)
            return bary[:, 0]
        
        def cons_v(x):
            bary = x.reshape(N, 2)
            return bary[:, 1]
        
        def cons_w(x):
            bary = x.reshape(N, 2)
            return 1.0 - bary[:, 0] - bary[:, 1]
        
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
    
    def evaluate(x):
        """Evaluate a configuration, return score and validity."""
        bary = x.reshape(N, 2)
        bary = _project_to_bary_simplex(bary)
        cart = _bary_to_cart(bary)
        
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
    
    # Phase 1: Global search with Differential Evolution
    bounds = [(0, 1)] * (2 * N)
    
    try:
        de_result = differential_evolution(
            _objective_smooth,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            callback=lambda xk, convergence: None
        )
        if time.time() > deadline - 30:
            raise TimeoutError
        x_opt = de_result.x
        score, valid = evaluate(x_opt)
        if valid and score > best_score:
            best_score = score
            best_x = x_opt
    except Exception as e:
        pass
    
    # Phase 2: Polish with SLSQP
    if best_x is not None:
        x_polished = _local_optimize_slsqp(best_x, deadline, maxiter=100)
        score, valid = evaluate(x_polished)
        if valid and score > best_score:
            best_score = score
            best_x = x_polished
    
    # Phase 3: More diverse search with perturbations
    if time.time() < deadline - 20 and best_x is not None:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
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
