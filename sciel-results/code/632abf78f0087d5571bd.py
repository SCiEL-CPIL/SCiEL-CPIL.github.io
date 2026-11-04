"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle.
Combines barycentric coordinate representation, FPS initializations, smooth min-surrogate optimization
(L-BFGS-B), SLSQP refinement, and Nelder-Mead polishing with simulated annealing for combinatorial search."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    Since V0=(0,0): Point = v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * V1[0] + w * V2[0]
    y = v * V1[1] + w * V2[1]
    return np.column_stack((x, y))

def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords).
    bary: (N, 2) array of (u, v) where w = 1-u-v
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_from_bary(bary):
    """Returns the minimum absolute determinant (proportional to min area)."""
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    bary: (N, 2) array of (u, v)
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _objective_smooth(x, beta, penalty_lambda=50.0):
    """
    Smooth objective for L-BFGS-B using log-sum-exp soft-min.
    x: (2N,) array of u, v coords (flattened barycentric).
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    # Soft min: m - (1/beta) * log(sum(exp(-beta * shifted)))
    max_exp = np.max(-beta * shifted)
    log_sum_exp = max_exp + np.log(np.sum(np.exp(-beta * shifted - max_exp)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _objective_raw(x):
    """
    Objective: negative min triangle area.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    min_area = _min_area_from_bary(bary_clamped)
    return -min_area

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 25
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - np.array(selected), axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _generate_fps_initialization(rng, n_points=N, n_candidates=400):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling in barycentric coords.
    """
    points = np.zeros((n_points, 2))
    
    # Start with centroid
    points[0] = [1/3, 1/3]
    
    for k in range(1, n_points):
        # Generate random candidates in barycentric coords
        u = rng.uniform(0, 1, n_candidates)
        v = rng.uniform(0, 1, n_candidates)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        
        candidates = np.column_stack([u, v])
        
        # Calculate min distance from each candidate to existing points
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        # Pick candidate with max min_dist
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def _local_optimize_lbfgsb(x0, beta=50.0, maxiter=200):
    """Run L-BFGS-B local optimization from x0 using smooth surrogate."""
    try:
        res = minimize(
            lambda x: _objective_smooth(x, beta), x0, method='L-BFGS-B',
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, maxiter=200):
    """Run SLSQP local optimization from x0."""
    try:
        res = minimize(
            _objective_raw, x0, method='SLSQP',
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_nm(x0, maxiter=3000, xatol=1e-10, fatol=1e-12):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_raw, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _simulated_annealing(x0, rng, n_iterations=5000, t_start=0.1, t_end=1e-6, alpha=0.999):
    """
    Simulated annealing with topological swaps and local jitter.
    x0: (2N,) array of barycentric coords.
    """
    current = x0.copy()
    current_score = -_objective_raw(current)
    best = current.copy()
    best_score = current_score
    
    t = t_start
    
    for it in range(n_iterations):
        # Cooling
        t *= alpha
        
        # Generate neighbor
        neighbor = current.copy()
        op = rng.random()
        
        if op < 0.3:
            # Topological swap: swap barycentric coords of two random points
            i, j = rng.choice(N, size=2, replace=False)
            neighbor[2*i:2*i+2], neighbor[2*j:2*j+2] = neighbor[2*j:2*j+2].copy(), neighbor[2*i:2*i+2].copy()
        elif op < 0.7:
            # Local jitter: Gaussian perturbation on a random subset of points
            n_perturb = rng.randint(1, 4)
            indices = rng.choice(N, size=n_perturb, replace=False)
            for idx in indices:
                noise = rng.normal(0, 0.05, size=2)
                neighbor[2*idx:2*idx+2] += noise
        else:
            # Larger perturbation
            idx = rng.randint(N)
            noise = rng.normal(0, 0.1, size=2)
            neighbor[2*idx:2*idx+2] += noise
        
        # Project to simplex
        neighbor_bary = _project_to_bary_simplex(neighbor.reshape(N, 2))
        neighbor = neighbor_bary.flatten()
        
        # Evaluate
        neighbor_score = -_objective_raw(neighbor)
        
        # Accept or reject
        if neighbor_score > current_score or rng.random() < np.exp((neighbor_score - current_score) / max(t, 1e-10)):
            current = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. More FPS inits
    for i in range(8):
        pts = _generate_fps_initialization(rng)
        inits.append(pts.flatten())
        
    # 3. Random Inits
    for _ in range(10):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 4. Structured starts
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts).flatten())
    
    bary1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.75],
    ])
    inits.append(_project_to_bary_simplex(bary1).flatten())
    
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary2).flatten())
    
    bary3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary3).flatten())
    
    # 5. Optimize each initialization
    for init in inits:
        if time_left() < 5:
            break
            
        try:
            init_bary = _project_to_bary_simplex(init.reshape(N, 2))
            init_flat = init_bary.flatten()
            
            # Stage 1: L-BFGS-B with smooth surrogate
            x1 = _local_optimize_lbfgsb(init_flat, beta=50.0, maxiter=100)
            score1 = -_objective_raw(x1)
            
            # Stage 2: SLSQP refinement
            x2 = _local_optimize_slsqp(x1, maxiter=150)
            score2 = -_objective_raw(x2)
            
            # Stage 3: Nelder-Mead polishing
            x3 = _local_optimize_nm(x2, maxiter=2000)
            score3 = -_objective_raw(x3)
            
            if score3 > best_score:
                best_score = score3
                best_x = x3.copy()
        except Exception:
            pass
    
    # 6. Simulated annealing from best solution
    if best_x is not None and time_left() > 30:
        try:
            sa_result = _simulated_annealing(best_x, rng, n_iterations=3000, t_start=0.05, t_end=1e-7, alpha=0.9995)
            sa_score = -_objective_raw(sa_result)
            if sa_score > best_score:
                best_score = sa_score
                best_x = sa_result.copy()
        except Exception:
            pass
    
    # 7. Final polishing with Nelder-Mead
    if best_x is not None and time_left() > 15:
        try:
            x_final = _local_optimize_nm(best_x, maxiter=5000, xatol=1e-12, fatol=1e-14)
            score_final = -_objective_raw(x_final)
            if score_final > best_score:
                best_score = score_final
                best_x = x_final.copy()
        except Exception:
            pass
    
    # 8. If time permits, try a few more SA runs from perturbed best
    if best_x is not None and time_left() > 25:
        for i in range(3):
            if time_left() < 10:
                break
            try:
                noise = rng.normal(0, 0.02, size=2*N)
                perturbed = _project_to_bary_simplex((best_x + noise).reshape(N, 2)).flatten()
                sa_result = _simulated_annealing(perturbed, rng, n_iterations=2000, t_start=0.03, t_end=1e-7, alpha=0.9995)
                sa_score = -_objective_raw(sa_result)
                if sa_score > best_score:
                    best_score = sa_score
                    best_x = sa_result.copy()
            except Exception:
                pass
    
    if best_x is None:
        # Fallback: Center points
        best_x = np.zeros(2 * N)
        best_x[0::2] = 1/3
        best_x[1::2] = 1/3

    # Convert to Cartesian
    u = best_x[0::2]
    v = best_x[1::2]
    points = _bary_to_cart(u, v)
    return np.array(points)
