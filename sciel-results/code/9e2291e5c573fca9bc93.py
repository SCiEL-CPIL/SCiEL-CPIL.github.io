"""
Refined program for packing 50 equal circles in a unit circle.
Strategy:
1. Robust Soft-Min Surrogate: Uses a stabilized log-sum-exp to approximate the minimum distance constraint.
2. Simulated Annealing with Gradient Descent: Escapes local minima by accepting worse solutions probabilistically.
3. High-Precision Polish: Final stage uses very high beta or direct sub-gradient ascent to maximize the exact minimum distance.
4. Diverse Initializations: Includes ring-based and hexagonal lattices to cover different symmetry classes.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin for 120s limit

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """
    Compute the maximum radius r for which circles centered at 'centers' 
    are disjoint and inside the unit circle.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
    """
    if centers.size == 0:
        return 0.0
    n = centers.shape[0]
    
    # Pairwise distances
    # Using broadcasting for vectorization
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid zero distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    min_bound_dist = np.min(1.0 - norms)
    
    return min(min_pair_dist / 2.0, min_bound_dist)

def _soft_min_obj_and_grad(params, n, beta):
    """
    Objective: -soft_min(constraints)
    Constraints: d_ij / 2 for pairs, 1 - |c_i| for boundary.
    We want to maximize the minimum of these, so we minimize the negative soft-min.
    
    Soft min definition:
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    This is a smooth approximation of min(x_i).
    
    To improve numerical stability, we shift x by the current minimum:
    Let m = min(x).
    soft_min(x) = m - 1/beta * log(sum(exp(-beta * (x_i - m))))
    Note: exp(0) = 1, so the sum is at least 1.
    """
    centers = params.reshape(n, 2)
    
    # 1. Compute pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # 2. Compute boundary distances
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    # 3. Combine all constraints into a single vector
    # Values we want to maximize: v_k
    vals = np.concatenate([pair_dists / 2.0, bound_dists])
    
    # 4. Compute Soft Min
    m = np.min(vals)
    # Shift to prevent overflow in exp
    shifted = vals - m
    # exp(-beta * shifted)
    # Since shifted >= 0, -beta * shifted <= 0, so exp is <= 1. Safe.
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    
    # soft_min = m - (1/beta) * log(S)
    obj_val = -(m - (1.0 / beta) * np.log(S))
    
    # 5. Compute Gradient
    # d(soft_min)/dx = - (1/S) * sum_j (d(exp_j)/dx) * exp_j * (-beta) ... wait.
    # Let f(x) = -1/beta * log(sum(exp(-beta x_k)))
    # df/dx_i = -1/beta * (1/S) * sum_k (d(exp(-beta x_k))/dx_i)
    # d(exp(-beta x_k))/dx_i = -beta * exp(-beta x_k) * (dx_k/dx_i)
    # So df/dx_i = -1/beta * (1/S) * sum_k (-beta * exp_k * dx_k/dx_i)
    # df/dx_i = (1/S) * sum_k (exp_k * dx_k/dx_i)
    
    # We are minimizing obj = -soft_min, so grad_obj = -grad_soft_min
    # grad_soft_min = (1/S) * sum_k exp_k * grad(x_k)
    # grad_obj = -(1/S) * sum_k exp_k * grad(x_k)
    
    weights = exp_terms / S
    
    grad_centers = np.zeros_like(centers)
    
    # Pair contributions
    # x_k = dist(i,j) / 2
    # d(dist(i,j))/dx_i = (c_i - c_j) / dist(i,j)
    # d(dist(i,j))/dx_j = (c_j - c_i) / dist(i,j)
    # d(x_k)/dx_i = 0.5 * (c_i - c_j) / dist(i,j)
    
    pair_weights = weights[:n*(n-1)//2]
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w < 1e-10:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-8:
            vec = (centers[i] - centers[j]) / d
            # grad_obj += -w * grad(x_k)
            # grad(x_k) w.r.t c_i is 0.5 * vec
            # grad(x_k) w.r.t c_j is -0.5 * vec
            grad_centers[i] -= w * 0.5 * vec
            grad_centers[j] -= w * (-0.5 * vec) # which is + w * 0.5 * vec? 
            # Let's re-verify signs.
            # obj = -soft_min.
            # soft_min decreases if any x_k decreases.
            # We want to increase x_k.
            # Gradient of soft_min points in direction of increasing x.
            # So gradient of -soft_min points in direction of decreasing x.
            # Wait. We want to MAXIMIZE min(x).
            # So we MINIMIZE -min(x).
            # If we move in direction of -grad(-soft_min) = grad(soft_min), we increase soft_min.
            # L-BFGS minimizes. So we provide grad of obj.
            # obj = -soft_min.
            # grad_obj = - grad_soft_min.
            # grad_soft_min = sum w_k grad(x_k).
            # So grad_obj = - sum w_k grad(x_k).
            
            # For pair i,j:
            # x_k = 0.5 * ||c_i - c_j||
            # grad_{c_i} x_k = 0.5 * (c_i - c_j) / ||c_i - c_j|| = 0.5 * vec
            # grad_{c_j} x_k = -0.5 * vec
            # grad_obj[c_i] += -w * (0.5 * vec)
            # grad_obj[c_j] += -w * (-0.5 * vec) = + w * 0.5 * vec
            
            # So:
            grad_centers[i] -= w * 0.5 * vec
            grad_centers[j] += w * 0.5 * vec

    # Boundary contributions
    # x_k = 1 - ||c_i||
    # grad_{c_i} x_k = - c_i / ||c_i||
    bound_weights = weights[n*(n-1)//2:]
    for i in range(n):
        w = bound_weights[i]
        if w < 1e-10:
            continue
        norm = norms[i]
        if norm > 1e-8:
            vec = centers[i] / norm
            # grad_obj[c_i] += -w * grad(x_k) = -w * (-vec) = + w * vec
            grad_centers[i] += w * vec
            
    return obj_val, grad_centers

def _get_ring_initialization(n):
    """
    Generate a ring-based initialization.
    Approximate optimal distribution for N=50:
    Center: 1
    Ring 1: 6
    Ring 2: 12
    Ring 3: 12
    Ring 4: 9
    Ring 5: 10
    Total: 50
    """
    centers = []
    centers.append([0.0, 0.0])
    
    # Radii are approximate, optimization will fix them
    rings = [
        (6, 0.22),
        (12, 0.44),
        (12, 0.66),
        (9, 0.82),
        (10, 0.92)
    ]
    
    count = 1
    for count_in_ring, radius in rings:
        if count >= n:
            break
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
            
    return np.array(centers[:n])

def _get_hex_lattice_initialization(n):
    """
    Generate a hexagonal lattice initialization.
    """
    r_est = 0.12
    s = 2 * r_est
    points = []
    max_range = int(2.0 / s) + 3
    
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) < n:
        return _get_ring_initialization(n)
        
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        # Rejection sampling for uniform area
        while True:
            radius = rng.uniform(0, 1.0)
            if radius <= 0.95:
                break
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _optimize_with_sa(centers, start_time, max_time_frac=0.6):
    """
    Optimize using Simulated Annealing with L-BFGS-B steps.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    # Time budget for this phase
    total_time = DEADLINE
    allocated_time = max_time_frac * (DEADLINE - (time.time() - start_time))
    end_time = time.time() + allocated_time
    
    # Annealing parameters
    T_start = 0.05
    T_end = 1e-5
    cooling_rate = 0.95
    
    T = T_start
    beta = 100.0
    
    while time.time() < end_time:
        # 1. Run L-BFGS-B for a short duration or fixed iterations
        # We use a short maxiter to allow frequent acceptance checks
        try:
            res = minimize(
                _soft_min_obj_and_grad, 
                current.flatten(), 
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 20, 'gtol': 1e-5}
            )
            candidate = res.x.reshape(N, 2)
            if not np.all(np.isfinite(candidate)):
                continue
                
            # Check validity
            norms = np.linalg.norm(candidate, axis=1)
            if np.any(norms > 1.0):
                # Project back
                mask = norms > 1.0
                candidate[mask] *= 1.0 / norms[mask]
                
            score = _compute_score(candidate)
            
            # 2. Simulated Annealing Acceptance
            if score > best_score:
                best_score = score
                best_centers = candidate.copy()
                current = candidate.copy()
            else:
                # Accept with probability exp((score - best_score) / T)
                # Note: score is radius, higher is better.
                # Delta = score - best_score (negative)
                delta = score - best_score
                if delta > 0:
                    current = candidate.copy()
                else:
                    prob = np.exp(delta / T)
                    if np.random.rand() < prob:
                        current = candidate.copy()
                        
        except Exception:
            pass
            
        # Update Temperature and Beta
        T *= cooling_rate
        beta *= 1.1 # Increase sharpness slightly to focus on min
        
        if T < T_end:
            break
            
    return best_centers, best_score

def _final_polish(centers, start_time):
    """
    High-precision polish using very high beta soft-min or sub-gradient.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    # Use very high beta to approximate hard min
    betas = [500, 2000, 10000, 50000]
    
    for beta in betas:
        if time.time() > DEADLINE - 2:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad, 
                current.flatten(), 
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            candidate = res.x.reshape(N, 2)
            if np.all(np.isfinite(candidate)):
                score = _compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_centers = candidate.copy()
                    current = candidate
        except Exception:
            pass
            
    # Sub-gradient polish for final nudge
    if time.time() < DEADLINE - 1:
        current = best_centers.copy()
        step = 1e-4
        for _ in range(500):
            if time.time() > DEADLINE - 1:
                break
            n = len(current)
            diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            norms = np.linalg.norm(current, axis=1)
            min_bound_dist = np.min(1.0 - norms)
            
            if min_pair_dist / 2.0 < min_bound_dist:
                i, j = np.unravel_index(np.argmin(dists), dists.shape)
                if i > j: i, j = j, i
                d = dists[i, j]
                if d > 1e-8:
                    vec = (current[i] - current[j]) / d
                    current[i] += vec * step * 0.5
                    current[j] -= vec * step * 0.5
            else:
                i = np.argmin(1.0 - norms)
                norm = norms[i]
                if norm > 1e-8:
                    vec = current[i] / norm
                    current[i] -= vec * step
                    
            # Safety clamp
            new_norms = np.linalg.norm(current, axis=1)
            mask = new_norms > 0.999
            if np.any(mask):
                current[mask] *= 0.999 / new_norms[mask]
                
            score = _compute_score(current)
            if score > best_score:
                best_score = score
                best_centers = current.copy()
                
    return best_centers, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Diverse initializations
    initializations = [
        _get_ring_initialization(N),
        _get_hex_lattice_initialization(N),
        _get_random_initialization(N, seed=1),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=99),
    ]
    
    # Optimize each initialization
    for init in initializations:
        if time.time() - start_time > DEADLINE - 30:
            break
            
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
            
        # Run SA optimization
        optimized, score = _optimize_with_sa(init, start_time, max_time_frac=0.4)
        if score > best_score:
            best_score = score
            best_centers = optimized.copy()
            
    if best_centers is None:
        return initializations[0]
        
    # Final Polish on the best result
    final_centers, final_score = _final_polish(best_centers, start_time)
    if final_score > best_score:
        best_score = final_score
        best_centers = final_centers
        
    return best_centers

# EVOLVE-BLOCK-END
