"""Packing 50 equal circles in unit circle: Soft-min surrogate + L-BFGS + Active Set Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
# Safety margin: Evaluator kills at 150s. We aim to finish by 135s.
DEADLINE = 135.0

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """Compute the maximum radius r for which circles are disjoint and inside unit circle."""
    n = centers.shape[0]
    # Vectorized pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _soft_min_obj_and_grad(params, n, beta):
    """
    Objective: -soft_min(constraints)
    Constraints: pair_dists/2 and boundary_dists.
    Gradient is computed analytically.
    """
    centers = params.reshape(n, 2)

    # 1. Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # 2. Boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms

    # Combine constraints
    # Pair constraints are dist/2. Boundary constraints are 1-norm.
    vals = np.concatenate([pair_dists / 2.0, bound_dists])

    # Soft min calculation: -log(sum(exp(-beta * vals))) / beta
    # To avoid overflow/underflow, subtract min(vals)
    m = np.min(vals)
    shifted = vals - m
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    weights = exp_terms / S
    
    # Objective value: -(m - (1/beta)*log(S))
    # We want to maximize min, so we minimize -min.
    obj_val = -(m - (1.0 / beta) * np.log(S))

    # Gradient
    grad = np.zeros_like(centers)
    n_pairs = n * (n - 1) // 2

    # Pair contributions
    # grad(c_i) -= w_i * 0.5 * (c_i - c_j) / d_ij
    # grad(c_j) += w_i * 0.5 * (c_i - c_j) / d_ij
    pair_weights = weights[:n_pairs]
    # Filter out negligible weights to save time
    mask = pair_weights > 1e-10
    if np.any(mask):
        idx = np.where(mask)[0]
        ii = i_idx[idx]
        jj = j_idx[idx]
        dd = pair_dists[idx]
        
        # Avoid division by zero
        valid = dd > 1e-8
        if np.any(valid):
            ii_v = ii[valid]
            jj_v = jj[valid]
            dd_v = dd[valid]
            ww_v = pair_weights[idx[valid]]
            
            vecs = (centers[ii_v] - centers[jj_v]) / dd_v[:, np.newaxis]
            contrib = ww_v[:, np.newaxis] * vecs * 0.5
            
            np.add.at(grad, ii_v, -contrib)
            np.add.at(grad, jj_v, contrib)

    # Boundary contributions
    # grad(c_i) += w_i * (c_i / |c_i|)  <-- Wait, grad of (1-|c|) is -c/|c|.
    # We are minimizing -soft_min. 
    # d/dc (1 - |c|) = -c/|c|.
    # d/dc (-soft_min) = ... 
    # Let's re-derive carefully.
    # f = - (m - 1/beta log S)
    # df/dx = - d/dx (m - 1/beta log S)
    # d/dx log S = d/dx log(sum exp(-beta v_k)) = sum exp(-beta v_k) (-beta dv_k/dx) / S
    #            = -beta sum w_k dv_k/dx
    # d/dx (m) = 0 (m is min, subgradient, but in soft min it's smoothed)
    # Actually, the derivative of the soft min approximation is:
    # d/dx [ -1/beta log(sum exp(-beta v_k)) ] = sum w_k dv_k/dx
    # So obj = - soft_min. 
    # grad_obj = - sum w_k grad(v_k)
    
    # v_pair = d_ij / 2. grad(v_pair) wrt c_i = 0.5 * (c_i-c_j)/d_ij.
    # v_bound = 1 - |c_i|. grad(v_bound) wrt c_i = -c_i/|c_i|.
    
    # So grad_obj[c_i] = - sum_w_pair * 0.5 * (c_i-c_j)/d_ij  - sum_w_bound * (-c_i/|c_i|)
    #                   = - sum_w_pair * 0.5 * (c_i-c_j)/d_ij  + sum_w_bound * c_i/|c_i|
    
    # My previous code had:
    # grad[idx_v] += (w_v / norm_v)[:, np.newaxis] * centers[idx_v]
    # This is sum w * c/|c|. This matches the boundary part.
    
    # For pairs:
    # grad[ii] -= w * 0.5 * vec
    # grad[jj] += w * 0.5 * vec
    # This matches.
    
    bound_weights = weights[n_pairs:]
    mask_b = bound_weights > 1e-10
    if np.any(mask_b):
        idx_b = np.where(mask_b)[0]
        valid_b = norms[idx_b] > 1e-8
        if np.any(valid_b):
            idx_v = idx_b[valid_b]
            w_v = bound_weights[idx_v]
            norm_v = norms[idx_v]
            # grad += w * c / |c|
            grad[idx_v] += (w_v / norm_v)[:, np.newaxis] * centers[idx_v]

    return obj_val, grad

def _optimize_lbfgs(centers, beta, maxiter=200, gtol=1e-8):
    """Run L-BFGS-B optimization."""
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            centers.flatten(),
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-12}
        )
        if np.all(np.isfinite(res.x)):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return centers

def _get_ring_init(n):
    """
    Initialize based on the known optimal structure: 1+6+12+18+13.
    Radii are estimated to maximize local packing.
    """
    centers = [[0.0, 0.0]]
    # Ring 1: 6 circles
    r1 = 0.25
    for i in range(6):
        a = 2 * np.pi * i / 6
        centers.append([r1 * np.cos(a), r1 * np.sin(a)])
    # Ring 2: 12 circles
    r2 = 0.45
    for i in range(12):
        a = 2 * np.pi * i / 12
        centers.append([r2 * np.cos(a), r2 * np.sin(a)])
    # Ring 3: 18 circles
    r3 = 0.65
    for i in range(18):
        a = 2 * np.pi * i / 18
        centers.append([r3 * np.cos(a), r3 * np.sin(a)])
    # Ring 4: 13 circles (50-1-6-12-18 = 13)
    r4 = 0.85
    for i in range(13):
        a = 2 * np.pi * i / 13
        centers.append([r4 * np.cos(a), r4 * np.sin(a)])
        
    return np.array(centers[:n])

def _get_random_init(n, seed):
    """Random initialization inside unit circle."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(centers)

def _subgradient_polish(centers, start_time, max_iters=200, step_decay=0.99):
    """
    Direct sub-gradient ascent on the exact min objective.
    Useful for final precision when soft-min is too smooth or too sharp.
    """
    current = centers.copy()
    n = current.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=1)
    best_score = _compute_score(current)
    step = 0.001
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 2:
            break

        # Compute constraints
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        pair_dists = dists[i_idx, j_idx]
        norms = np.sqrt(np.sum(current * current, axis=1))
        bound_dists = 1.0 - norms

        vals = np.concatenate([pair_dists / 2.0, bound_dists])
        min_val = np.min(vals)
        
        # Find active constraints (within tolerance)
        tol = 1e-6
        active = np.where(np.abs(vals - min_val) < tol)[0]
        
        if len(active) == 0:
            break

        # Compute sub-gradient: average of gradients of active constraints
        grad = np.zeros_like(current)
        n_pairs = n * (n - 1) // 2

        for a in active:
            if a < n_pairs:
                i, j = i_idx[a], j_idx[a]
                d = dists[i, j]
                if d > 1e-10:
                    vec = (current[i] - current[j]) / d
                    grad[i] += 0.5 * vec
                    grad[j] -= 0.5 * vec
            else:
                i = a - n_pairs
                norm = norms[i]
                if norm > 1e-10:
                    vec = current[i] / norm
                    # Gradient of (1-|c|) is -c/|c|.
                    # We are maximizing min, so we move in direction of gradient of the min function.
                    # Subgradient of min is in convex hull of gradients of active functions.
                    # We want to increase the minimum.
                    # If boundary is active, we want to decrease norm. Grad of (1-norm) is -c/|c|.
                    # So we move in direction -c/|c|.
                    grad[i] -= vec

        # Normalize
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        grad /= gnorm

        # Update
        current += step * grad

        # Clamp to container
        norms = np.sqrt(np.sum(current * current, axis=1))
        mask = norms > 0.999
        if np.any(mask):
            current[mask] *= 0.999 / norms[mask]

        # Check improvement
        r = _compute_score(current)
        if r > best_score:
            best_score = r
            step = max(step * 1.01, 0.005) # Slightly increase step if improving
        else:
            step *= step_decay
            if step < 1e-6:
                break

    return current, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None

    # Initializations
    inits = [
        _get_ring_init(N),
        _get_random_init(N, 1),
        _get_random_init(N, 42),
        _get_random_init(N, 123),
        _get_random_init(N, 999),
    ]

    # Phase 1: Optimization with moderate beta
    # We use beta=100 initially to get a good shape
    for init in inits:
        if time.time() - start_time > DEADLINE - 60:
            break
        
        # Ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.98 / norms[mask]
            
        # Optimize
        opt_centers = _optimize_lbfgs(init, beta=100.0, maxiter=150, gtol=1e-6)
        if np.all(np.isfinite(opt_centers)):
            r = _compute_score(opt_centers)
            if r > best_score:
                best_score = r
                best_centers = opt_centers.copy()

    if best_centers is None:
        return inits[0]

    # Phase 2: Progressive Beta Polish
    # Increase beta to sharpen the soft-min towards the true min
    betas = [200, 500, 1000, 2000, 5000]
    current = best_centers
    current_score = best_score
    
    for beta in betas:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        polished = _optimize_lbfgs(current, beta=beta, maxiter=100, gtol=1e-8)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > current_score:
                current_score = r
                current = polished.copy()

    # Phase 3: Sub-gradient polish for final precision
    if time.time() - start_time < DEADLINE - 5:
        current, current_score = _subgradient_polish(current, start_time, max_iters=100)

    # Final validation
    if best_centers is None:
        return inits[0]
        
    # Ensure strict validity
    norms = np.linalg.norm(current, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        current = current.copy()
        current[mask] *= 0.9999 / norms[mask]
        
    return current

# EVOLVE-BLOCK-END
