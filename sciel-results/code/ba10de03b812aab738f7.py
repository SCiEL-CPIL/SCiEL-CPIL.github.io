import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(centers):
    """Compute the radius score: min of pairwise distances/2 and edge distances."""
    n = len(centers)
    edge_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    min_edge = np.min(edge_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_edge, min_pair)

def _smooth_obj(centers_flat, beta):
    """
    Smooth surrogate: maximize the soft-min of all constraints.
    We minimize -soft_min, where soft_min approximates min(constraints).
    
    Constraints: for each pair (i,j), d_ij/2 >= r
                for each point i, edge_dist_i >= r
    
    We use: soft_min(x) = -log(sum(exp(-x_i / beta))) * beta
    This is smooth and approximates min(x) as beta -> 0.
    """
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    
    # Edge constraints: edge_dist_i for each i
    edge_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    
    # Pairwise constraints: d_ij / 2 for each pair i<j
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Get upper triangle
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Combine all constraints into one array
    all_constraints = np.concatenate([edge_dists, pair_dists])
    
    # Soft-min: -beta * log(sum(exp(-c / beta)))
    # For numerical stability, subtract max
    max_c = np.max(all_constraints)
    shifted = all_constraints - max_c
    log_sum = np.log(np.sum(np.exp(-shifted / beta)))
    soft_min = max_c - beta * log_sum
    
    return -soft_min  # We minimize this (i.e., maximize soft_min)

def _smooth_grad(centers_flat, beta):
    """Gradient of the smooth objective."""
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    
    # Edge constraints
    edge_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    
    all_constraints = np.concatenate([edge_dists, pair_dists])
    
    # Soft-min weights: w_k = exp(-c_k/beta) / sum(exp(-c_j/beta))
    max_c = np.max(all_constraints)
    shifted = all_constraints - max_c
    exp_neg = np.exp(-shifted / beta)
    sum_exp = np.sum(exp_neg)
    weights = exp_neg / sum_exp  # shape (n + n_pairs,)
    
    grad = np.zeros_like(centers_flat)
    
    # Edge gradient contributions
    n_edge = n
    edge_weights = weights[:n_edge]
    
    for i in range(n):
        if edge_weights[i] < 1e-15:
            continue
        d_left = 0.5 - centers[i, 0]
        d_right = centers[i, 0] + 0.5
        d_bottom = 0.5 - centers[i, 1]
        d_top = centers[i, 1] + 0.5
        min_d = min(d_left, d_right, d_bottom, d_top)
        
        # Gradient of edge_dist_i w.r.t. centers[i]
        # edge_dist_i = min(d_left, d_right, d_bottom, d_top)
        # Approximate gradient: average of active edge gradients
        g = np.zeros(2)
        if d_left <= min_d + 1e-12:
            g[0] += -1.0
        if d_right <= min_d + 1e-12:
            g[0] += 1.0
        if d_bottom <= min_d + 1e-12:
            g[1] += -1.0
        if d_top <= min_d + 1e-12:
            g[1] += 1.0
        norm_g = np.linalg.norm(g)
        if norm_g > 0:
            g /= norm_g
        
        grad[2*i:2*i+2] += edge_weights[i] * g
    
    # Pairwise gradient contributions
    pair_weights = weights[n_edge:]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if pair_weights[idx] < 1e-15:
                idx += 1
                continue
            vec = centers[i] - centers[j]
            d = np.linalg.norm(vec)
            if d > 1e-15:
                # Gradient of d/2 w.r.t. centers[i] = vec / (2*d)
                # Gradient of d/2 w.r.t. centers[j] = -vec / (2*d)
                g = vec / (2.0 * d)
                grad[2*i:2*i+2] += pair_weights[idx] * g
                grad[2*j:2*j+2] -= pair_weights[idx] * g
            idx += 1
    
    return -grad  # Gradient of the objective (we're minimizing -soft_min)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s
    
    rng = np.random.RandomState(42)
    
    # Initial guess: Hexagonal lattice
    s = 1.0
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    init_positions = points[indices].copy()
    
    max_abs = np.max(np.abs(init_positions))
    if max_abs > 0:
        scale = 0.49 / max_abs
        init_positions = init_positions * scale
    
    best_centers = init_positions.copy()
    best_score = _score(best_centers)
    
    # Annealed smooth optimization with multiple restarts
    # Start with large beta (smooth), decrease to small beta (sharp)
    
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        if restart == 0:
            centers = best_centers.copy()
        elif restart == 1:
            jitter = 0.02
            centers = best_centers + rng.uniform(-jitter, jitter, size=best_centers.shape)
            centers = np.clip(centers, -0.499, 0.499)
        elif restart == 2:
            # Square grid alternative
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            centers = grid[:N].copy()
        elif restart == 3:
            # Hex grid with different phase
            s2 = 1.0
            points2 = []
            for j in range(-10, 11):
                for i in range(-10, 11):
                    x = i * s2 + (j % 2) * s2 / 2.0 + 0.3
                    y = j * s2 * np.sqrt(3.0) / 2.0 + 0.2
                    points2.append([x, y])
            points2 = np.array(points2)
            d2c = np.sum(points2**2, axis=1)
            idx2 = np.argsort(d2c)[:N]
            centers = points2[idx2].copy()
            ma = np.max(np.abs(centers))
            if ma > 0:
                centers = centers * (0.49 / ma)
        else:
            jitter = 0.04 / (restart + 1)
            centers = best_centers + rng.uniform(-jitter, jitter, size=best_centers.shape)
            centers = np.clip(centers, -0.499, 0.499)
        
        # Anneal beta from large to small
        betas = [0.05, 0.02, 0.008, 0.003, 0.001, 0.0003]
        
        current_centers = centers.copy()
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            
            try:
                result = minimize(
                    _smooth_obj,
                    current_centers.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.49999, 0.49999)] * (2*N),
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50},
                    jac=lambda x: _smooth_grad(x, beta),
                    args=(beta,)
                )
                
                new_score = _score(result.x.reshape(-1, 2))
                if new_score > best_score:
                    best_score = new_score
                    best_centers = result.x.reshape(-1, 2).copy()
                
                current_centers = result.x.reshape(-1, 2).copy()
            except Exception:
                pass
        
        # If this restart improved, use it as base for next
        if time.time() > deadline - 8.0:
            break
    
    # Final high-precision polish with very small beta
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                _smooth_obj,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.499999, 0.499999)] * (2*N),
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100},
                jac=lambda x: _smooth_grad(x, 0.0001),
                args=(0.0001,)
            )
            new_score = _score(result.x.reshape(-1, 2))
            if new_score > best_score:
                best_score = new_score
                best_centers = result.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    # One more polish with even tighter settings
    if time.time() < deadline - 1.0:
        try:
            result = minimize(
                _smooth_obj,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999999, 0.4999999)] * (2*N),
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100},
                jac=lambda x: _smooth_grad(x, 0.00003),
                args=(0.00003,)
            )
            new_score = _score(result.x.reshape(-1, 2))
            if new_score > best_score:
                best_score = new_score
                best_centers = result.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    # Ensure strict validity
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
