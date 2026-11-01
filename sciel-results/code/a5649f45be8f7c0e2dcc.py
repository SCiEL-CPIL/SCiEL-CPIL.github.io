import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # 1. Initial Radial Ring Configuration
    # Place points on concentric rings for better initial packing
    def radial_ring_init(n_points):
        """Create initial configuration with points on concentric rings."""
        points = []
        # Center point
        points.append([0.0, 0.0])
        n_remaining = n_points - 1
        
        # Distribute remaining points on rings
        ring_indices = 1
        while n_remaining > 0:
            # Number of points on this ring
            n_on_ring = min(n_remaining, 6 * ring_indices)
            n_remaining -= n_on_ring
            
            # Ring radius (roughly proportional to sqrt of ring index for even density)
            ring_radius = 0.95 * np.sqrt(ring_indices / (np.ceil(n_points / 6.0)))
            ring_radius = min(ring_radius, 0.98)
            
            for k in range(n_on_ring):
                angle = 2.0 * np.pi * k / n_on_ring
                x = ring_radius * np.cos(angle)
                y = ring_radius * np.sin(angle)
                points.append([x, y])
            
            ring_indices += 1
        
        return np.array(points[:n_points])
    
    # Also try hexagonal lattice as alternative
    def hex_lattice_init(n_points):
        """Create initial hexagonal lattice configuration."""
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            radius = np.hypot(pts[:, 0], pts[:, 1])
            pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(pts) >= n_points:
                return pts[np.argsort(radius, kind="stable")[:n_points]].copy()
            spacing *= 0.98
    
    # Choose better initial configuration
    x0_ring = radial_ring_init(N)
    x0_hex = hex_lattice_init(N)
    
    # Score both initial configs
    def get_radius_candidates(x):
        x = x.reshape(-1, 2)
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        d_pair = pdist(x) / 2.0
        return np.concatenate([dist_boundary, d_pair])
    
    score_ring = np.min(get_radius_candidates(x0_ring))
    score_hex = np.min(get_radius_candidates(x0_hex))
    
    best_x = x0_ring.copy() if score_ring >= score_hex else x0_hex.copy()
    best_raw_score = max(score_ring, score_hex)
    
    # 2. Soft-min objective with analytical gradient
    def soft_min_obj_and_grad(x, beta):
        """
        Returns (objective, gradient) for soft-min objective.
        We minimize f(x) = (1/beta) * log(sum(exp(-beta * d_i)))
        where d_i are the radius candidates (boundary distances and half pairwise distances).
        """
        x = x.reshape(-1, 2)
        
        # Compute all radius candidates
        norms = np.linalg.norm(x, axis=1)
        d_boundary = 1.0 - norms
        
        # Pairwise distances
        d_pair = pdist(x) / 2.0
        
        d = np.concatenate([d_boundary, d_pair])
        
        # Clip to avoid numerical issues
        d = np.maximum(d, 1e-12)
        
        # Compute log-sum-exp for stability
        A = -beta * d
        max_A = np.max(A)
        exp_A = np.exp(A - max_A)
        sum_exp = np.sum(exp_A)
        
        if sum_exp == 0 or not np.isfinite(sum_exp):
            return 1e10, np.zeros(2 * N)
        
        log_sum_exp = max_A + np.log(sum_exp)
        obj = log_sum_exp / beta
        
        # Gradient computation
        # d(obj)/d(x) = (1/beta) * d(log(sum(exp(A)))/d(x)
        # = (1/beta) * sum_j [exp(A_j) * d(A_j)/d(x)] / sum(exp(A_j))
        # = (1/beta) * sum_j [w_j * d(A_j)/d(x)]
        # where w_j = exp(A_j) / sum(exp(A_j))
        
        w = exp_A / sum_exp
        
        # Gradient with respect to x
        grad = np.zeros((N, 2))
        
        # Boundary terms: d_i = 1 - ||x_i||, A_i = -beta * d_i
        # d(A_i)/d(x_i) = -beta * d(d_i)/d(x_i) = -beta * (-x_i / ||x_i||) = beta * x_i / ||x_i||
        # So contribution: w_i * beta * x_i / ||x_i|| * (1/beta) = w_i * x_i / ||x_i||
        
        for i in range(N):
            if norms[i] > 1e-10:
                grad[i] += w[i] * x[i] / norms[i]
        
        # Pairwise terms: d_ij = ||x_i - x_j|| / 2, A_ij = -beta * d_ij
        # d(A_ij)/d(x_i) = -beta * (x_i - x_j) / (2 * ||x_i - x_j||)
        # d(A_ij)/d(x_j) = -beta * (x_j - x_i) / (2 * ||x_i - x_j||) = beta * (x_i - x_j) / (2 * ||x_i - x_j||)
        # Contribution to grad[i]: w_ij * (1/beta) * (-beta) * (x_i - x_j) / (2 * ||x_i - x_j||) = -w_ij * (x_i - x_j) / (2 * ||x_i - x_j||)
        # Contribution to grad[j]: w_ij * (1/beta) * beta * (x_i - x_j) / (2 * ||x_i - x_j||) = w_ij * (x_i - x_j) / (2 * ||x_i - x_j||)
        
        # Convert pairwise distance indices to (i, j) pairs
        # pdist uses condensed form: index k corresponds to pair (i, j) where
        # i = k // (2*N - 1 - k // (2*N - 1)) ... actually let's just compute directly
        
        # For efficiency, we'll compute the gradient contribution from pairs
        # This is O(N^2) which is fine for N=300
        
        pair_idx = N  # start index for pairwise terms in d array
        for i in range(N):
            for j in range(i + 1, N):
                pair_pos = pair_idx + (i * (2 * N - 1 - i) // 2 + (j - i - 1))
                # Actually, let's use a simpler indexing
                # pdist condensed index for (i,j) with i<j:
                # idx = i*(2*N-1-i)//2 + (j-i-1)
                idx = i * (2 * N - 1 - i) // 2 + (j - i - 1)
                w_val = w[N + idx]
                
                diff = x[i] - x[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    grad[i] -= w_val * diff / (2.0 * dist)
                    grad[j] += w_val * diff / (2.0 * dist)
        
        grad_flat = grad.flatten()
        
        # Check for non-finite values
        if not np.all(np.isfinite(grad_flat)):
            grad_flat = np.zeros(2 * N)
        
        return obj, grad_flat
    
    # 3. Annealing loop with increasing beta
    betas = [5, 10, 20, 50, 100, 200, 400]
    current_x = best_x.copy()
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        try:
            # Use L-BFGS-B with analytical gradient
            res = minimize(soft_min_obj_and_grad, current_x, args=(beta,),
                           method='L-BFGS-B', bounds=bounds, jac=True,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000})
            
            if res.success or res.fun < soft_min_obj_and_grad(current_x, beta)[0]:
                current_x = res.x
        except Exception:
            pass
        
        # Update best raw score
        raw_s = np.min(get_radius_candidates(current_x))
        if raw_s > best_raw_score:
            best_raw_score = raw_s
            best_x = current_x.copy()
        
        # Early termination if we've hit a good score
        if best_raw_score > 0.05315:
            break
    
    # 4. Final high-precision polish with Nelder-Mead
    if time.time() < deadline - 10:
        def neg_raw_score(x):
            x = x.reshape(-1, 2)
            d = get_radius_candidates(x)
            return -np.min(d)
        
        try:
            res = minimize(neg_raw_score, best_x, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < -best_raw_score:
                best_x = res.x
                best_raw_score = -res.fun
        except Exception:
            pass
    
    # 5. Another polish pass with tighter tolerance if time allows
    if time.time() < deadline - 5:
        def neg_raw_score_tight(x):
            x = x.reshape(-1, 2)
            d = get_radius_candidates(x)
            return -np.min(d)
        
        try:
            res = minimize(neg_raw_score_tight, best_x, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            if res.fun < -best_raw_score:
                best_x = res.x
                best_raw_score = -res.fun
        except Exception:
            pass
    
    # Final validation
    best_x = best_x.reshape(-1, 2)
    norms = np.linalg.norm(best_x, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_x[mask] *= (1.0 / norms[mask] - 0.001)[:, np.newaxis]
    
    return best_x
