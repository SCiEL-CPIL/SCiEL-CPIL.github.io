import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial configuration: Hexagonal lattice
    def get_hex_init():
        # Generate a hex grid covering the unit disk
        spacing = 0.2
        points = []
        # Iterate over a grid large enough
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - spacing/2:
                    points.append([x, y])
        
        points = np.array(points)
        # Sort by distance to origin to pick the most central ones
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        
        # We need exactly N points
        if len(points) >= N:
            return points[indices[:N]]
        else:
            # Fallback: random points inside circle
            return np.random.rand(N, 2) * 2 - 1 # then filter? No, just use rings if this fails
            # Actually, let's just use a simple ring structure if hex fails or is sparse
            return get_ring_init()

    def get_ring_init():
        # 1 center, then rings of 6, 12, 18, ...
        centers = [[0.0, 0.0]]
        count = 1
        radius = 0.1
        while count < N:
            # Determine how many in this ring
            # Max circles in ring of radius r is approx 2*pi*r / (2*spacing)
            # Let's just add 6, 12, 18...
            if len(centers) == 1:
                num_in_ring = 6
            else:
                num_in_ring = 6 * len(centers) # This grows too fast? 
                # Standard hex packing: 1, 6, 12, 18, 24...
                # Let's stick to 6, 12, 18, 24...
                # Actually, just use 6 * (ring_index)
                pass
            
            # Simpler: just place points on a circle of radius r
            # We need to fit them.
            # Let's use a fixed set of radii
            pass
        
        # Let's use a simpler deterministic init:
        # 1 center
        # 6 at r=0.2
        # 12 at r=0.4
        # 18 at r=0.6
        # 13 at r=0.8
        pts = []
        pts.append([0.0, 0.0])
        for r, n in zip([0.2, 0.4, 0.6, 0.8], [6, 12, 18, 13]):
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            for a in angles:
                pts.append([r*np.cos(a), r*np.sin(a)])
        return np.array(pts[:N])

    x0 = get_hex_init()
    # Ensure x0 is valid (inside unit circle)
    # Add small random perturbation to break symmetry
    x0 = x0 + np.random.normal(0, 0.01, x0.shape)
    # Project back inside if necessary
    norms = np.linalg.norm(x0, axis=1)
    for i in range(N):
        if norms[i] > 0.9:
            x0[i] = x0[i] * (0.9 / norms[i])

    def objective_and_grad(flat_x):
        x = flat_x.reshape(N, 2)
        
        # Distances between all pairs
        # d_ij
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        # Zero out diagonal
        np.fill_diagonal(d, np.inf)
        
        # Distances to boundary
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        
        # Soft min approximation
        # We want to minimize max(-d_ij, -d_ib)
        # Which is equivalent to minimizing -min(d_ij, d_ib)
        # Using log-sum-exp on negative distances
        beta = 100.0
        
        # Sum of exp(-beta * d) for all pairs and boundaries
        # d is NxN, take upper triangle
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        d_bnds = dist_boundary
        
        # Combine all distances
        all_d = np.concatenate([d_pairs, d_bnds])
        
        # Compute log-sum-exp
        # L = (1/beta) * log( sum(exp(-beta * all_d)) )
        # To avoid overflow/underflow, use max trick
        max_neg_d = -np.min(all_d) # max of -d is -min(d)
        # Actually, we want to minimize the "badness".
        # The objective is approximating the minimum distance.
        # min(all_d) is what we want to maximize.
        # So we want to minimize -min(all_d).
        # Let's just minimize the sum of squared violations? No, soft-min is better for max-min.
        
        # Let's use the standard soft-min:
        # soft_min(z) = -1/beta * log(sum(exp(-beta*z)))
        # We want to maximize soft_min(all_d)
        # So we minimize -soft_min(all_d) = 1/beta * log(sum(exp(-beta*all_d)))
        
        # Numerical stability
        # exp(-beta * all_d)
        # If all_d is small (bad), exp is large.
        # Let's shift by min(all_d)
        min_d = np.min(all_d)
        # If min_d is negative (overlap), exp(-beta*neg) = exp(pos) -> large. Good.
        
        # To prevent overflow, subtract max exponent
        exponents = -beta * all_d
        max_exp = np.max(exponents)
        
        if np.isfinite(max_exp):
            sum_exp = np.sum(np.exp(exponents - max_exp))
            log_sum = max_exp + np.log(sum_exp)
            obj = log_sum / beta
        else:
            obj = 100.0 # Large penalty
        
        # Gradient
        # d(obj)/d(x_k)
        # obj = (1/beta) * log( sum_j exp(-beta * d_j) )
        # d(obj)/d(x_k) = (1/beta) * (1/S) * sum_j (-beta * exp(-beta*d_j)) * d(d_j)/d(x_k)
        # = - (1/S) * sum_j exp(-beta*d_j) * d(d_j)/d(x_k)
        # where S = sum exp(-beta*d_j)
        
        # We need d(d_j)/d(x_k)
        # This is computationally heavy to compute analytically for all pairs in a loop.
        # We can approximate or use automatic differentiation? No, manual is faster.
        
        # Let's use a simpler gradient:
        # The gradient of the soft-min is a weighted sum of the gradients of the individual distances.
        # weights w_j = exp(-beta * d_j) / S
        # grad_obj = - sum_j w_j * grad(d_j)
        
        # Calculate weights
        weights = np.exp(exponents - max_exp) / sum_exp
        
        # Separate weights for pairs and boundaries
        n_pairs = len(d_pairs)
        w_pairs = weights[:n_pairs]
        w_bnds = weights[n_pairs:]
        
        grad = np.zeros_like(x)
        
        # Pair contributions
        # For each pair (i, j), d_ij = ||x_i - x_j||
        # d(d_ij)/d(x_i) = (x_i - x_j) / d_ij
        # d(d_ij)/d(x_j) = (x_j - x_i) / d_ij
        # Contribution to grad[i] += -w_ij * (x_i - x_j)/d_ij
        # Contribution to grad[j] += -w_ij * (x_j - x_i)/d_ij
        
        # Vectorized update
        # We have w_pairs corresponding to iu indices
        i_idx, j_idx = iu
        # d_pairs is the distance for these pairs
        # Avoid division by zero
        d_safe = np.where(d_pairs > 1e-8, d_pairs, 1e-8)
        
        # Vector (x_i - x_j)
        vecs = x[i_idx] - x[j_idx]
        # Normalized vectors
        norm_vecs = vecs / d_safe[:, np.newaxis]
        
        # Gradient contribution for i: -w * norm_vec
        # Gradient contribution for j: +w * norm_vec
        
        # Use np.add.at for accumulation
        np.add.at(grad, i_idx, -w_pairs[:, np.newaxis] * norm_vecs)
        np.add.at(grad, j_idx, w_pairs[:, np.newaxis] * norm_vecs)
        
        # Boundary contributions
        # d_bnds = 1 - ||x_i||
        # d(d_bi)/d(x_i) = -x_i / ||x_i||
        # Contribution to grad[i] += -w_bi * (-x_i / ||x_i||) = w_bi * x_i / ||x_i||
        
        norms = np.linalg.norm(x, axis=1)
        norms_safe = np.where(norms > 1e-8, norms, 1e-8)
        norm_x = x / norms_safe[:, np.newaxis]
        
        np.add.at(grad, np.arange(N), w_bnds[:, np.newaxis] * norm_x)
        
        return obj, grad

    # Optimization loop
    best_x = x0
    best_score = -np.inf
    
    # Calculate initial score (raw min distance)
    def calc_score(x):
        d = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
        return min(min_pair, min_bound)

    best_score = calc_score(x0)
    
    # Run minimization
    # Use L-BFGS-B
    # We might need multiple restarts or increasing beta
    betas = [20, 50, 100, 200]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Redefine objective with current beta
        def obj_func(flat_x, beta_val=beta):
            # Re-implementing the logic inside to capture beta
            x = flat_x.reshape(N, 2)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
            
            iu = np.triu_indices(N, k=1)
            d_pairs = d[iu]
            d_bnds = dist_boundary
            all_d = np.concatenate([d_pairs, d_bnds])
            
            exponents = -beta_val * all_d
            max_exp = np.max(exponents)
            
            if not np.isfinite(max_exp):
                return 100.0, np.zeros_like(flat_x)
            
            sum_exp = np.sum(np.exp(exponents - max_exp))
            log_sum = max_exp + np.log(sum_exp)
            obj = log_sum / beta_val
            
            weights = np.exp(exponents - max_exp) / sum_exp
            grad = np.zeros_like(x)
            
            n_pairs = len(d_pairs)
            w_pairs = weights[:n_pairs]
            w_bnds = weights[n_pairs:]
            
            i_idx, j_idx = iu
            d_safe = np.where(d_pairs > 1e-8, d_pairs, 1e-8)
            vecs = x[i_idx] - x[j_idx]
            norm_vecs = vecs / d_safe[:, np.newaxis]
            
            np.add.at(grad, i_idx, -w_pairs[:, np.newaxis] * norm_vecs)
            np.add.at(grad, j_idx, w_pairs[:, np.newaxis] * norm_vecs)
            
            norms = np.linalg.norm(x, axis=1)
            norms_safe = np.where(norms > 1e-8, norms, 1e-8)
            norm_x = x / norms_safe[:, np.newaxis]
            
            np.add.at(grad, np.arange(N), w_bnds[:, np.newaxis] * norm_x)
            
            return obj, grad.flatten()

        res = minimize(obj_func, current_x.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.x is not None:
            new_x = res.x.reshape(N, 2)
            new_score = calc_score(new_x)
            if new_score > best_score:
                best_score = new_score
                best_x = new_x
            current_x = new_x

    # Final polish: Nelder-Mead for last few steps if time permits
    if time.time() < deadline - 10:
        def nm_obj(flat_x):
            x = flat_x.reshape(N, 2)
            d = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            min_pair = np.min(d) / 2.0
            min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
            return -min(min_pair, min_bound) # Minimize negative of score
            
        res_nm = minimize(nm_obj, best_x.flatten(), method='Nelder-Mead', 
                          options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
        if res_nm.x is not None:
            final_x = res_nm.x.reshape(N, 2)
            final_score = calc_score(final_x)
            if final_score > best_score:
                best_x = final_x
                best_score = final_score

    return best_x
