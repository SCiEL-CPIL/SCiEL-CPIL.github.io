import numpy as np
import time

N = 50

def construct():
        start_time = time.time()
        deadline = start_time + 110  # 10s buffer for safety
        
        # 1. Initialization
        # Use the seed's lattice generation to get a good start
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                c = points[np.argsort(radius, kind="stable")[:N]].copy()
                break
            spacing *= 0.98
        
        # Initial score
        def get_score(c):
            # Pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists) / 2.0
            
            # Boundary distances
            boundary_dists = 1.0 - np.linalg.norm(c, axis=1)
            min_boundary_dist = np.min(boundary_dists)
            
            return min(min_pair_dist, min_boundary_dist)
        
        best_c = c.copy()
        best_score = get_score(best_c)
        
        # Optimization parameters
        beta = 10.0  # Sharpness for soft-min
        eta = 0.01   # Step size
        
        # Precompute indices for pairs
        # We only need to update gradients for pairs that are "active" (close)
        # But for N=50, computing all is fast.
        
        while time.time() < deadline:
            # Compute values v_k
            # v_k is either dist/2 or 1-|c|
            
            # Distances between all pairs
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, np.inf)
            
            # Boundary distances
            boundary_dists = 1.0 - np.linalg.norm(c, axis=1)
            
            # We want to maximize the minimum.
            # Let's use a soft-min approximation.
            # Values to consider: all pair distances / 2, and all boundary distances.
            # Total values: N*(N-1)/2 + N = 1225 + 50 = 1275 values.
            
            # Flatten pair distances (upper triangle)
            iu, ju = np.triu_indices(N, k=1)
            pair_vals = dists[iu, ju] / 2.0
            
            # Combine with boundary values
            all_vals = np.concatenate([pair_vals, boundary_dists])
            
            # Calculate weights for soft-min
            # w_k = exp(-beta * v_k) / sum(exp(-beta * v_k))
            # To avoid underflow/overflow, subtract min
            min_val = np.min(all_vals)
            exp_vals = np.exp(-beta * (all_vals - min_val))
            weights = exp_vals / np.sum(exp_vals)
            
            # Split weights back
            pair_weights = weights[:len(pair_vals)]
            boundary_weights = weights[len(pair_vals):]
            
            # Compute gradients
            # Grad for c_i is sum of contributions from pairs and boundary.
            # Pair (i,j) with weight w:
            # Contribution to c_i: w * (c_i - c_j) / |c_i - c_j|
            # Contribution to c_j: w * (c_j - c_i) / |c_i - c_j|
            
            grad = np.zeros_like(c)
            
            # Vectorized pair gradient update is tricky with sparse weights, 
            # but N is small. We can loop over pairs or use scatter_add.
            # Let's use a loop for clarity and simplicity, N=50 is small.
            # Actually, 1225 pairs * 50 iterations is fine.
            
            # Reset grad
            grad = np.zeros_like(c)
            
            # Add boundary contributions
            # For each i, grad[i] += boundary_weights[i] * (-c_i / |c_i|)
            norms = np.linalg.norm(c, axis=1, keepdims=True)
            # Avoid div by zero
            norms[norms == 0] = 1e-9
            boundary_grad = -c / norms
            grad += boundary_weights[:, np.newaxis] * boundary_grad
            
            # Add pair contributions
            # For each pair (i,j), weight w
            # grad[i] += w * (c_i - c_j) / dist_ij
            # grad[j] += w * (c_j - c_i) / dist_ij
            
            # We can vectorize this:
            # diff_ij = c[iu] - c[ju]
            # dist_ij = dists[iu, ju]
            # unit_vec = diff_ij / dist_ij
            # contrib = pair_weights[:, np.newaxis] * unit_vec
            # grad[iu] += contrib
            # grad[ju] -= contrib
            
            diff_ij = c[iu] - c[ju]
            dist_ij = dists[iu, ju]
            # Avoid div by zero
            dist_ij[dist_ij == 0] = 1e-9
            unit_vec = diff_ij / dist_ij[:, np.newaxis]
            contrib = pair_weights[:, np.newaxis] * unit_vec
            
            np.add.at(grad, iu, contrib)
            np.add.at(grad, ju, -contrib)
            
            # Update
            c += eta * grad
            
            # Check new score
            current_score = get_score(c)
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
                
            # Anneal beta? Or keep fixed?
            # Increasing beta makes it focus more on the true minimum.
            # But if beta is too high, it becomes non-differentiable.
            # Let's slowly increase beta.
            beta *= 1.01
            # Decay step size?
            # eta *= 0.999
            
            # If we are stuck, maybe perturb?
            # For now, simple gradient ascent.
            
        return best_c
