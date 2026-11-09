import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    best_c = None
    best_r = -np.inf
    
    # Soft-min parameter
    beta = 50.0
    
    # Objective function to minimize: -softmin(distances)
    # Distances: pairwise distances / 2, and distances to walls
    def objective(c):
        c = c.reshape(-1, 3)
        # Pairwise distances
        # Using broadcasting for efficiency
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # We only need i < j
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu] / 2.0
        
        # Wall distances
        # c is in [-0.5, 0.5]
        # Distance to wall is min(0.5 - c, 0.5 + c) for each dimension
        # Actually, distance to boundary of [-0.5, 0.5] is 0.5 - |c|
        wall_dists = 0.5 - np.abs(c)
        # Flatten
        wall_dists_flat = wall_dists.flatten()
        
        # Combine all constraints
        all_dists = np.concatenate([pair_dists, wall_dists_flat])
        
        # Soft min: -1/beta * log(sum(exp(-beta * d)))
        # To avoid overflow/underflow, shift by min
        min_d = np.min(all_dists)
        exp_terms = np.exp(-beta * (all_dists - min_d))
        soft_min_val = min_d - (1.0 / beta) * np.log(np.sum(exp_terms))
        
        return -soft_min_val

    def jacobian(c):
        c = c.reshape(-1, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu] / 2.0
        
        wall_dists = 0.5 - np.abs(c)
        wall_dists_flat = wall_dists.flatten()
        
        all_dists = np.concatenate([pair_dists, wall_dists_flat])
        min_d = np.min(all_dists)
        exp_terms = np.exp(-beta * (all_dists - min_d))
        sum_exp = np.sum(exp_terms)
        probs = exp_terms / sum_exp
        
        # Gradient of soft_min w.r.t. d_k is -probs[k]
        # So gradient of -soft_min is +probs[k]
        # We need d(c)
        
        grad = np.zeros_like(c)
        
        # 1. Pairwise terms
        # d_k corresponds to pair (i, j)
        # d_k = 0.5 * ||c_i - c_j||
        # d(d_k)/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # d(d_k)/d(c_j) = 0.5 * (c_j - c_i) / ||c_i - c_j||
        
        # We have pair_dists flattened. We need to map back to indices.
        # This is tricky with the flattened array. 
        # It's easier to compute the gradient directly without flattening into one array for the gradient step,
        # or reconstruct the full matrix.
        
        # Let's compute the gradient of the soft-min of the full set.
        # Let S = set of distances.
        # grad_soft_min = - sum_k probs[k] * grad_d_k
        
        # We will compute the contribution of each pair and each wall term.
        
        # Re-compute distances for gradient (or reuse if we stored them, but we didn't)
        # For efficiency, let's just compute the gradient of the objective directly.
        
        # Objective f(c) = -soft_min(D)
        # grad f = sum_k probs[k] * grad D_k
        
        # D_k for pairs: 0.5 * norm(c_i - c_j)
        # D_k for walls: 0.5 - |c_{ik}|
        
        # We need the probabilities probs.
        # We have probs for the flattened array.
        # Let's split probs back.
        
        n_pairs = len(pair_dists)
        n_walls = len(wall_dists_flat)
        
        probs_pairs = probs[:n_pairs]
        probs_walls = probs[n_pairs:]
        
        # Gradient from pairs
        # We need to accumulate gradients for each c_i.
        # Iterate over pairs? N=50, pairs=1225. This is fine.
        
        # To avoid O(N^2) python loop overhead, we can vectorize if possible,
        # but simple loops are okay for 1225 iterations per gradient call.
        # However, L-BFGS-B calls gradient many times.
        # Let's try to vectorize the pair gradient.
        
        # Create a matrix of pair gradients?
        # For each pair (i,j), weight w = probs_pairs[idx]
        # grad_c_i += w * 0.5 * (c_i - c_j) / dist_ij
        # grad_c_j += w * 0.5 * (c_j - c_i) / dist_ij
        
        # We can use the full distance matrix.
        # dists is NxN.
        # We need to mask i<j.
        
        # Let's construct the full probability matrix P (NxN)
        P = np.zeros((N, N))
        # Map flat indices to 2D indices
        # This is the bottleneck. 
        # Alternative: Compute gradient using the full matrix and symmetry?
        # No, soft-min is over the set of values.
        
        # Let's just loop. 1225 iterations is fast.
        for idx in range(n_pairs):
            i, j = iu[0][idx], iu[1][idx]
            w = probs_pairs[idx]
            if w < 1e-10: continue
            d_vec = c[i] - c[j]
            d_norm = np.linalg.norm(d_vec)
            if d_norm < 1e-12: continue
            grad_vec = 0.5 * d_vec / d_norm
            grad[i] += w * grad_vec
            grad[j] -= w * grad_vec # Note: grad of norm(c_j - c_i) is -(c_i - c_j)/norm
            
        # Gradient from walls
        # wall_dists_flat is flattened (N, 3)
        # probs_walls corresponds to these.
        # d(0.5 - |c_{ik}|)/d(c_{ik}) = -sign(c_{ik})
        
        # Reshape probs_walls to (N, 3)
        probs_walls_2d = probs_walls.reshape(N, 3)
        
        # grad_wall_k = -sign(c_k)
        # So grad_c += probs_walls_2d * (-sign(c))
        
        signs = np.sign(c)
        # Handle 0s in sign (derivative is undefined, but we can use 0 or small epsilon)
        # If c is 0, dist is 0.5. Gradient of 0.5-|c| is -sign(c). At 0, it's 0.
        grad -= probs_walls_2d * signs
        
        return grad.flatten()

    # Optimization loop with restarts
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        
        # Initialization
        # Jittered grid is usually a better start than pure random
        # Grid size approx 4x4x4 = 64 > 50.
        # Let's use a 3x4x5 grid? 60 points.
        # Or just random.
        
        # Let's try random initialization within the cube
        c0 = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        # Run optimization
        try:
            res = minimize(objective, c0.flatten(), method='L-BFGS-B', jac=jacobian, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            c_opt = res.x.reshape(N, 3)
            
            # Calculate actual score (hard min)
            # We need to compute the true radius
            diff = c_opt[:, np.newaxis, :] - c_opt[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            iu = np.triu_indices(N, 1)
            min_pair = np.min(dists[iu]) / 2.0
            
            wall_dists = 0.5 - np.abs(c_opt)
            min_wall = np.min(wall_dists)
            
            r = min(min_pair, min_wall)
            
            if r > best_r:
                best_r = r
                best_c = c_opt.copy()
                
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline - 5:
            break

    if best_c is None:
        # Fallback to grid if something went wrong
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_c
