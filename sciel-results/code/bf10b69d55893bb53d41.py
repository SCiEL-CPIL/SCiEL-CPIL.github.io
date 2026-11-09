"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum distance.
    Uses a multi-stage smooth surrogate optimization with increasing beta.
    """
    start_time = time.time()
    time_limit = 120.0
    margin = 10.0
    effective_limit = time_limit - margin
    
    def check_time():
        return (time.time() - start_time) >= effective_limit

    # Objective: Minimize soft-min of distances
    # d_ij are pairwise distances
    # b_ik are boundary clearances (0.5 - |coord|)
    # We want to maximize min(d_ij, b_ik)
    # Equivalent to minimizing -min(d_ij, b_ik)
    # Smooth approximation: (1/beta) * ln( sum exp(-beta * d) ) approximates min(d) for large beta
    # Note: We want to maximize the minimum, so we minimize the negative of it.
    # Let's define the "risk" as the sum of exponentials of negative distances.
    # Minimizing log(sum(exp(-beta * d))) is equivalent to maximizing the soft-min.
    
    def soft_min_obj(x, beta):
        """
        x: flat array of size N*3
        beta: temperature parameter
        Returns: negative soft-min score (to be minimized)
        """
        centres = x.reshape(N, 3)
        
        # Pairwise distances
        # Use cdist or manual calculation. Manual might be faster for small N if vectorized carefully, 
        # but cdist is optimized.
        # We need all pairwise distances i < j
        # And boundary distances
        
        # Boundary distances
        # b_ik = 0.5 - |c_ik|
        boundary_dists = 0.5 - np.abs(centres)
        
        # Pairwise distances
        # To avoid calculating full matrix and filling diagonal, we can use np.triu_indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = centres[i_idx] - centres[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        
        # Combine all distances into one array
        # We want to maximize the minimum of these.
        # Soft min: (1/beta) * log(sum(exp(-beta * d)))
        # If we minimize this, we maximize the soft min.
        # Wait, if d is large, exp(-beta*d) is small. Sum is small. Log is negative.
        # So minimizing (1/beta)*log(sum(exp(-beta*d))) means making the sum small, 
        # which means making d large. This is correct.
        
        all_dists = np.concatenate([pair_dists, boundary_dists.flatten()])
        
        # Numerical stability: subtract max
        max_d = np.max(all_dists)
        exp_terms = np.exp(-beta * (all_dists - max_d))
        
        # If all distances are very large, exp terms are small.
        # log(sum(exp(-beta*(d - max_d)))) = -beta*(d_min - max_d) + log(sum(exp(-beta*(d - d_min))))
        # Actually, simpler:
        # soft_min = (1/beta) * log( sum( exp(-beta * d) ) )
        # To avoid underflow, we can factor out exp(-beta * min_d)? No, that's for max.
        # For min, we factor out exp(-beta * max_d)? No.
        # Let's just use the standard log-sum-exp trick.
        # log(sum(exp(-beta * d)))
        # Let M = min(all_dists). Then -beta * d >= -beta * M.
        # This doesn't help with overflow if d is negative (impossible here).
        # If d is large, -beta*d is large negative -> exp is 0.
        # If d is small (close to 0), -beta*d is close to 0 -> exp is 1.
        # The sum is dominated by the smallest distances.
        
        # To be safe against underflow in sum (if all d are huge), 
        # but d is bounded by 1.73 (diagonal of cube).
        # beta * 1.73 is not huge for beta=1000. exp(-1730) is 0.
        # So if min_dist is 0.1, exp(-100) is 3e-44. Sum might be small.
        # log(3e-44) is fine.
        
        # However, if we have many terms, sum might be larger.
        
        log_sum = np.log(np.sum(exp_terms))
        # Reconstruct:
        # sum(exp(-beta * d)) = exp(-beta * max_d) * sum(exp(-beta * (d - max_d)))
        # log(sum) = -beta * max_d + log(sum(exp(-beta * (d - max_d))))
        # soft_min = (1/beta) * (-beta * max_d + log_sum_adjusted)
        #          = -max_d + (1/beta) * log_sum_adjusted
        
        # In my code above:
        # exp_terms = exp(-beta * (d - max_d))
        # sum_exp = sum(exp_terms)
        # log_sum_exp = log(sum_exp)
        # log(sum(exp(-beta*d))) = -beta*max_d + log_sum_exp
        # soft_min = (1/beta) * (-beta*max_d + log_sum_exp)
        #          = -max_d + log_sum_exp / beta
        
        soft_min_val = -max_d + log_sum / beta
        
        # We want to maximize soft_min_val.
        # So we minimize -soft_min_val.
        return -soft_min_val

    def grad_soft_min_obj(x, beta):
        """
        Gradient of the objective function.
        f(x) = - (1/beta) * log( sum exp(-beta * d_k) )
        df/dx = (1/beta) * d/dx log( sum exp(-beta * d_k) )
             = (1/beta) * ( 1 / sum exp(-beta * d_k) ) * sum ( exp(-beta * d_k) * (-beta) * dd_k/dx )
             = - sum ( exp(-beta * d_k) * dd_k/dx ) / sum exp(-beta * d_k)
             = - sum ( p_k * dd_k/dx )
        where p_k = exp(-beta * d_k) / sum exp(-beta * d_k)
        
        dd_k/dx:
        For pair distance d_ij = ||c_i - c_j||
        d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
        d(d_ij)/d(c_j) = (c_j - c_i) / d_ij
        
        For boundary distance b_ik = 0.5 - |c_ik|
        d(b_ik)/d(c_ik) = -sign(c_ik)
        d(b_ik)/d(c_other) = 0
        """
        centres = x.reshape(N, 3)
        
        # Boundary distances
        boundary_dists = 0.5 - np.abs(centres)
        # Gradients for boundary terms
        # grad_b = -sign(centres)
        grad_boundary = -np.sign(centres)
        
        # Pairwise distances
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = centres[i_idx] - centres[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        
        # Avoid division by zero
        pair_dists_safe = np.where(pair_dists > 1e-10, pair_dists, 1e-10)
        
        # Unit vectors
        unit_vecs = diffs / pair_dists_safe[:, np.newaxis]
        
        # Combine all distances
        all_dists = np.concatenate([pair_dists, boundary_dists.flatten()])
        
        # Weights p_k
        max_d = np.max(all_dists)
        exp_terms = np.exp(-beta * (all_dists - max_d))
        sum_exp = np.sum(exp_terms)
        
        # p_k = exp(-beta * d_k) / sum(exp(-beta * d_k))
        # = exp(-beta * (d_k - max_d)) / sum_exp
        p_k = exp_terms / sum_exp
        
        # Gradient calculation
        # grad_f = - sum p_k * grad_d_k
        
        grad_flat = np.zeros_like(x)
        
        # Pairwise part
        # p_pair = p_k[:N*(N-1)/2]
        p_pair = p_k[:len(i_idx)]
        
        # grad_d_ij / d c_i = unit_vec
        # grad_d_ij / d c_j = -unit_vec
        
        # Accumulate into grad_flat
        # For each pair (i, j):
        # grad_flat[i*3:(i+1)*3] += p_pair[idx] * unit_vec[idx]
        # grad_flat[j*3:(j+1)*3] += p_pair[idx] * (-unit_vec[idx])
        
        # Using np.add.at for accumulation
        # We need to reshape p_pair and unit_vec to match indices
        # i_idx and j_idx are flat arrays of indices
        
        # grad_i contribution
        # p_pair[:, np.newaxis] * unit_vec
        # Sum over all pairs where i is the first index
        # This is tricky with add.at on flat array.
        
        # Let's construct the gradient for centres directly
        grad_centres = np.zeros((N, 3))
        
        # Pairwise
        # For each pair k with indices i, j:
        # grad_centres[i] += p_k * unit_vec_k
        # grad_centres[j] -= p_k * unit_vec_k
        
        # We can use np.add.at
        np.add.at(grad_centres, i_idx, p_pair[:, np.newaxis] * unit_vec)
        np.add.at(grad_centres, j_idx, -p_pair[:, np.newaxis] * unit_vec)
        
        # Boundary part
        # p_boundary = p_k[len(i_idx):]
        # These correspond to all N*3 boundary distances
        # Reshape p_boundary to (N, 3)
        p_boundary = p_k[len(i_idx):].reshape(N, 3)
        
        # grad_b_ik = -sign(c_ik)
        # Contribution: - p_boundary * grad_b_ik = - p_boundary * (-sign(c_ik)) = p_boundary * sign(c_ik)
        # Wait, the formula is grad_f = - sum p_k grad_d_k
        # So for boundary: - p_boundary * (-sign(c)) = p_boundary * sign(c)
        
        grad_centres += p_boundary * np.sign(centres)
        
        # Total gradient is negative of this?
        # f = -soft_min
        # grad_f = - grad_soft_min
        # grad_soft_min = (1/beta) * ... 
        # Let's re-verify the sign.
        # soft_min approx min(d). We want to maximize min(d).
        # So we minimize -min(d).
        # grad(-min(d)) = - grad(min(d)).
        # grad(soft_min) = sum p_k grad_d_k ?
        # soft_min = (1/beta) log sum exp(-beta d)
        # d(soft_min)/dx = (1/beta) * (1/sum) * sum exp(-beta d) * (-beta) * d(d)/dx
        #                = - sum (exp(-beta d)/sum) * d(d)/dx
        #                = - sum p_k d(d)/dx
        # So grad(soft_min) = - sum p_k grad_d_k
        # We minimize f = -soft_min
        # grad_f = - grad(soft_min) = sum p_k grad_d_k
        
        # So my calculation above:
        # grad_centres currently holds - sum p_k grad_d_k (which is grad_soft_min)
        # Wait, in the code:
        # np.add.at(grad_centres, i_idx, p_pair * unit_vec) -> this is sum p_k grad_d_ij (for i)
        # np.add.at(grad_centres, j_idx, -p_pair * unit_vec) -> this is sum p_k grad_d_ij (for j)
        # grad_centres += p_boundary * sign(centres) -> this is sum p_k grad_b (since grad_b = -sign, and we have -p_k*grad_b? No.)
        
        # Let's restart the gradient derivation carefully.
        # Objective to minimize: F(x) = - S(x) where S(x) = (1/beta) log(sum(exp(-beta d_k)))
        # dS/dx = (1/beta) * (1/Sum) * Sum( exp(-beta d_k) * (-beta) * dd_k/dx )
        # dS/dx = - Sum( (exp(-beta d_k)/Sum) * dd_k/dx )
        # dS/dx = - Sum( p_k * dd_k/dx )
        
        # dF/dx = - dS/dx = Sum( p_k * dd_k/dx )
        
        # So we need to compute Sum( p_k * dd_k/dx ).
        
        # For pairs:
        # dd_ij/dc_i = unit_vec
        # dd_ij/dc_j = -unit_vec
        # Contribution to grad_F at c_i: p_k * unit_vec
        # Contribution to grad_F at c_j: p_k * (-unit_vec)
        
        # For boundaries:
        # d_b_ik/dc_ik = -sign(c_ik)
        # Contribution to grad_F at c_ik: p_k * (-sign(c_ik))
        
        # My code:
        # np.add.at(grad_centres, i_idx, p_pair[:, np.newaxis] * unit_vec) 
        #   -> Adds p_k * unit_vec to c_i. Correct.
        # np.add.at(grad_centres, j_idx, -p_pair[:, np.newaxis] * unit_vec)
        #   -> Adds -p_k * unit_vec to c_j. Correct.
        # grad_centres += p_boundary * np.sign(centres)
        #   -> Adds p_k * sign(c) to c. 
        #   But we need p_k * (-sign(c)). 
        #   So this term should be -p_boundary * np.sign(centres).
        
        # Let's fix the boundary gradient sign.
        
        # Re-calculate boundary part
        # grad_centres_boundary = - p_boundary * np.sign(centres)
        
        # So:
        # grad_centres (from pairs) is correct.
        # grad_centres (from boundary) needs to be negative.
        
        # I will return the negative of what I calculated for boundary, 
        # or just flip the sign in the addition.
        
        # Let's just return the correct gradient.
        # Current grad_centres has pair parts correct.
        # I added p_boundary * sign. I should have added -p_boundary * sign.
        # So I subtract 2 * p_boundary * sign? Or just recompute.
        
        # Let's just recompute the boundary part correctly in the return.
        
        # Actually, I can just flip the sign of the boundary contribution I added.
        # I added: p_boundary * sign
        # I wanted: -p_boundary * sign
        # So I subtract 2 * p_boundary * sign? No, just correct the line.
        
        # Since I'm writing the code, I'll just write it correctly.
        
        # Correction:
        # grad_centres (pairs) is sum p_k grad_d_pair
        # grad_centres (boundary) should be sum p_k grad_d_boundary = sum p_k (-sign(c))
        
        # So:
        # grad_centres_final = grad_pairs - p_boundary * sign(centres)
        
        # In my code above, I did:
        # grad_centres += p_boundary * np.sign(centres)
        # This is WRONG. It should be -=.
        
        # So I will return:
        # grad_flat = (grad_centres - 2 * p_boundary * np.sign(centres)).flatten() ?
        # No, I just need to fix the logic in the function I'm writing.
        
        pass # Placeholder, will write correct version below

    # Correct Gradient Function
    def grad_soft_min_obj_correct(x, beta):
        centres = x.reshape(N, 3)
        
        # Boundary distances
        boundary_dists = 0.5 - np.abs(centres)
        
        # Pairwise distances
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = centres[i_idx] - centres[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        pair_dists_safe = np.where(pair_dists > 1e-10, pair_dists, 1e-10)
        unit_vecs = diffs / pair_dists_safe[:, np.newaxis]
        
        all_dists = np.concatenate([pair_dists, boundary_dists.flatten()])
        
        max_d = np.max(all_dists)
        exp_terms = np.exp(-beta * (all_dists - max_d))
        sum_exp = np.sum(exp_terms)
        
        p_k = exp_terms / sum_exp
        
        grad_centres = np.zeros((N, 3))
        
        # Pairwise part: sum p_k * grad_d_pair
        # grad_d_pair for c_i is unit_vec
        # grad_d_pair for c_j is -unit_vec
        p_pair = p_k[:len(i_idx)]
        np.add.at(grad_centres, i_idx, p_pair[:, np.newaxis] * unit_vecs)
        np.add.at(grad_centres, j_idx, -p_pair[:, np.newaxis] * unit_vecs)
        
        # Boundary part: sum p_k * grad_d_boundary
        # grad_d_boundary for c_ik is -sign(c_ik)
        p_boundary = p_k[len(i_idx):].reshape(N, 3)
        grad_centres += p_boundary * (-np.sign(centres))
        
        # Objective F = -soft_min
        # grad_F = - grad_soft_min
        # grad_soft_min = - sum p_k grad_d  (from derivation dS/dx = - sum p_k grad_d)
        # Wait.
        # S = (1/beta) log(sum exp(-beta d))
        # dS/dx = (1/beta) * (1/Sum) * sum(exp(-beta d) * (-beta) * dd/dx)
        # dS/dx = - sum (exp(-beta d)/Sum) * dd/dx = - sum p_k dd/dx
        # F = -S
        # dF/dx = - dS/dx = sum p_k dd/dx
        
        # So grad_F = sum p_k dd/dx.
        # My grad_centres calculation above IS sum p_k dd/dx.
        # So I should return grad_centres.flatten().
        
        return grad_centres.flatten()

    # Initializers
    def get_initializers():
        inits = []
        
        # 1. Grid 4x4x4
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        inits.append(grid[:N].copy())
        
        # 2. Fibonacci Lattice
        # Generate points in sphere, scale to cube? Or just use a 3D lattice?
        # Better: Random points with a seed.
        rng = np.random.RandomState(42)
        inits.append(rng.uniform(-0.5, 0.5, size=(N, 3)))
        
        # 3. Another random
        rng2 = np.random.RandomState(123)
        inits.append(rng2.uniform(-0.5, 0.5, size=(N, 3)))
        
        # 4. Grid 5x5x5 trimmed? 
        # 5^3 = 125. Too many.
        # Let's try a denser grid and pick best 50?
        # Or just use the 4x4x4 grid shifted?
        
        return inits

    initializers = get_initializers()
    
    best_centres = None
    best_score = -np.inf
    
    betas = [10, 50, 200, 1000]
    
    for init in initializers:
        if check_time():
            break
            
        current_centres = init.copy()
        
        # Run optimization for each beta
        for beta in betas:
            if check_time():
                break
                
            try:
                res = minimize(
                    soft_min_obj,
                    current_centres.flatten(),
                    jac=grad_soft_min_obj_correct,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success:
                    candidate = res.x.reshape(N, 3)
                    # Clamp to bounds just in case
                    candidate = np.clip(candidate, -0.5, 0.5)
                    
                    # Evaluate true score
                    # Calculate true min radius
                    dists = np.linalg.norm(candidate[:, np.newaxis] - candidate[np.newaxis, :], axis=2)
                    np.fill_diagonal(dists, np.inf)
                    min_pair = np.min(dists)
                    
                    boundary_dists = 0.5 - np.abs(candidate)
                    min_boundary = np.min(boundary_dists)
                    
                    true_score = min(min_pair / 2.0, min_boundary)
                    
                    if true_score > best_score:
                        best_score = true_score
                        best_centres = candidate.copy()
                        
                    # Use this result as start for next beta
                    current_centres = candidate
                    
            except Exception as e:
                pass

    # Final Polish: Coordinate Descent / Local Search
    # If we have a good candidate, try to improve it with a simple local search
    if best_centres is not None:
        improved_centres = best_centres.copy()
        
        # Simple local search: move each point slightly to increase min dist
        # This is slow, so we limit iterations
        max_iter = 50
        step_size = 0.01
        
        for it in range(max_iter):
            if check_time():
                break
                
            improved = False
            for i in range(N):
                if check_time():
                    break
                    
                # Compute current score
                dists = np.linalg.norm(improved_centres[:, np.newaxis] - improved_centres[np.newaxis, :], axis=2)
                np.fill_diagonal(dists, np.inf)
                min_pair = np.min(dists)
                boundary_dists = 0.5 - np.abs(improved_centres)
                min_boundary = np.min(boundary_dists)
                current_score = min(min_pair / 2.0, min_boundary)
                
                # Try moving point i in 6 directions (and maybe diagonals?)
                # Just 6 orthogonal directions for speed
                directions = [
                    np.array([1, 0, 0]), np.array([-1, 0, 0]),
                    np.array([0, 1, 0]), np.array([0, -1, 0]),
                    np.array([0, 0, 1]), np.array([0, 0, -1])
                ]
                
                best_move = None
                best_new_score = current_score
                
                for d in directions:
                    new_pos = improved_centres[i] + step_size * d
                    new_pos = np.clip(new_pos, -0.5, 0.5)
                    
                    old_pos = improved_centres[i].copy()
                    improved_centres[i] = new_pos
                    
                    # Recalculate score
                    dists = np.linalg.norm(improved_centres[:, np.newaxis] - improved_centres[np.newaxis, :], axis=2)
                    np.fill_diagonal(dists, np.inf)
                    min_pair = np.min(dists)
                    boundary_dists = 0.5 - np.abs(improved_centres)
                    min_boundary = np.min(boundary_dists)
                    new_score = min(min_pair / 2.0, min_boundary)
                    
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_move = new_pos.copy()
                    else:
                        improved_centres[i] = old_pos
                    
                    if best_move is not None:
                        improved_centres[i] = best_move
                        improved = True
                        break # Move to next point if one found? Or try all?
                        # Usually better to try all and pick best, but for speed, break
                        # Let's try all 6 and pick the best one.
                        # Actually, the loop above updates best_move but reverts if not best.
                        # If we find a better one, we set best_move.
                        # We should finish the loop to check all directions.
                        # So remove break.
                        # But for speed, maybe just take the first improvement?
                        # Let's take the best of the 6.
                
                # The logic above is a bit messy. Let's simplify:
                # Try all 6, keep the best one.
                # If best_new_score > current_score, apply it.
                
                # Redoing the inner loop logic for clarity and correctness
                # (The previous block was inside the 'for d' loop which is inefficient if we break)
                # I will rewrite the local search block properly.
                
                pass # The previous block was flawed. I will implement a cleaner version below.
            
            if not improved:
                break

    # Proper Local Search Implementation
    if best_centres is not None and not check_time():
        improved_centres = best_centres.copy()
        step_size = 0.02
        max_iter = 20
        
        for it in range(max_iter):
            if check_time():
                break
                
            improved = False
            for i in range(N):
                if check_time():
                    break
                    
                # Current score
                dists = np.linalg.norm(improved_centres[:, np.newaxis] - improved_centres[np.newaxis, :], axis=2)
                np.fill_diagonal(dists, np.inf)
                min_pair = np.min(dists)
                boundary_dists = 0.5 - np.abs(improved_centres)
                min_boundary = np.min(boundary_dists)
                current_score = min(min_pair / 2.0, min_boundary)
                
                best_new_score = current_score
                best_new_pos = improved_centres[i].copy()
                
                directions = [
                    np.array([1, 0, 0]), np.array([-1, 0, 0]),
                    np.array([0, 1, 0]), np.array([0, -1, 0]),
                    np.array([0, 0, 1]), np.array([0, 0, -1])
                ]
                
                for d in directions:
                    new_pos = improved_centres[i] + step_size * d
                    new_pos = np.clip(new_pos, -0.5, 0.5)
                    
                    old_pos = improved_centres[i].copy()
                    improved_centres[i] = new_pos
                    
                    dists = np.linalg.norm(improved_centres[:, np.newaxis] - improved_centres[np.newaxis, :], axis=2)
                    np.fill_diagonal(dists, np.inf)
                    min_pair = np.min(dists)
                    boundary_dists = 0.5 - np.abs(improved_centres)
                    min_boundary = np.min(boundary_dists)
                    new_score = min(min_pair / 2.0, min_boundary)
                    
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_pos = new_pos.copy()
                    
                    improved_centres[i] = old_pos
                
                if best_new_score > current_score:
                    improved_centres[i] = best_new_pos
                    improved = True
            
            if not improved:
                break
                
        # Check if improved
        dists = np.linalg.norm(improved_centres[:, np.newaxis] - improved_centres[np.newaxis, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        boundary_dists = 0.5 - np.abs(improved_centres)
        min_boundary = np.min(boundary_dists)
        final_score = min(min_pair / 2.0, min_boundary)
        
        if final_score > best_score:
            best_centres = improved_centres
            best_score = final_score

    # Fallback
    if best_centres is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N].copy()
        
    return best_centres

# EVOLVE-BLOCK-END
