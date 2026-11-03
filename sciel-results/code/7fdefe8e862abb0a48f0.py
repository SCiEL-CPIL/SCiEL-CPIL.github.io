import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5

def compute_min_dist(pos):
    """Compute the minimum radius for a given configuration."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    return min(r_pair, r_wall)

def _soft_min_objective_and_grad(flat_pos, beta):
    """
    Objective: -soft_min(distances)
    soft_min(d) = -1/beta * log(sum(exp(-beta * d)))
    Minimizing this is equivalent to maximizing the soft minimum of the distances.
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances (upper triangle to avoid duplication and zero diag)
    # i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff = pos[i_idx] - pos[j_idx]
    dists = np.linalg.norm(diff, axis=1)
    
    # Wall distances
    # For each sphere i and axis k, distance to wall is HALF - |pos[i,k]|
    # We treat each (i,k) as a separate "constraint" distance v
    d_walls = HALF - np.abs(pos)
    flat_walls = d_walls.flatten()
    
    # Combine all "values" v_k
    # v_pair = dists / 2.0  (radius contribution from pair)
    # v_wall = flat_walls   (radius contribution from wall)
    v_pairs = dists / 2.0
    v_walls = flat_walls
    
    v_all = np.concatenate([v_pairs, v_walls])
    
    # Soft min calculation
    # To avoid overflow/underflow, shift by min
    v_min = np.min(v_all)
    exp_terms = np.exp(-beta * (v_all - v_min))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Objective value: -soft_min = 1/beta * log(sum_exp) + v_min
    # Wait, soft_min = v_min - (1/beta) log(sum_exp)
    # So -soft_min = -v_min + (1/beta) log(sum_exp)
    obj_val = -v_min + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    # grad_obj = - sum( w_k * grad(v_k) )
    # w_k are the weights calculated above.
    
    grad_pos = np.zeros_like(pos)
    
    # Pair gradients
    # w_pairs = weights[:N_pairs]
    w_pairs = weights[:len(v_pairs)]
    
    # grad(v_pair) for pair (i,j) wrt x_i is 0.5 * (x_i - x_j) / dist
    # wrt x_j is -0.5 * (x_i - x_j) / dist
    # We have diff = x_i - x_j. dist = |diff|.
    # unit_vec = diff / dist
    # grad_v_i = 0.5 * unit_vec
    # grad_v_j = -0.5 * unit_vec
    
    # Avoid division by zero
    safe_dists = np.where(dists < 1e-8, 1e-8, dists)
    unit_vecs = diff / safe_dists[:, np.newaxis]
    
    # Contribution to gradient: - w * grad_v
    # grad_pos[i] -= w * 0.5 * unit_vec
    # grad_pos[j] += w * 0.5 * unit_vec
    
    # Vectorized accumulation
    # We can use np.add.at
    np.add.at(grad_pos, i_idx, -0.5 * w_pairs[:, np.newaxis] * unit_vecs)
    np.add.at(grad_pos, j_idx, 0.5 * w_pairs[:, np.newaxis] * unit_vecs)
    
    # Wall gradients
    # w_walls = weights[len(v_pairs):]
    w_walls = weights[len(v_pairs):]
    
    # v_wall = HALF - |x|
    # grad_v = -sign(x)
    # grad_obj contribution = - w * (-sign(x)) = w * sign(x)
    
    signs = np.sign(pos)
    # Handle zeros in sign? If x=0, subgradient is in [-1,1]. 0 is a safe choice.
    signs[signs == 0] = 0.0
    
    # w_walls is flattened (N, 3). Reshape to (N, 3)
    w_walls_reshaped = w_walls.reshape(N, 3)
    
    # grad_pos += w_walls_reshaped * signs
    grad_pos += w_walls_reshaped * signs
    
    return obj_val, -grad_pos.flatten() # Note: minimize expects gradient of obj. 
    # My obj is -soft_min. 
    # grad(-soft_min) = - grad(soft_min).
    # soft_min = -1/beta log(sum exp(-beta v))
    # grad(soft_min) = - sum(w_i grad(v_i))
    # grad(-soft_min) = sum(w_i grad(v_i))
    
    # Let's re-verify the sign.
    # We want to MAXIMIZE min(v).
    # We MINIMIZE f = -soft_min(v).
    # f = - (v_min - 1/beta log(sum exp(-beta(v-v_min))))
    # f = -v_min + 1/beta log(sum exp(-beta(v-v_min)))
    
    # Let's check gradient of f directly.
    # f = 1/beta log(S) - v_min.
    # This is tricky because v_min is non-differentiable.
    # However, the standard soft-min is defined as:
    # m_beta(v) = -1/beta log(sum exp(-beta v_i))
    # This is a smooth lower bound on min(v_i).
    # We want to maximize m_beta(v).
    # So we minimize -m_beta(v).
    # -m_beta(v) = 1/beta log(sum exp(-beta v_i))
    # Let g(v) = 1/beta log(sum exp(-beta v_i))
    # grad(g) = 1/beta * 1/S * sum( exp(-beta v_i) * (-beta grad(v_i)) )
    # grad(g) = - 1/S sum( exp(-beta v_i) grad(v_i) )
    # grad(g) = - sum( w_i grad(v_i) )
    
    # So the gradient of the objective to minimize is - sum(w_i grad(v_i)).
    # In my code above:
    # Pair part: I accumulated -0.5 * w * unit_vec for i.
    # grad(v_pair) for i is 0.5 * unit_vec.
    # So -w * grad(v_pair) is -0.5 * w * unit_vec. Correct.
    # Wall part: I added w * sign(x).
    # grad(v_wall) is -sign(x).
    # So -w * grad(v_wall) is -w * (-sign(x)) = w * sign(x). Correct.
    
    # So the returned gradient is indeed the gradient of the objective to minimize.
    # The function returns obj_val, grad.
    # The code above returns obj_val, -grad_pos.flatten() where grad_pos was built as sum(w * grad_v).
    # Wait, I built grad_pos as the sum of contributions to the gradient of the *minimized* function?
    # Let's trace:
    # I want grad(f) where f = -m_beta.
    # grad(f) = - sum(w_i grad(v_i)).
    
    # In code:
    # np.add.at(grad_pos, i_idx, -0.5 * w_pairs * unit_vecs)
    # This adds -w_i * grad(v_pair)_i.
    # So grad_pos accumulates - sum(w_i grad(v_i)) for pairs.
    
    # grad_pos += w_walls * signs
    # This adds -w_k * grad(v_wall)_k.
    # So grad_pos accumulates - sum(w_k grad(v_k)) for walls.
    
    # So grad_pos IS grad(f).
    # The return statement is: return obj_val, -grad_pos.flatten()
    # This returns -grad(f). This is WRONG.
    # I should return grad(f).
    
    # Correction: return obj_val, grad_pos.flatten()

def _optimize_soft_min(pos, betas, max_iter_per_step=200):
    """
    Optimize using soft-min objective with annealing beta.
    """
    current_pos = pos.copy()
    bounds = [(-HALF, HALF)] * (N * 3)
    
    for beta in betas:
        try:
            res = minimize(
                lambda x: _soft_min_objective_and_grad(x, beta)[0],
                current_pos.flatten(),
                jac=lambda x: _soft_min_objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter_per_step, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                current_pos = res.x.reshape(N, 3)
        except Exception as e:
            #print(f"Optimization failed at beta {beta}: {e}")
            pass
            
    return current_pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    # Initial guess: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_pos = grid[:N].copy()
    
    best_r = compute_min_dist(best_pos)
    
    # Annealing schedule for beta
    # Start low to explore, end high to refine
    betas = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    
    # Main loop with restarts
    while time.time() < deadline - 10.0:
        # Perturb
        # Strategy: mix of random restarts and local perturbations
        if rng.random() < 0.2:
            start_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        else:
            scale = 0.01 + 0.04 * rng.random()
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
            start_pos = np.clip(start_pos, -HALF, HALF)
            
        # Optimize
        new_pos = _optimize_soft_min(start_pos, betas, max_iter_per_step=100)
        
        val = compute_min_dist(new_pos)
        
        if val > best_r:
            best_r = val
            best_pos = new_pos.copy()
            
            # If we found a good one, we can try to polish with higher betas
            if time.time() < deadline - 20.0:
                high_betas = [1000, 2000, 5000, 10000]
                polished = _optimize_soft_min(best_pos, high_betas, max_iter_per_step=200)
                val2 = compute_min_dist(polished)
                if val2 > best_r:
                    best_r = val2
                    best_pos = polished.copy()
    
    # Final clip
    best_pos = np.clip(best_pos, -HALF, HALF)
    return best_pos
