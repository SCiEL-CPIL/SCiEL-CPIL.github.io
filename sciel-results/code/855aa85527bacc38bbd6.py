import numpy as np
import time
from scipy.optimize import minimize

N = 50

# Precompute upper triangle indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    """
    if len(c) == 0:
        return 0.0
    
    # Pairwise distances using precomputed indices
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    Returns (radius, gradient) where gradient is (N, 3).
    Uses log-sum-exp for stability.
    """
    diff = c[_iu] - c[_ju]
    dists_sq = np.sum(diff * diff, axis=-1)
    dists = np.sqrt(dists_sq)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # We want to maximize the minimum distance.
    # Softmin approximation: -t * log(sum(exp(-d/t)))
    # Gradient of this w.r.t c gives direction to increase min d.
    
    # To avoid underflow/overflow in exp, shift by min value (which makes exponent 0 for min)
    # Actually, we are summing exp(-d/t). The term with smallest d has largest exp.
    # Let d_min = min(all_dists).
    # sum(exp(-d/t)) = exp(-d_min/t) * sum(exp(-(d-d_min)/t))
    # log(sum) = -d_min/t + log(sum(exp(-(d-d_min)/t)))
    # softmin = -t * log(sum) = d_min - t * log(sum(exp(-(d-d_min)/t)))
    
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    d_min = np.min(all_vals)
    
    # Shifted values
    shifted_pairs = pair_vals - d_min
    shifted_walls = wall_vals.ravel() - d_min
    
    # Compute weights
    # exp(-shifted/t) is <= 1. Safe.
    pair_weights = np.exp(-shifted_pairs / t)
    wall_weights = np.exp(-shifted_walls / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        return d_min, np.zeros_like(c)
        
    # Softmin value
    softmin_val = d_min - t * np.log(total_sum)
    
    # Gradient calculation
    # d(softmin)/dc = -t * d/dc(log(sum)) = -t * (1/sum) * d(sum)/dc
    # d(sum)/dc = sum(d(exp(-d_k/t))/dc) = sum(exp(-d_k/t) * (-1/t) * d(d_k)/dc)
    # So grad(softmin) = -t * (1/sum) * (-1/t) * sum(weights * grad(d_k))
    # grad(softmin) = (1/sum) * sum(weights * grad(d_k))
    # This is a weighted average of the gradients of the individual distances.
    
    grad_c = np.zeros_like(c)
    
    # Pair contributions: d_k = ||c_i - c_j||/2
    # grad_{c_i} d_k = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # grad_{c_j} d_k = 0.5 * (c_j - c_i) / ||c_i - c_j|| = -0.5 * (c_i - c_j) / ||c_i - c_j||
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # pair_weights are already scaled by exp(-(d-d_min)/t)
    # The formula derived above: grad = (1/total_sum) * sum( w_k * grad(d_k) )
    # Note: w_k here is exp(-d_k/t). In our shifted calculation, 
    # exp(-d_k/t) = exp(-(d_k-d_min)/t) * exp(-d_min/t) = pair_weights_k * exp(-d_min/t).
    # The factor exp(-d_min/t) cancels out in the ratio (1/total_sum).
    # So we can just use pair_weights and wall_weights directly.
    
    pair_grads = 0.5 * pair_weights[:, np.newaxis] * unit_vecs
    
    for idx in range(_NUM_PAIRS):
        i, j = _PAIR_IDX[idx]
        grad_c[i] += pair_grads[idx]
        grad_c[j] -= pair_grads[idx]
    
    # Wall contributions: d_k = 0.5 - |c_k|
    # grad_{c_k} d_k = -sign(c_k)
    # Wall distance is min over coordinates. The softmin handles the "min" part.
    # We treat each wall distance as a separate term in the sum.
    # wall_vals is (N, 3).
    # signs = np.sign(c)
    # grad_wall = -wall_weights * signs
    # grad_c += grad_wall
    
    signs = np.sign(c)
    grad_c -= wall_weights.reshape(N, 3) * signs
    
    # Normalize by total_sum
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _seed_fcc():
    """
    Generate an FCC-inspired initial configuration for 50 spheres.
    """
    m = 4
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                for dx, dy, dz in [(0, 0, 0),
                                    (0.5, 0.5, 0),
                                    (0.5, 0, 0.5),
                                    (0, 0.5, 0.5)]:
                    px = x + dx * spacing
                    py = y + dy * spacing
                    pz = z + dz * spacing
                    if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5 and -0.5 <= pz <= 0.5:
                        points.append([px, py, pz])
    
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        # Randomly select N points to break symmetry
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    
    return points

def _repulsion_anneal(c_init, deadline, t_start=0.1, t_end=0.001, steps_per_temp=50):
    """
    Vectorized repulsion-based simulated annealing with Metropolis acceptance.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    current_r = best_r
    
    # Temperature schedule
    num_temps = 20
    temps = np.logspace(np.log10(t_start), np.log10(t_end), num_temps)
    
    # Step size scaling
    base_step = 0.05
    
    for t in temps:
        if time.time() > deadline:
            break
        
        for step in range(steps_per_temp):
            if time.time() > deadline:
                break
            
            # Compute repulsive forces
            forces = np.zeros_like(c)
            
            # Pair repulsion: F ~ 1/d^2
            diff = c[_iu] - c[_ju]
            dists_sq = np.sum(diff * diff, axis=-1)
            dists = np.sqrt(dists_sq)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                force_mag = 1.0 / (dists_sq + 1e-6)
            force_mag = np.clip(force_mag, 0, 1e4)
            
            unit_vecs = diff / dists[:, np.newaxis]
            unit_vecs[np.isnan(unit_vecs)] = 0.0
            
            pair_forces = force_mag[:, np.newaxis] * unit_vecs
            
            for idx in range(_NUM_PAIRS):
                i, j = _PAIR_IDX[idx]
                forces[i] += pair_forces[idx]
                forces[j] -= pair_forces[idx]
            
            # Wall repulsion: F ~ 1/d^2
            wall_dists = 0.5 - np.abs(c)
            with np.errstate(divide='ignore', invalid='ignore'):
                wall_force_mag = 1.0 / (wall_dists**2 + 1e-6)
            wall_force_mag = np.clip(wall_force_mag, 0, 1e4)
            
            signs = np.sign(c)
            forces += wall_force_mag * signs
            
            # Normalize forces to prevent huge jumps
            force_norm = np.linalg.norm(forces, axis=1, keepdims=True)
            forces = forces / (force_norm + 1e-6)
            
            # Propose move
            noise = np.random.normal(0, t, size=c.shape)
            c_new = c + forces * base_step * t + noise
            c_new = np.clip(c_new, -0.4999, 0.4999)
            
            r_new = _compute_radius(c_new)
            
            # Metropolis acceptance
            if r_new > current_r:
                c = c_new
                current_r = r_new
                if r_new > best_r:
                    best_r = r_new
                    best_c = c_new.copy()
            else:
                # Accept worse move with probability exp((r_new - current_r) / t)
                # Note: r is radius, we want to maximize.
                # delta = r_new - current_r < 0
                # prob = exp(delta / t)
                delta = r_new - current_r
                if delta > -1e-10: # Handle numerical noise
                    c = c_new
                    current_r = r_new
                else:
                    accept_prob = np.exp(delta / t)
                    if np.random.random() < accept_prob:
                        c = c_new
                        current_r = r_new
                        
    return best_c, best_r

def _gradient_polish(c_init, deadline, t_init=0.05, t_end=0.005, max_iter=200):
    """
    Gradient descent on the softmin radius approximation.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    t = t_init
    lr = 0.01
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        val, grad = _softmin_radius_and_grad(c, t)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad = grad / (grad_norm + 1e-6)
        
        # Update
        c_new = c + lr * grad
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new
        else:
            # If we didn't improve, reduce learning rate and temperature
            lr *= 0.9
            t *= 0.95
            if lr < 1e-6:
                break
        
    return best_c, best_r

def construct():
    start_time = time.time()
    # Safety margin: 150s limit. Leave 10s for overhead.
    deadline = start_time + 140.0
    
    # 1. Initial Seed
    c = _seed_fcc()
    r = _compute_radius(c)
    
    # 2. Global Search: Repulsion Annealing
    # Use most of the time for this
    c, r = _repulsion_anneal(c, deadline, t_start=0.15, t_end=0.005, steps_per_temp=100)
    
    # 3. Local Refinement: Gradient Polish
    if time.time() < deadline - 5.0:
        c, r = _gradient_polish(c, deadline - 5.0, t_init=0.05, t_end=0.001, max_iter=500)
        
    # 4. Final Polish: L-BFGS on exact objective (if time permits)
    if time.time() < deadline - 2.0:
        def exact_obj(x):
            c_t = x.reshape(N, 3)
            return -_compute_radius(c_t)
            
        try:
            res = minimize(
                exact_obj, 
                c.flatten(), 
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N*3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            c_final = res.x.reshape(N, 3)
            c_final = np.clip(c_final, -0.4999, 0.4999)
            r_final = _compute_radius(c_final)
            if r_final > r:
                c = c_final
                r = r_final
        except Exception:
            pass

    return c

if __name__ == "__main__":
    # Simple test
    c = construct()
    r = _compute_radius(c)
    print(f"Radius: {r}")
    print(f"Score: {r / 0.1359545126417}")
