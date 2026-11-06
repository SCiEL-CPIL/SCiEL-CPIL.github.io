import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _calc_score(pts):
    """Calculate the actual score (radius) for a set of points."""
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
        
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e10)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    min_wall_dist = np.min(0.5 - np.abs(pts))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _soft_min_objective(x_flat, temp):
    """
    Differentiable approximation of the minimum distance.
    We want to maximize r = min(d_ij/2, d_wall_i).
    Equivalent to minimizing -r.
    
    We use a soft-min: -1/temp * log(sum(exp(-d/temp))) approximates min(d).
    However, for numerical stability and gradient flow, we can use:
    f(d) = - (1/temp) * log( sum( exp(-d_i / temp) ) )
    
    To handle the mixture of pair distances (divided by 2) and wall distances,
    we construct a vector of all relevant 'effective radii' constraints.
    r_pairs = d_ij / 2
    r_walls = 0.5 - |x_i|
    
    We want to maximize min(r).
    Let's minimize -soft_min(r).
    
    Gradient of soft_min:
    g_i = (exp(-r_i/temp) / sum(exp(-r_j/temp))) * (-1/temp)
    Wait, d/dx (-soft_min) = sum( p_i * (d r_i / dx) ) where p_i are softmax weights on -r.
    p_i = exp(-r_i/temp) / sum(exp(-r_j/temp))
    
    This focuses the gradient on the most constraining pairs/walls (smallest r).
    """
    x = x_flat.reshape(N, 3)
    
    # 1. Compute all pair distances / 2
    # To save memory/time, we might not want to compute all 100*100 if N is large, 
    # but N=100 is small enough (10k pairs).
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e10)
    
    # We only care about upper triangle to avoid double counting in the sum?
    # Soft min is symmetric, so double counting just scales the "mass" but 
    # the gradient direction is the same if we are consistent. 
    # Actually, for optimization, we just need the gradient to point towards increasing the min.
    # Let's use all pairs i < j to keep the sum magnitude reasonable.
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(dist_sq[iu]) / 2.0
    
    # 2. Compute wall distances
    # For each point, min distance to any of the 6 walls.
    # d_wall_i = min(0.5 - x_i0, 0.5 + x_i0, 0.5 - x_i1, 0.5 + x_i1, 0.5 - x_i2, 0.5 + x_i2)
    # This is also a soft min problem. 
    # For simplicity and differentiability, we can approximate the wall constraint 
    # by treating each wall separately in the global soft min, or compute the 
    # hard min for the value but use a soft approximation for the gradient.
    # Better: Include all 6*N wall distances in the global soft min vector.
    
    x_abs = np.abs(x)
    # Distances to walls: 0.5 - x, 0.5 + x
    # Let's form a vector of all wall distances.
    # Shape (N, 6)
    walls = np.stack([
        0.5 - x[:, 0], 0.5 + x[:, 0],
        0.5 - x[:, 1], 0.5 + x[:, 1],
        0.5 - x[:, 2], 0.5 + x[:, 2]
    ], axis=1)
    d_walls = walls.flatten()
    
    # Combine all radii into one vector
    # r_all = concat(d_pairs, d_walls)
    r_all = np.concatenate([d_pairs, d_walls])
    
    # Soft min: -1/temp * logsumexp(-r_all / temp)
    # For stability:
    # logsumexp(z) = max(z) + log(sum(exp(z - max(z))))
    z = -r_all / temp
    z_max = np.max(z)
    log_sum_exp = z_max + np.log(np.sum(np.exp(z - z_max)))
    soft_min_val = -temp * log_sum_exp
    
    # We want to minimize -soft_min_val? 
    # No, we want to MAXIMIZE soft_min_val.
    # So we minimize -soft_min_val.
    objective_val = -soft_min_val
    
    # Gradient of objective_val wrt x
    # d(objective_val)/dx = d(-soft_min)/dx
    # soft_min = -1/temp * log(sum(exp(-r_i/temp)))
    # d(soft_min)/dr_i = (1/temp) * exp(-r_i/temp) / sum(exp(-r_j/temp)) = p_i
    # where p_i are the softmax probabilities of the "tightness" (higher p for smaller r).
    # d(soft_min)/dx = sum_i p_i * (dr_i/dx)
    # d(objective_val)/dx = - sum_i p_i * (dr_i/dx)
    
    # Calculate probabilities p
    exp_z = np.exp(z - z_max)
    sum_exp_z = np.sum(exp_z)
    p = exp_z / sum_exp_z
    
    # p is shape (M,), where M = N*(N-1)/2 + 6*N
    # We need to map p back to gradients for x.
    
    # Part 1: Pairs
    # d_pairs indices correspond to iu
    # dr_pair/dx_k = -0.5 * (x_k - x_l) / d_kl  for pair (k,l)
    # Note: d_pairs = d_kl / 2. So d(d_pairs)/dx_k = -0.5 * (x_k - x_kl) / d_kl = -0.5 * (x_k - x_kl) / (2 * d_pairs) = -0.25 * (x_k - x_kl) / d_pairs?
    # Let's stick to base distances.
    # r_pair = d_kl / 2
    # dr_pair/dx_k = 0.5 * d(d_kl)/dx_k = 0.5 * (x_k - x_l) / d_kl
    # Wait, d_kl = |x_k - x_l|. d/dx_k |x_k - x_l| = (x_k - x_l) / |x_k - x_l|.
    # So dr_pair/dx_k = 0.5 * (x_k - x_l) / d_kl.
    
    # Let's compute grad_pairs (N, 3)
    grad_pairs = np.zeros_like(x)
    
    # p_pair is the first N*(N-1)/2 elements of p
    p_pairs = p[:len(d_pairs)]
    
    # We need to add p_pair * dr_pair/dx_k to the gradient for each point k involved in the pair.
    # For pair (i, j) with weight w = p_pairs[idx]:
    # grad_x_i += w * 0.5 * (x_i - x_j) / d_ij
    # grad_x_j += w * 0.5 * (x_j - x_i) / d_ij
    
    # d_pairs is computed from dist_sq[iu]
    # d_ij = sqrt(dist_sq[iu])
    d_ij = np.sqrt(dist_sq[iu])
    # Avoid division by zero (should be rare if initialized well)
    d_ij_safe = np.where(d_ij > 1e-10, d_ij, 1e-10)
    
    # Vector: (x_i - x_j)
    diff_ij = x[iu[0]] - x[iu[1]]
    
    # Coefficient for each pair: w * 0.5 / d_ij
    coeff = p_pairs * 0.5 / d_ij_safe
    coeff = coeff[:, np.newaxis]
    
    # Weighted diff
    w_diff = diff_ij * coeff
    
    # Add to grad_pairs
    # np.add.at(grad_pairs, iu[0], w_diff)
    # np.add.at(grad_pairs, iu[1], -w_diff)
    
    # Using scatter add
    np.add.at(grad_pairs, iu[0], w_diff)
    np.add.at(grad_pairs, iu[1], -w_diff)
    
    # Part 2: Walls
    # p_walls is the remaining elements of p
    p_walls = p[len(d_pairs):]
    
    # Walls were stacked as:
    # [0.5 - x0, 0.5 + x0, 0.5 - x1, 0.5 + x1, 0.5 - x2, 0.5 + x2] for each point
    # So for point i, the 6 values are at indices 6*i to 6*i+5.
    # p_walls is flattened (N, 6).
    p_walls_reshaped = p_walls.reshape(N, 6)
    
    # dr_wall/dx:
    # w0: 0.5 - x0 -> dr/dx0 = -1
    # w1: 0.5 + x0 -> dr/dx0 = +1
    # w2: 0.5 - x1 -> dr/dx1 = -1
    # w3: 0.5 + x1 -> dr/dx1 = +1
    # w4: 0.5 - x2 -> dr/dx2 = -1
    # w5: 0.5 + x2 -> dr/dx2 = +1
    
    # grad_wall contribution for point i:
    # gx += p[6i] * (-1) + p[6i+1] * (1)
    # gy += p[6i+2] * (-1) + p[6i+3] * (1)
    # gz += p[6i+4] * (-1) + p[6i+5] * (1)
    
    grad_walls = np.zeros_like(x)
    grad_walls[:, 0] = -p_walls_reshaped[:, 0] + p_walls_reshaped[:, 1]
    grad_walls[:, 1] = -p_walls_reshaped[:, 2] + p_walls_reshaped[:, 3]
    grad_walls[:, 2] = -p_walls_reshaped[:, 4] + p_walls_reshaped[:, 5]
    
    total_grad = -(grad_pairs + grad_walls)
    
    return objective_val, total_grad.flatten()

def _generate_lattice_init(type, rng):
    """
    Generate lattice-based initializations.
    Type 0: BCC
    Type 1: FCC
    Type 2: Random
    Type 3: Simple Grid
    """
    if type == 2:
        return rng.uniform(-0.4, 0.4, (N, 3))
    
    points = []
    
    if type == 3: # Simple Grid 5x5x4 = 100
        # 5x5x4 grid
        xs = np.linspace(-0.45, 0.45, 5)
        ys = np.linspace(-0.45, 0.45, 5)
        zs = np.linspace(-0.45, 0.45, 4)
        for x in xs:
            for y in ys:
                for z in zs:
                    points.append([x, y, z])
        return np.array(points)
        
    if type == 0: # BCC
        # 4x4x4 BCC cells? 4^3 * 2 = 128. Prune to 100.
        a = 1.0 / 4.0
        for i in range(-4, 5):
            for j in range(-4, 5):
                for k in range(-4, 5):
                    p1 = np.array([i*a, j*a, k*a])
                    p2 = p1 + np.array([a/2, a/2, a/2])
                    points.append(p1)
                    points.append(p2)
                    
    elif type == 1: # FCC
        a = 1.0 / 4.0
        for i in range(-4, 5):
            for j in range(-4, 5):
                for k in range(-4, 5):
                    p1 = np.array([i*a, j*a, k*a])
                    p2 = p1 + np.array([a/2, a/2, 0])
                    p3 = p1 + np.array([a/2, 0, a/2])
                    p4 = p1 + np.array([0, a/2, a/2])
                    points.extend([p1, p2, p3, p4])

    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) > N:
        idx = rng.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        pad = rng.uniform(-0.4, 0.4, (N - len(points), 3))
        points = np.vstack([points, pad])
        
    return points

def _optimize_start(x0, start_temp, end_temp, deadline):
    """
    Perform optimization with annealing temperature.
    """
    current_x = x0.copy()
    current_temp = start_temp
    
    # We will run L-BFGS-B with a dynamic objective.
    # Since scipy.optimize.minimize does not support dynamic parameters easily without closure,
    # we will run several iterations, reducing temperature each time.
    
    steps_per_temp = 50
    num_temps = 5
    
    for i in range(num_temps):
        if time.time() > deadline - 5:
            break
            
        temp = start_temp * (end_temp / start_temp) ** (i / max(num_temps - 1, 1))
        
        def obj_wrapper(x_flat):
            val, grad = _soft_min_objective(x_flat, temp)
            return val, grad
        
        try:
            res = minimize(obj_wrapper, current_x.flatten(), method='L-BFGS-B', jac=True,
                           bounds=[(-0.499, 0.499)] * 3 * N,
                           options={'maxiter': steps_per_temp, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            if res.success:
                current_x = res.x.reshape(N, 3)
        except Exception:
            pass
            
    return current_x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_score = 0.0
    
    # Strategies:
    # 1. Grid Init
    # 2. BCC Init
    # 3. FCC Init
    # 4. Random Init
    # 5. Jittered Grid
    # 6. Jittered BCC
    
    inits = [
        (3, "Grid"),
        (0, "BCC"),
        (1, "FCC"),
        (2, "Random"),
        (3, "Grid_Jitter"),
        (0, "BCC_Jitter"),
    ]
    
    for init_type, name in inits:
        if time.time() > deadline - 10:
            break
            
        if "Jitter" in name:
            base_type = 3 if "Grid" in name else 0
            x0 = _generate_lattice_init(base_type, rng)
            x0 = x0 + rng.normal(0, 0.05, x0.shape)
        else:
            x0 = _generate_lattice_init(init_type, rng)
        
        # Annealing: Start with high temp (smooth landscape) to find basin, 
        # then low temp (sharp landscape) to refine.
        # High temp ~ 0.1, Low temp ~ 0.001
        x_opt = _optimize_start(x0, 0.1, 0.001, deadline)
        
        # Ensure points are inside box
        x_opt = np.clip(x_opt, -0.499, 0.499)
        
        score = _calc_score(x_opt)
        
        if score > best_score:
            best_score = score
            best_x = x_opt.copy()
            
    # If we have time, do a final polish on the best solution with very low temp
    if best_x is not None and time.time() < deadline - 10:
        # Use a very low temperature to sharpen the min
        x_final = _optimize_start(best_x, 0.01, 0.0001, deadline)
        x_final = np.clip(x_final, -0.499, 0.499)
        score = _calc_score(x_final)
        if score > best_score:
            best_score = score
            best_x = x_final.copy()
            
    # Fallback
    if best_x is None:
        x_ticks = np.linspace(-0.4, 0.4, 5)
        grid = np.array([(x, y, z) for x in x_ticks for y in x_ticks for z in x_ticks])
        best_x = grid[:N]
        best_x = np.clip(best_x, -0.499, 0.499)
        
    return best_x

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(f"Score: {_calc_score(pts)}")
