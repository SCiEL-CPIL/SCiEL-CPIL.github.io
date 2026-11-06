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

def _logsumexp_min(x, beta):
    """
    Soft minimum of x using log-sum-exp trick.
    Approximates min(x) as beta -> infinity.
    """
    # To avoid overflow, subtract max. But for min, we use -max(-x)
    # min(x) = -max(-x)
    # log(sum(exp(beta * x_i))) / beta  is soft max.
    # So soft min is -1/beta * log(sum(exp(-beta * x_i)))
    
    # For stability:
    # Let y = -x. Soft max(y) = 1/beta * log(sum(exp(beta*y)))
    # min(x) = - soft_max(-x)
    
    y = -x
    max_y = np.max(y)
    # If max_y is very large, exp(beta*max_y) might overflow.
    # We can scale.
    # Actually, for the objective we want to MINIMIZE the soft min of distances.
    # So we want to maximize the soft min.
    # Let's define the objective as: -soft_min(distances) + penalty
    
    # Standard log-sum-exp:
    # LSE(z) = log(sum(exp(z)))
    # SoftMax(z) = LSE(z)/beta
    # SoftMin(x) = -SoftMax(-x) = -LSE(-beta*x)/beta
    
    # To compute LSE(-beta*x) stably:
    # LSE(z) = max(z) + log(sum(exp(z - max(z))))
    z = -beta * x
    max_z = np.max(z)
    # If max_z is inf, handle it? x are distances, finite.
    lse = max_z + np.log(np.sum(np.exp(z - max_z)))
    
    return -lse / beta

def _objective_and_grad(x_flat, beta):
    """
    Objective: Minimize -soft_min(pair_dists) - soft_min(wall_dists) + penalty for violations.
    Actually, we want to MAXIMIZE min(pair_dists/2, wall_dists).
    Let r = min(pair_dists/2, wall_dists).
    We want to maximize r.
    Equivalent to minimizing -r.
    
    Let d_pairs be the matrix of distances (not squared).
    Let d_walls be the vector of distances to walls.
    
    We use a soft minimum to approximate min(d_pairs, 2*d_walls) / 2?
    No, the score is min(min_pair_dist/2, min_wall_dist).
    Let's define effective distances:
    d_eff_pairs = d_pairs / 2
    d_eff_walls = d_walls
    
    We want to maximize soft_min(d_eff_pairs, d_eff_walls).
    
    To make it differentiable and focus on the closest ones, we use log-sum-exp.
    SoftMin(S) = -1/beta * log(sum(exp(-beta * s_i)))
    
    We want to MINIMIZE -SoftMin(S).
    
    Gradient of -SoftMin(S) wrt s_i:
    Let L = log(sum(exp(-beta * s_j)))
    SoftMin = -L/beta
    -SoftMin = L/beta
    d(L/beta)/d(s_i) = (1/beta) * (1/sum(exp(-beta*s_j))) * exp(-beta*s_i) * (-beta)
    = - exp(-beta*s_i) / sum(exp(-beta*s_j))
    
    So grad wrt s_i is -p_i, where p_i is the softmax probability of -s_i.
    
    We need to chain rule to x (positions).
    """
    x = x_flat.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Avoid division by zero in gradient
    dist_sq_safe = np.maximum(dist_sq, 1e-12)
    d_pairs = np.sqrt(dist_sq_safe)
    np.fill_diagonal(d_pairs, 1e10) # Ignore self
    
    # Wall distances
    d_walls = np.minimum(x + 0.5, 0.5 - x) # min distance to any wall for each coord?
    # Wait, distance to boundary for point x in [-0.5, 0.5] is min(x-(-0.5), 0.5-x)
    # So for each coordinate k, d_wall_k = min(x_k+0.5, 0.5-x_k)
    # The sphere is inside if min_k d_wall_k >= r.
    # So the wall distance for sphere i is min_k (x_ik+0.5, 0.5-x_ik).
    
    d_walls_vec = np.min(np.stack([x + 0.5, 0.5 - x], axis=-1), axis=-1)
    
    # Effective distances for radius
    # r is limited by d_pairs/2 and d_walls
    # We combine them into a single set of "constraints" values to take the soft min of.
    # Values: d_pairs[i,j] / 2 for i<j, and d_walls[i] for all i.
    
    # To be efficient, we can compute gradients analytically.
    # Let V be the vector of all effective distances.
    # V = [d_pairs[i,j]/2 (i<j), d_walls[i] (all i)]
    
    # SoftMin(V) = -1/beta * log(sum(exp(-beta * V_k)))
    # We want to MINIMIZE -SoftMin(V).
    # Let E = -SoftMin(V) = 1/beta * log(sum(exp(-beta * V_k)))
    
    # grad E wrt V_k = - exp(-beta * V_k) / sum(exp(-beta * V_j))
    # Let p_k = exp(-beta * V_k) / sum(exp(-beta * V_j))
    # grad E wrt V_k = -p_k
    
    # Now chain rule to x.
    
    # 1. Pairwise part
    # V_pair[i,j] = 0.5 * sqrt((x_i - x_j)^2)
    # d V_pair / d x_i = 0.5 * (x_i - x_j) / d_pairs[i,j]
    # d V_pair / d x_j = 0.5 * (x_j - x_i) / d_pairs[i,j] = - d V_pair / d x_i
    
    # 2. Wall part
    # V_wall[i] = min_k (x_ik+0.5, 0.5-x_ik)
    # This is non-differentiable at the center of the face.
    # However, in a good packing, spheres are usually not exactly centered on the face normal unless symmetric.
    # For gradient purposes, we can approximate the derivative.
    # If x_ik > 0, d/dx_ik (0.5 - x_ik) = -1.
    # If x_ik < 0, d/dx_ik (x_ik + 0.5) = 1.
    # If x_ik == 0, derivative is undefined (subgradient [-1, 1]). We can use 0 or sign(x).
    # Let's use sign(x_ik) for the derivative of the active wall constraint.
    # Actually, d_walls_vec[i] is the min over 6 walls.
    # The gradient of min(a, b) is grad(a) if a<b, grad(b) if b<a, mix if equal.
    
    # To keep it simple and robust, let's compute the weights p_k for all V_k.
    # Then sum the contributions.
    
    # Flatten V for the log-sum-exp calculation
    # Number of pairs: N*(N-1)/2 = 4950
    # Number of walls: N = 100
    # Total V size: 5050
    
    # Create indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_pairs_flat = d_pairs[i_idx, j_idx]
    V_pairs = 0.5 * d_pairs_flat
    
    V_walls = d_walls_vec
    
    V_all = np.concatenate([V_pairs, V_walls])
    
    # Calculate p_k (probabilities)
    z = -beta * V_all
    max_z = np.max(z)
    exp_z = np.exp(z - max_z)
    sum_exp = np.sum(exp_z)
    p_all = exp_z / sum_exp
    
    # Objective value
    E = -np.log(sum_exp) / beta + max_z / beta # Wait.
    # E = 1/beta * log(sum(exp(-beta V)))
    # log(sum(exp(z))) = max_z + log(sum(exp(z-max_z)))
    # E = 1/beta * (max_z + log(sum_exp))
    # Note max_z is -beta * min(V). So E = -min(V) + 1/beta * log(sum_exp).
    # As beta -> inf, E -> -min(V). Correct.
    
    E = (max_z + np.log(sum_exp)) / beta
    
    # Gradient wrt x
    grad_x = np.zeros_like(x)
    
    # Pairwise gradients
    # For each pair (i,j), contribution to grad_x[i] is p_pair * d(V_pair)/dx_i
    # d(V_pair)/dx_i = 0.5 * (x_i - x_j) / d_pairs[i,j]
    # Note: V_pair = 0.5 * d_pairs. So d V_pair / d x_i = 0.5 * (x_i - x_j) / d_pairs.
    
    # p_pairs is p_all[:len(i_idx)]
    p_pairs = p_all[:len(i_idx)]
    
    # Vectorized update
    # diff_ij = x[i] - x[j]
    diff_ij = x[i_idx] - x[j_idx]
    
    # grad_factor = p_pairs * 0.5 / d_pairs_flat
    # Note: if d_pairs is 0, this blows up. We used 1e-12 in dist_sq, so d_pairs >= 1e-6.
    # But we should be careful.
    grad_factor = p_pairs * 0.5 / d_pairs_flat
    
    # grad_x[i] += sum over j of grad_factor * diff_ij
    # grad_x[j] -= sum over i of grad_factor * diff_ij
    
    # Use np.add.at for accumulation
    np.add.at(grad_x, i_idx, grad_factor[:, np.newaxis] * diff_ij)
    np.add.at(grad_x, j_idx, -grad_factor[:, np.newaxis] * diff_ij)
    
    # Wall gradients
    # V_walls = min_k (x_ik+0.5, 0.5-x_ik)
    # p_walls = p_all[len(i_idx):]
    p_walls = p_all[len(i_idx):]
    
    # For each sphere i, we need to find which wall is active (min distance).
    # d_walls_vec[i] is the min.
    # The gradient of the min is the gradient of the active term.
    # Active term for x_ik:
    # If x_ik > 0, active is 0.5 - x_ik, grad is -1.
    # If x_ik < 0, active is x_ik + 0.5, grad is 1.
    # If x_ik == 0, either.
    
    # However, d_walls_vec[i] is min over ALL 6 walls (x+0.5, 0.5-x for x,y,z).
    # We need to know which specific wall gave the min for each sphere.
    
    # Let's compute the 6 wall distances for each sphere
    # Walls:
    # 0: x = -0.5, dist = x+0.5, grad_x = 1
    # 1: x = 0.5, dist = 0.5-x, grad_x = -1
    # 2: y = -0.5, dist = y+0.5, grad_y = 1
    # 3: y = 0.5, dist = 0.5-y, grad_y = -1
    # 4: z = -0.5, dist = z+0.5, grad_z = 1
    # 5: z = 0.5, dist = 0.5-z, grad_z = -1
    
    # dists shape (N, 6)
    wall_dists = np.stack([
        x[:, 0] + 0.5, 0.5 - x[:, 0],
        x[:, 1] + 0.5, 0.5 - x[:, 1],
        x[:, 2] + 0.5, 0.5 - x[:, 2]
    ], axis=1)
    
    # Find index of min dist for each sphere
    min_idx = np.argmin(wall_dists, axis=1)
    
    # Gradient of the active wall distance
    # grad_vec shape (N, 3)
    grad_wall_active = np.zeros_like(x)
    
    for i in range(N):
        idx = min_idx[i]
        if idx == 0: grad_wall_active[i, 0] = 1.0
        elif idx == 1: grad_wall_active[i, 0] = -1.0
        elif idx == 2: grad_wall_active[i, 1] = 1.0
        elif idx == 3: grad_wall_active[i, 1] = -1.0
        elif idx == 4: grad_wall_active[i, 2] = 1.0
        elif idx == 5: grad_wall_active[i, 2] = -1.0
        
    # Contribution to grad_x
    # grad_x += p_walls[:, np.newaxis] * grad_wall_active
    grad_x += p_walls[:, np.newaxis] * grad_wall_active
    
    return E, grad_x.flatten()

def _bcs_initial_grid():
    """
    Generate initial points using a Body-Centered Cubic (BCC) lattice pattern.
    """
    points = []
    
    # 5x4x5 grid of BCC unit cells
    # To get 100 points, we can use a 5x4x5 grid of simple cubic points,
    # but shifted to be more like BCC?
    # Actually, a simple grid 5x4x5 = 100 points is a decent start.
    # Let's try to make it slightly better by using BCC structure.
    # A BCC lattice in a cube of side L has 2 points per unit cell of side a.
    # If we use 5x4x5 cells, that's 100 cells * 2 points = 200 points. Too many.
    # If we use 5x4x5 simple grid, it's 100 points.
    
    # Let's stick to a simple grid but centered.
    x_ticks = np.linspace(-0.4, 0.4, 5)
    y_ticks = np.linspace(-0.375, 0.375, 4)
    z_ticks = np.linspace(-0.4, 0.4, 5)
    
    for x in x_ticks:
        for y in y_ticks:
            for z in z_ticks:
                points.append([x, y, z])
                if len(points) >= N:
                    break
            if len(points) >= N:
                break
        if len(points) >= N:
            break
    
    return np.array(points[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    np.random.seed(42)
    
    # Initial grid
    best_x = _bcs_initial_grid()
    best_score = _calc_score(best_x)
    
    bounds = [(-0.5, 0.5)] * 3 * N
    
    # Multi-start optimization
    # We will run several iterations with different beta values (sharpness of min)
    # and different initial perturbations.
    
    betas = [10.0, 20.0, 50.0, 100.0]
    
    # Phase 1: Optimization from grid with increasing beta
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        x0 = best_x.copy()
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                x0.flatten(),
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success:
                new_x = res.x.reshape(N, 3)
                new_x = np.clip(new_x, -0.5, 0.5)
                new_score = _calc_score(new_x)
                
                if new_score > best_score:
                    best_score = new_score
                    best_x = new_x
                    
        except Exception:
            pass

    # Phase 2: Random restarts with high beta to escape local minima
    # Use the best found so far as a reference, but add noise.
    # Or just random restarts if time permits.
    
    num_restarts = 5
    restart_time_budget = 10.0
    
    if time.time() < deadline - 30:
        for i in range(num_restarts):
            if time.time() > deadline - 20:
                break
            
            # Create a perturbed version of the best solution
            # Or a completely random one?
            # Perturbed is usually better for local improvements.
            # Let's try a mix: 50% perturbed best, 50% random.
            
            if i % 2 == 0:
                # Perturbed best
                noise_scale = 0.05
                x0 = best_x + np.random.normal(0, noise_scale, best_x.shape)
                x0 = np.clip(x0, -0.49, 0.49)
            else:
                # Random
                x0 = np.random.uniform(-0.4, 0.4, (N, 3))
            
            beta = 50.0
            
            try:
                res = minimize(
                    lambda x: _objective_and_grad(x, beta)[0],
                    x0.flatten(),
                    jac=lambda x: _objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.success:
                    new_x = res.x.reshape(N, 3)
                    new_x = np.clip(new_x, -0.5, 0.5)
                    new_score = _calc_score(new_x)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_x = new_x
                        
            except Exception:
                pass

    # Phase 3: Final fine-tuning with very high beta
    if time.time() < deadline - 5:
        beta = 200.0
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                best_x.flatten(),
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success:
                final_x = res.x.reshape(N, 3)
                final_x = np.clip(final_x, -0.5, 0.5)
                final_score = _calc_score(final_x)
                
                if final_score > best_score:
                    best_score = final_score
                    best_x = final_x
                    
        except Exception:
            pass

    return best_x

if __name__ == "__main__":
    pts = construct()
    score = _calc_score(pts)
    print(f"Score: {score}")
