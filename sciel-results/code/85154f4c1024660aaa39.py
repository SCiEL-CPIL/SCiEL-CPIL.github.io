"""Rewritten program for packing 200 equal circles in a unit square."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200


def _score(coords):
    """Compute the exact radius for a given set of coordinates."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    min_pair = np.min(dist[~np.eye(N, dtype=bool)])
    min_bound = np.min(np.minimum(np.minimum(0.5 - coords[:, 0], coords[:, 0] + 0.5),
                                   np.minimum(0.5 - coords[:, 1], coords[:, 1] + 0.5)))
    return min(min_pair / 2.0, min_bound)


def _force_directed_phase(coords, iterations=200, deadline=None):
    """Force-directed phase to separate overlapping circles using repulsive forces."""
    coords = coords.copy()
    for it in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise distances
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # Repulsive forces: stronger for closer pairs
        # F = k / d^2 in the direction away from each other
        force = np.zeros_like(coords)
        mask = dist < 0.1  # Only consider close pairs for efficiency
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if mask[i, j]:
                    d = dist[i, j]
                    if d > 1e-10:
                        # Repulsive force proportional to 1/d^2
                        f_mag = 0.01 / (d * d + 1e-6)
                        force[i] += (diff[i, j] / d) * f_mag
                        force[j] -= (diff[i, j] / d) * f_mag
        
        # Boundary forces: push circles away from walls
        d_left = coords[:, 0] + 0.5
        d_right = 0.5 - coords[:, 0]
        d_bottom = coords[:, 1] + 0.5
        d_top = 0.5 - coords[:, 1]
        
        # Add boundary repulsion
        force[:, 0] += np.where(d_left < 0.05, 0.02 / (d_left + 1e-6), 0.0)
        force[:, 0] -= np.where(d_right < 0.05, 0.02 / (d_right + 1e-6), 0.0)
        force[:, 1] += np.where(d_bottom < 0.05, 0.02 / (d_bottom + 1e-6), 0.0)
        force[:, 1] -= np.where(d_top < 0.05, 0.02 / (d_top + 1e-6), 0.0)
        
        # Apply forces with small step
        step = 0.001
        coords += force * step
        coords = np.clip(coords, -0.4999, 0.4999)
    
    return coords


def _soft_min_obj(coords, beta):
    """
    Smooth surrogate objective: minimize -soft_min(min_pair/2, min_bound)
    using log-sum-exp for smoothness.
    Returns (objective_value, gradient).
    """
    # Pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Set diagonal to large value
    np.fill_diagonal(dist_sq, np.inf)
    # Use squared distances for numerical stability; min_pair/2 = 0.5 * sqrt(min dist_sq)
    # soft_min over pairwise: we want soft min of (dist_ij / 2)
    # log-sum-exp trick: soft_min(x_i) = -logsumexp(-beta * x_i) / beta
    # Here x_i = dist_ij / 2 for all pairs i<j
    # Use dist_sq to avoid sqrt in the exponent: dist/2 = sqrt(dist_sq)/2
    # -beta * dist/2 = -beta/2 * sqrt(dist_sq)
    neg_beta_half = -beta / 2.0
    exp_vals = np.exp(neg_beta_half * np.sqrt(dist_sq))
    # Sum over all pairs (off-diagonal)
    pair_sum = np.sum(exp_vals)
    soft_min_pair = -np.log(pair_sum) / beta

    # Boundary distances
    d_left = coords[:, 0] + 0.5
    d_right = 0.5 - coords[:, 0]
    d_bottom = coords[:, 1] + 0.5
    d_top = 0.5 - coords[:, 1]
    # Combine into a single array of boundary distances
    d_bound = np.concatenate([d_left, d_right, d_bottom, d_top])
    exp_bound = np.exp(-beta * d_bound)
    bound_sum = np.sum(exp_bound)
    soft_min_bound = -np.log(bound_sum) / beta

    # Now soft-min of soft_min_pair and soft_min_bound
    # Use log-sum-exp again: soft_min(a, b) = -log(exp(-beta*a) + exp(-beta*b)) / beta
    total_exp = np.exp(-beta * soft_min_pair) + np.exp(-beta * soft_min_bound)
    soft_min_total = -np.log(total_exp) / beta

    objective = -soft_min_total

    # Gradient computation
    # d(soft_min_total)/d(coords) via chain rule
    # soft_min_total = -log(exp(-beta*sm_pair) + exp(-beta*sm_bound)) / beta
    # Let E = exp(-beta*sm_pair) + exp(-beta*sm_bound)
    # d(soft_min_total)/d(sm_pair) = (1/beta) * (1/E) * (-beta) * exp(-beta*sm_pair) = -exp(-beta*sm_pair)/E
    # d(soft_min_total)/d(sm_bound) = -exp(-beta*sm_bound)/E
    # d(objective)/d(coords) = -d(soft_min_total)/d(coords)

    w_pair = np.exp(-beta * soft_min_pair) / total_exp
    w_bound = np.exp(-beta * soft_min_bound) / total_exp

    # Gradient of soft_min_pair w.r.t. coords
    # sm_pair = -log(sum_ij exp(-beta/2 * sqrt(dist_sq_ij))) / beta
    # d(sm_pair)/d(dist_sq_ij) = (1/beta) * (1/S) * (-beta/2) * exp(-beta/2*sqrt(dist_sq)) * (1/(2*sqrt(dist_sq)))
    # where S = sum of exp terms
    # = -exp(-beta/2*sqrt(dist_sq)) / (2 * S * sqrt(dist_sq))
    sqrt_dist_sq = np.sqrt(dist_sq)
    exp_pair = np.exp(-beta / 2.0 * sqrt_dist_sq)
    S_pair = np.sum(exp_pair)
    # d(sm_pair)/d(dist_sq_ij) = -exp_pair_ij / (2 * S_pair * sqrt_dist_sq_ij)
    # d(dist_sq_ij)/d(coords_i) = 2*(coords_i - coords_j)
    # d(dist_sq_ij)/d(coords_j) = 2*(coords_j - coords_i)
    # So d(sm_pair)/d(coords_i) = sum_j d(sm_pair)/d(dist_sq_ij) * 2*(coords_i - coords_j)

    grad_sm_pair = np.zeros_like(coords)
    # Vectorized: for each i, sum over j of [-exp_pair[i,j] / (2*S*sqrt_dist_sq[i,j])] * 2*(coords[i]-coords[j])
    # = sum_j [-exp_pair[i,j] / (S*sqrt_dist_sq[i,j])] * (coords[i]-coords[j])
    # Set diagonal to 0
    exp_pair = exp_pair.copy()
    np.fill_diagonal(exp_pair, 0.0)
    sqrt_dist_sq = sqrt_dist_sq.copy()
    np.fill_diagonal(sqrt_dist_sq, 1e-12)  # avoid div by zero
    factor = exp_pair / (S_pair * sqrt_dist_sq)
    # factor[i,j] * (coords[i] - coords[j]), sum over j
    # grad_sm_pair[i] = -sum_j factor[i,j] * (coords[i] - coords[j])
    # = -sum_j factor[i,j]*coords[i] + sum_j factor[i,j]*coords[j]
    # = -coords[i]*sum_j factor[i,j] + sum_j factor[i,j]*coords[j]
    row_sums = np.sum(factor, axis=1)
    weighted_coords = factor @ coords
    grad_sm_pair = -row_sums[:, np.newaxis] * coords + weighted_coords

    # Gradient of soft_min_bound w.r.t. coords
    # sm_bound = -log(sum exp(-beta*d_k)) / beta
    # d(sm_bound)/d(d_k) = exp(-beta*d_k) / S_bound
    # where S_bound = sum exp(-beta*d_k)
    # d_k are: d_left[i]=x_i+0.5, d_right[i]=0.5-x_i, d_bottom[i]=y_i+0.5, d_top[i]=0.5-y_i
    # d(d_left[i])/d(coords[i,0]) = 1
    # d(d_right[i])/d(coords[i,0]) = -1
    # d(d_bottom[i])/d(coords[i,1]) = 1
    # d(d_top[i])/d(coords[i,1]) = -1
    exp_bound = np.exp(-beta * d_bound)
    S_bound = np.sum(exp_bound)
    weights_bound = exp_bound / S_bound  # shape (4N,)
    # Split into d_left, d_right, d_bottom, d_top weights
    w_left = weights_bound[:N]
    w_right = weights_bound[N:2*N]
    w_bottom = weights_bound[2*N:3*N]
    w_top = weights_bound[3*N:4*N]

    grad_sm_bound = np.zeros_like(coords)
    # d(sm_bound)/d(coords[i,0]) = w_left[i]*1 + w_right[i]*(-1) = w_left[i] - w_right[i]
    # d(sm_bound)/d(coords[i,1]) = w_bottom[i]*1 + w_top[i]*(-1) = w_bottom[i] - w_top[i]
    grad_sm_bound[:, 0] = w_left - w_right
    grad_sm_bound[:, 1] = w_bottom - w_top

    # Total gradient of objective = -[w_pair * grad_sm_pair + w_bound * grad_sm_bound]
    grad = -(w_pair * grad_sm_pair + w_bound * grad_sm_bound)

    return objective, grad


def _generate_hex_initial(n_points, seed=42):
    """Generate initial hexagonal lattice points, centered in unit square."""
    rng = np.random.RandomState(seed)
    # Target spacing based on known optimal radius ~0.0366
    # For hex packing, spacing = 2 * r / sin(60°) = 2*r / (sqrt(3)/2) = 4*r/sqrt(3)
    # But we want to start slightly larger to avoid overlaps
    target_r = 0.0366
    s_est = 2.0 * target_r / (np.sqrt(3.0) / 2.0) * 1.1  # 10% larger to start
    points = []
    max_radius = 0.55
    for y_row in range(-30, 31):
        y = y_row * s_est * np.sqrt(3.0) / 2.0
        if abs(y) > max_radius:
            continue
        x_offset = (s_est / 2.0) if (y_row % 2 != 0) else 0.0
        for x_col in range(-30, 31):
            x = x_col * s_est + x_offset
            if abs(x) > max_radius or abs(y) > max_radius:
                continue
            points.append([x, y])
    points = np.array(points)
    if len(points) < n_points:
        # Fallback: grid
        m = int(np.ceil(np.sqrt(n_points)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid
    idx = rng.choice(len(points), n_points, replace=False)
    return points[idx]


def _polish_gradient(coords, deadline):
    """Polish using gradient-based optimization on the exact objective."""
    coords = coords.copy()
    best_score = _score(coords)
    
    # Use L-BFGS-B with a smooth approximation that gets sharper over time
    betas = [100.0, 200.0, 500.0, 1000.0, 2000.0]
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
        
        def obj(x, b=beta):
            return _soft_min_obj(x, b)[0]
        
        def grad(x, b=beta):
            return _soft_min_obj(x, b)[1]
        
        try:
            result = minimize(obj, coords, jac=grad, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.x is not None:
                coords = np.clip(result.x, -0.4999, 0.4999)
                new_score = _score(coords)
                if new_score > best_score:
                    best_score = new_score
        except Exception:
            pass
    
    # Final local search: try small random perturbations and keep improvements
    rng = np.random.RandomState(42)
    for _ in range(1000):
        if time.time() > deadline - 2.0:
            break
        
        trial = coords.copy()
        # Perturb 5-10 random points
        n_perturb = rng.randint(5, 11)
        idx = rng.choice(N, n_perturb, replace=False)
        trial[idx] += rng.normal(0, 0.002, (n_perturb, 2))
        trial = np.clip(trial, -0.4999, 0.4999)
        
        new_score = _score(trial)
        if new_score > best_score + 1e-10:
            best_score = new_score
            coords = trial
    
    return coords





def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    best_coords = None
    best_score = -np.inf
    
    # Try multiple initial configurations
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        
        seed = 42 + restart * 17
        rng = np.random.RandomState(seed)
        
        # Generate initial configuration
        if restart == 0:
            # Pure hexagonal
            coords = _generate_hex_initial(N, seed=seed)
        elif restart == 1:
            # Hex + small jitter
            coords = _generate_hex_initial(N, seed=seed)
            coords += rng.normal(0, 0.003, (N, 2))
        elif restart == 2:
            # Hex + medium jitter
            coords = _generate_hex_initial(N, seed=seed)
            coords += rng.normal(0, 0.008, (N, 2))
        elif restart == 3:
            # Hex + larger jitter
            coords = _generate_hex_initial(N, seed=seed)
            coords += rng.normal(0, 0.015, (N, 2))
        elif restart == 4:
            # Different hex spacing
            coords = _generate_hex_initial(N, seed=seed + 100)
            coords += rng.normal(0, 0.005, (N, 2))
        elif restart == 5:
            # Grid-based
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            coords = grid[:N].copy()
            coords += rng.normal(0, 0.005, (N, 2))
        else:
            # Random with hex bias
            coords = _generate_hex_initial(N, seed=seed + 50)
            coords += rng.normal(0, 0.02, (N, 2))
        
        coords = np.clip(coords, -0.4999, 0.4999)
        
        # Force-directed phase to separate overlaps
        if time.time() < deadline - 35.0:
            coords = _force_directed_phase(coords, iterations=100, deadline=deadline - 30.0)
        
        # Optimize using soft-min objective with L-BFGS-B
        try:
            # Start with moderate beta, increase for sharper approximation
            betas = [50.0, 100.0, 200.0, 500.0]
            for beta in betas:
                if time.time() > deadline - 25.0:
                    break
                
                def obj(x, b=beta):
                    return _soft_min_obj(x, b)[0]
                
                def grad(x, b=beta):
                    return _soft_min_obj(x, b)[1]
                
                result = minimize(obj, coords, jac=grad, method='L-BFGS-B',
                                 options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                                          'maxfun': 3000})
                
                if result.x is not None:
                    coords = np.clip(result.x, -0.4999, 0.4999)
            
            score = _score(coords)
            if score > best_score:
                best_score = score
                best_coords = coords.copy()
                
        except Exception:
            pass
    
    # Final polish with gradient-based optimization
    if best_coords is not None and time.time() < deadline - 15.0:
        polished = _polish_gradient(best_coords, deadline - 5.0)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_coords = polished
            best_score = polished_score
    
    # Ensure coordinates are within bounds
    if best_coords is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_coords = grid[:N]
    else:
        # Clamp to valid range
        best_coords = np.clip(best_coords, -0.4999, 0.4999)
    
    return best_coords
