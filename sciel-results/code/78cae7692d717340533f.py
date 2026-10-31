"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    return np.logaddexp(0.0, beta * x) / beta

def _soft_max(x, beta=50.0):
    """Numerically stable soft max."""
    return np.logaddexp.reduce(x, axis=-1) / beta

def _energy(pos, radii, beta=50.0):
    """
    Compute the hard-sphere potential energy.
    Returns (energy, forces) where forces is the negative gradient w.r.t. pos and radii.
    """
    n = len(radii)
    
    # Boundary violations
    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] - (1.0 - radii)
    bottom_v = radii - pos[:, 1]
    top_v = pos[:, 1] - (1.0 - radii)
    
    boundary_energy = (np.sum(_softplus(left_v, beta)) + 
                       np.sum(_softplus(right_v, beta)) + 
                       np.sum(_softplus(bottom_v, beta)) + 
                       np.sum(_softplus(top_v, beta)))
    
    # Pairwise overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(dist_sq + 1e-12)
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlap_v = min_dist - dist
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    pairwise_energy = np.sum(_softplus(overlap_v[mask], beta))
    
    total_energy = boundary_energy + pairwise_energy
    
    # Compute forces (negative gradient)
    forces = np.zeros_like(pos)
    dE_dradii = np.zeros_like(radii)
    
    # Boundary forces
    sigma_left = 1.0 / (1.0 + np.exp(-beta * left_v))
    sigma_right = 1.0 / (1.0 + np.exp(-beta * right_v))
    sigma_bottom = 1.0 / (1.0 + np.exp(-beta * bottom_v))
    sigma_top = 1.0 / (1.0 + np.exp(-beta * top_v))
    
    # dE/dx_i for left: -sigma_left, dE/dx_i for right: +sigma_right
    forces[:, 0] += -sigma_left + sigma_right
    forces[:, 1] += -sigma_bottom + sigma_top
    
    # dE/dr_i for left: +sigma_left, right: -sigma_right, bottom: +sigma_bottom, top: -sigma_top
    dE_dradii += sigma_left - sigma_right + sigma_bottom - sigma_top
    
    # Pairwise forces
    # dE/dx_i = sum_j sigma_ij * (x_i - x_j) / dist_ij
    # dE/dr_i = sum_j sigma_ij
    
    sigma_matrix = 1.0 / (1.0 + np.exp(-beta * overlap_v))
    sigma_matrix = sigma_matrix * mask  # Only upper triangle
    
    # Force on i from j: sigma_ij * (pos_i - pos_j) / dist
    # Vectorized: sum over j of sigma[i,j] * (pos[i] - pos[j]) / dist[i,j]
    # We can compute this as: for each i, sum_j sigma[i,j] * (pos[i,0] - pos[j,0]) / dist[i,j]
    
    # Let's do it component-wise
    inv_dist = np.zeros_like(dist)
    valid_dist = dist > 1e-8
    inv_dist[valid_dist] = 1.0 / dist[valid_dist]
    
    # Force on x_i: sum_j sigma[i,j] * (x_i - x_j) / dist[i,j]
    # = x_i * sum_j sigma[i,j] / dist[i,j] - sum_j sigma[i,j] * x_j / dist[i,j]
    
    sigma_over_dist = sigma_matrix * inv_dist
    
    force_x_i = pos[:, 0] * np.sum(sigma_over_dist, axis=1) - np.sum(sigma_over_dist * pos[np.newaxis, :, 0], axis=1)
    force_y_i = pos[:, 1] * np.sum(sigma_over_dist, axis=1) - np.sum(sigma_over_dist * pos[np.newaxis, :, 1], axis=1)
    
    forces[:, 0] += force_x_i
    forces[:, 1] += force_y_i
    
    # dE/dr_i = sum_j sigma[i,j]
    dE_dradii += np.sum(sigma_matrix, axis=1)
    
    return total_energy, forces, dE_dradii

def _objective_and_grad(x, beta=50.0):
    """
    Objective: minimize -sum(radii) + energy.
    Returns (value, gradient).
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    energy, forces, dE_dradii = _energy(pos, radii, beta)
    
    value = -np.sum(radii) + energy
    
    grad = np.zeros_like(x)
    grad[:2*N] = -forces.flatten()  # Negative of negative gradient = gradient of energy
    grad[2*N:] = -1.0 + dE_dradii  # Gradient of -sum(r) + E
    
    return value, grad

def _is_valid(pos, radii, tol=1e-6):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.99999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_hexagonal():
    """
    Generate a hexagonal lattice initial packing.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Hexagonal packing in a square
    # Row 0: 0.1, 0.3, 0.5, 0.7, 0.9
    # Row 1: 0.2, 0.4, 0.6, 0.8
    # etc.
    
    rows = [
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.2, 0.4, 0.6, 0.8],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.2, 0.4, 0.6, 0.8],
        [0.1, 0.3, 0.5, 0.7, 0.9],
    ]
    
    idx = 0
    for r_idx, row in enumerate(rows):
        y = 0.1 + r_idx * 0.2
        for x in row:
            if idx >= N:
                break
            pos[idx] = [x, y]
            radii[idx] = 0.09
            idx += 1
    
    return pos, radii

def _initial_best_fit():
    """
    Greedy best-fit initialization.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(3000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.005:
                best_r = max_r_circ
                best_x = x
                best_y = y
        
        if best_r < 0.005:
            best_x, best_y, best_r = 0.5, 0.5, 0.005
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.98
        
    return pos, radii

def _initial_corner_biased():
    """
    Initialization biased towards corners for larger circles.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [
        [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8],
        [0.5, 0.15], [0.5, 0.85], [0.15, 0.5], [0.85, 0.5],
        [0.35, 0.35], [0.65, 0.35], [0.35, 0.65], [0.65, 0.65],
        [0.5, 0.5],
    ]
    
    for i in range(min(N, len(corners))):
        pos[i] = corners[i]
        radii[i] = 0.12
    
    if N > len(corners):
        for i in range(len(corners), N):
            best_x, best_y, best_r = 0.5, 0.5, 0.0
            rng = np.random.RandomState(i * 777)
            for _ in range(1000):
                x = rng.uniform(0.1, 0.9)
                y = rng.uniform(0.1, 0.9)
                max_r_wall = min(x, 1.0-x, y, 1.0-y)
                max_r_circ = max_r_wall
                for j in range(i):
                    dx = x - pos[j, 0]
                    dy = y - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    possible_r = dist - radii[j]
                    if possible_r < max_r_circ:
                        max_r_circ = possible_r
                if max_r_circ > best_r:
                    best_r = max_r_circ
                    best_x = x
                    best_y = y
            pos[i] = [best_x, best_y]
            radii[i] = best_r * 0.98
            
    return pos, radii

def _initial_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _simulated_annealing(pos, radii, time_limit=10.0):
    """
    Simulated annealing to escape local minima.
    """
    start = time.time()
    pos = pos.copy()
    radii = radii.copy()
    
    T = 0.02
    cooling = 0.995
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    while time.time() - start < time_limit:
        # Pick a random circle to move
        i = np.random.randint(N)
        
        # Random perturbation
        dx = np.random.normal(0, 0.01)
        dy = np.random.normal(0, 0.01)
        
        new_x = pos[i, 0] + dx
        new_y = pos[i, 1] + dy
        
        if new_x < 0.01 or new_x > 0.99 or new_y < 0.01 or new_y > 0.99:
            continue
        
        # Compute new max radius
        max_r = min(new_x, 1.0-new_x, new_y, 1.0-new_y)
        for j in range(N):
            if i != j:
                ddx = new_x - pos[j, 0]
                ddy = new_y - pos[j, 1]
                dist = np.sqrt(ddx*ddx + ddy*ddy)
                max_r = min(max_r, dist - radii[j])
        
        new_r = max(max_r * 0.999, 1e-5)
        
        old_score = np.sum(radii)
        new_score = old_score - radii[i] + new_r
        
        # Accept if better, or with probability exp(-delta/T)
        if new_score > old_score or np.random.random() < np.exp((new_score - old_score) / T):
            pos[i, 0] = new_x
            pos[i, 1] = new_y
            radii[i] = new_r
            
            if new_score > best_score:
                best_score = new_score
                best_pos = pos.copy()
                best_radii = radii.copy()
        
        T *= cooling
    
    return best_pos, best_radii

def _polish(pos, radii, time_limit=20.0):
    """
    High-precision polish using L-BFGS-B with increasing beta.
    """
    start = time.time()
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Increase beta gradually
    betas = [20.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() - start > time_limit:
            break
            
        def obj(x):
            val, _ = _objective_and_grad(x, beta)
            return val
        
        def grad(x):
            _, g = _objective_and_grad(x, beta)
            return g
        
        try:
            result = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            x0 = result.x
        except Exception:
            pass
        
        pos_opt = x0[:2*N].reshape(N, 2)
        radii_opt = np.maximum(x0[2*N:], 1e-4)
        
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        score = np.sum(radii_pp)
        
        if _is_valid(pos_pp, radii_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
            
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple initializations
    inits = []
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_cb, rad_cb = _initial_corner_biased()
    inits.append((pos_cb, rad_cb))
    
    for i in range(4):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    # For each initialization, run SA then polish
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # Simulated annealing
        sa_time = min(8.0, max(2.0, (deadline - time.time()) / 3.0))
        pos_sa, rad_sa = _simulated_annealing(pos_init, rad_init, time_limit=sa_time)
        
        # Polish
        polish_time = min(15.0, max(5.0, deadline - time.time() - 10))
        pos_pol, rad_pol = _polish(pos_sa, rad_sa, time_limit=polish_time)
        
        score = np.sum(rad_pol)
        if _is_valid(pos_pol, rad_pol) and score > best_score:
            best_score = score
            best_pos = pos_pol.copy()
            best_radii = rad_pol.copy()
    
    # Final polish on the best result
    if time.time() < deadline - 10:
        pos_final, rad_final = _polish(best_pos, best_radii, time_limit=15.0)
        score = np.sum(rad_final)
        if _is_valid(pos_final, rad_final) and score > best_score:
            best_score = score
            best_pos = pos_final.copy()
            best_radii = rad_final.copy()
    
    # Final post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Fallback
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
