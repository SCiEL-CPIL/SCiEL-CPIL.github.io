import numpy as np
from scipy.optimize import minimize
import time

N_FULL = 300
N_SECTOR = N_FULL // 6  # 50
N_VARS = N_SECTOR * 2

def _rotate_point(p, angle):
    """Rotate a 2D point by angle."""
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ p

def _generate_full_config(sector_points):
    """Generate the full 300-point configuration from 50 sector points."""
    points = []
    for k in range(6):
        angle = k * np.pi / 3.0
        for p in sector_points:
            points.append(_rotate_point(p, angle))
    return np.array(points)

def _compute_true_score_full(points):
    """Compute the true packing radius for the full 300-point config."""
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad_sector(x, beta):
    """
    Compute the negative soft-min value and gradient for the 50-point sector.
    x is a flat array of size 100.
    """
    P = x.reshape(N_SECTOR, 2)
    
    # 1. Distances within the sector
    D_in = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    L_in = np.linalg.norm(D_in, axis=2)
    np.fill_diagonal(L_in, np.inf)
    # We only need the upper triangle to avoid double counting in the sum, 
    # but for the gradient we need to be careful.
    # Actually, the soft-min sums over all pairs. 
    # Let's collect all pair distances.
    
    i_idx, j_idx = np.triu_indices(N_SECTOR, k=1)
    x_pairs_in = L_in[i_idx, j_idx] / 2.0
    
    # 2. Distances to the rotated neighbor (sector k and k+1)
    # We need |p_i - R(pi/3) p_j|
    R_rot = np.array([[np.cos(np.pi/3), -np.sin(np.pi/3)], 
                      [np.sin(np.pi/3), np.cos(np.pi/3)]])
    P_rot = P @ R_rot.T
    
    D_cross = P[:, np.newaxis, :] - P_rot[np.newaxis, :, :]
    L_cross = np.linalg.norm(D_cross, axis=2)
    # Note: The pair (i, j) in cross distance is distinct from (j, i) in the next sector?
    # The set of points is {R^k P_i}.
    # Interaction between R^0 P_i and R^1 P_j is |P_i - R P_j|.
    # Interaction between R^1 P_j and R^0 P_i is |R P_j - P_i| = |P_i - R P_j|.
    # So we must count each pair of indices (i,j) once for the cross interaction.
    # The distance matrix L_cross has N_SECTOR x N_SECTOR entries.
    # Are there duplicates?
    # |P_i - R P_j| vs |P_k - R P_l|.
    # Generally distinct.
    # We sum over all i,j.
    x_pairs_cross = L_cross.flatten() / 2.0
    
    # 3. Wall distances
    L_wall = np.linalg.norm(P, axis=1)
    x_walls = 1.0 - L_wall
    
    x_all = np.concatenate([x_pairs_in, x_pairs_cross, x_walls])
    
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P_dist = exp_terms / Z
    
    val = min_x - (1.0 / beta) * np.log(Z)
    
    grad = np.zeros_like(P)
    
    # Gradient for intra-sector pairs
    P_pairs_in = P_dist[:len(i_idx)]
    D_flat_in = D_in[i_idx, j_idx]
    L_flat_in = L_in[i_idx, j_idx]
    L_flat_in_safe = np.where(L_flat_in > 1e-12, L_flat_in, 1.0)
    U_in = D_flat_in / L_flat_in_safe[:, np.newaxis]
    W_in = P_pairs_in * 0.5
    np.add.at(grad, i_idx, W_in[:, np.newaxis] * U_in)
    np.add.at(grad, j_idx, -W_in[:, np.newaxis] * U_in)
    
    # Gradient for cross-sector pairs
    P_pairs_cross = P_dist[len(i_idx):len(i_idx)+N_SECTOR*N_SECTOR]
    # Reshape to (N_SECTOR, N_SECTOR)
    P_pairs_cross = P_pairs_cross.reshape(N_SECTOR, N_SECTOR)
    
    # d/dP_i |P_i - R P_j| = (P_i - R P_j) / |P_i - R P_j|
    # d/dP_j |P_i - R P_j| = -R.T (P_i - R P_j) / |P_i - R P_j|
    
    # We need to accumulate gradients for each P_i and P_j.
    # For each i, j:
    # grad_P[i] += P_dist[i,j] * 0.5 * (P_i - R P_j) / L_cross[i,j]
    # grad_P[j] += P_dist[i,j] * 0.5 * (-R.T (P_i - R P_j)) / L_cross[i,j]
    
    # Vectorized:
    # U_cross[i,j] = (P_i - R P_j) / L_cross[i,j]
    # This is D_cross / L_cross[:, :, np.newaxis]
    L_cross_safe = np.where(L_cross > 1e-12, L_cross, 1.0)
    U_cross = D_cross / L_cross_safe[:, :, np.newaxis]
    
    # Weight matrix
    W_cross = P_pairs_cross * 0.5
    
    # Sum over j for grad_P[i]
    # grad_P[i] = sum_j W_cross[i,j] * U_cross[i,j]
    grad_cross_i = np.sum(W_cross[:, :, np.newaxis] * U_cross, axis=1)
    
    # Sum over i for grad_P[j]
    # grad_P[j] = sum_i W_cross[i,j] * (-R.T U_cross[i,j])
    # Note: U_cross[i,j] is a vector.
    # -R.T U_cross[i,j] = -U_cross[i,j] @ R (since R is orthogonal, R.T = R^-1)
    # Actually, let's just compute the term.
    # Term for P_j is -R.T @ U_cross[i,j]
    # Sum over i: -sum_i W_cross[i,j] * (R.T @ U_cross[i,j])
    # = - (sum_i W_cross[i,j] * U_cross[i,j]) @ R
    # Let V = sum_i W_cross[i,j] * U_cross[i,j] (vector)
    # Then grad_P[j] = -V @ R = -R.T @ V
    
    # V_j = sum_i W_cross[i,j] * U_cross[i,j]
    # This is a matrix multiplication: W_cross.T @ U_cross? No.
    # U_cross is (N, N, 2). W_cross is (N, N).
    # We want sum_i W_cross[i,j] * U_cross[i,j] for each j.
    # This is np.sum(W_cross[:, :, np.newaxis] * U_cross, axis=0)
    V = np.sum(W_cross[:, :, np.newaxis] * U_cross, axis=0)
    
    # Now apply rotation to V
    # grad_P[j] = -R.T @ V_j
    # R.T @ V_j is (R.T @ V_j)
    # We can do this for all j at once:
    # V is (N, 2). We want (N, 2) result where row j is -R.T @ V_j.
    # This is -V @ R
    grad_cross_j = -V @ R_rot
    
    grad += grad_cross_i
    grad += grad_cross_j
    
    # Gradient for walls
    P_walls = P_dist[len(i_idx)+N_SECTOR*N_SECTOR:]
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -P / L_wall_safe[:, np.newaxis]
    grad += P_walls[:, np.newaxis] * U_walls
    
    return -val, -grad

def _hex_lattice_init_sector(n_points, seed=42):
    """Generate a hexagonal lattice of points inside the unit circle, restricted to the sector."""
    rng = np.random.RandomState(seed)
    s = np.sqrt(2 * np.pi / (N_FULL * np.sqrt(3)))
    k = int(1.0 / s) + 2
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                angle = np.arctan2(y, x)
                if angle < 0: angle += 2 * np.pi
                if 0 <= angle <= np.pi / 3:
                    points.append([x, y])
                
    points = np.array(points)
    
    if len(points) > n_points:
        # Keep the n_points closest to the center? Or random?
        # Random selection might be better to break symmetry locally
        indices = rng.choice(len(points), n_points, replace=False)
        points = points[indices]
    elif len(points) < n_points:
        # Add random points in the sector
        while len(points) < n_points:
            r = np.sqrt(rng.random())
            a = rng.random() * (np.pi / 3)
            points = np.vstack([points, [[r * np.cos(a), r * np.sin(a)]]])
            
    points += rng.normal(0, s * 0.01, size=points.shape)
    return points

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    best_points = None
    best_score = -np.inf
    
    # Multi-start with different initializations
    inits = []
    for seed in [42, 43, 44, 45, 46, 47, 48, 49]:
        inits.append(_hex_lattice_init_sector(N_SECTOR, seed=seed))
        
    # Also try a random init
    rng = np.random.RandomState(100)
    r = np.sqrt(rng.random(N_SECTOR))
    a = rng.random(N_SECTOR) * (np.pi / 3)
    inits.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    
    for idx, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        current = init.copy()
        
        # Anneal beta
        for beta in [50.0, 100.0, 200.0, 500.0, 1000.0]:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
            
            def obj_and_grad(x):
                if time.time() - start_time > time_limit:
                    raise TimeoutError()
                return _soft_min_obj_and_grad_sector(x, beta)
            
            try:
                res = minimize(obj_and_grad, current.flatten(), method='L-BFGS-B', jac=True,
                               options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-12})
                if res.success or res.status in [1, 2, 4]:
                    current = res.x.reshape(N_SECTOR, 2)
            except Exception:
                pass
            
        # Final polish
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        try:
            res = minimize(lambda x: _soft_min_obj_and_grad_sector(x, 2000.0)[0], 
                           current.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 300, 'gtol': 1e-8})
            if res.success or res.status in [1, 2, 4]:
                current = res.x.reshape(N_SECTOR, 2)
        except Exception:
            pass
        
        # Generate full config and score
        full_points = _generate_full_config(current)
        score = _compute_true_score_full(full_points)
        
        if score > best_score:
            best_score = score
            best_points = full_points.copy()
            
    # If no points found (shouldn't happen), return random
    if best_points is None:
        rng = np.random.RandomState(0)
        r = np.sqrt(rng.random(N_FULL))
        a = rng.random(N_FULL) * 2 * np.pi
        best_points = np.column_stack([r * np.cos(a), r * np.sin(a)])
        
    # Ensure inside
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] * (0.9999 / norms[mask])[:, np.newaxis]
        
    return best_points
