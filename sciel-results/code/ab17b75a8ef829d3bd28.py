import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_lattice_init(n_points, rotation=0.0):
    """Generate a hexagonal lattice of points inside the unit circle."""
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    k = int(1.0 / s) + 2
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if rotation != 0.0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        points = points @ R.T
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n_points]]
    
    return points

def _compute_true_score(points):
    """Compute the true packing radius."""
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    return min(min_pair, min_wall)

def _build_orbits_from_lattice(points, n_sym=12):
    """
    Group lattice points into orbits under n_sym-fold rotational symmetry.
    Returns: orbit_reps (list of representative points), orbit_sizes (list of orbit sizes)
    """
    # We'll use the symmetry to group points
    # For 12-fold symmetry, points related by rotation by pi/6 are in the same orbit
    
    # First, let's identify unique orbits by looking at the angular distribution
    # A simpler approach: use the lattice structure directly
    
    # Generate all 12 rotations
    angles = np.arange(n_sym) * (2 * np.pi / n_sym)
    
    # For each point, find its "canonical" representative (smallest angle in [0, 2pi/n_sym))
    # and group them
    
    # Actually, let's use a different approach: 
    # The hexagonal lattice has 6-fold symmetry. With 12-fold container symmetry,
    # we need to be careful. Let's just use the lattice points and optimize 
    # a global rotation plus radial scaling.
    
    # Better approach: identify the "fundamental domain" points
    # For 12-fold symmetry, the fundamental sector is 30 degrees (pi/6)
    
    # Let's group points by their orbit under the dihedral group D_12
    # For simplicity, we'll use rotational symmetry only
    
    # Find unique orbits
    used = np.zeros(len(points), dtype=bool)
    orbit_reps = []
    orbit_members = []
    
    for i in range(len(points)):
        if used[i]:
            continue
        
        # Find all points that are rotations of points[i]
        members = [i]
        used[i] = True
        
        for rot_idx in range(1, n_sym):
            angle = rot_idx * (2 * np.pi / n_sym)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated = points[i] @ R.T
            
            # Find closest point to rotated
            dists = np.linalg.norm(points - rotated, axis=1)
            j = np.argmin(dists)
            
            if not used[j] and dists[j] < 0.01 * np.linalg.norm(points[i]):
                members.append(j)
                used[j] = True
        
        orbit_reps.append(points[i])
        orbit_members.append(members)
    
    return orbit_reps, orbit_members

def _soft_min_obj_and_grad(x, beta, orbit_reps, orbit_sizes, n_sym=12):
    """
    Compute the negative soft-min value and its gradient for the symmetric packing.
    x contains: [rotation_angle, radial_scales...]
    """
    n_orbits = len(orbit_reps)
    
    # x[0] is the global rotation angle
    # x[1:] are radial scale factors for each orbit
    
    rot_angle = x[0]
    radial_scales = x[1:]
    
    # Build all points from orbits
    all_points = np.zeros((N, 2))
    idx = 0
    
    for o, (rep, size) in enumerate(zip(orbit_reps, orbit_sizes)):
        # Scale the representative radially
        r = np.linalg.norm(rep)
        if r > 1e-12:
            dir_vec = rep / r
        else:
            dir_vec = rep  # zero vector
        
        scaled_r = r * radial_scales[o]
        
        # Generate all points in this orbit
        for k in range(size):
            if idx >= N:
                break
            angle = k * (2 * np.pi / n_sym) + rot_angle
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            if r > 1e-12:
                base_point = dir_vec * scaled_r
            else:
                base_point = np.zeros(2)
            
            all_points[idx] = base_point @ R.T
            idx += 1
    
    # Compute soft-min objective
    D = all_points[:, np.newaxis, :] - all_points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    L_safe = np.where(L > 1e-12, L, 1.0)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs = L_safe[i_idx, j_idx] / 2.0
    
    L_wall = np.linalg.norm(all_points, axis=1)
    x_walls = 1.0 - L_wall
    
    x_all = np.concatenate([x_pairs, x_walls])
    
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    val = min_x - (1.0 / beta) * np.log(Z)
    
    # Gradient computation
    grad = np.zeros_like(x)
    
    # This is complex due to the symmetry structure
    # For now, use numerical gradient as fallback
    return -val, None

def _soft_min_obj_numeric(x, beta, orbit_reps, orbit_sizes, n_sym=12):
    """Numerical gradient version for reliability."""
    n_orbits = len(orbit_reps)
    
    def _build_points(x):
        rot_angle = x[0]
        radial_scales = x[1:]
        
        all_points = np.zeros((N, 2))
        idx = 0
        
        for o, (rep, size) in enumerate(zip(orbit_reps, orbit_sizes)):
            r = np.linalg.norm(rep)
            if r > 1e-12:
                dir_vec = rep / r
            else:
                dir_vec = rep
            
            scaled_r = r * radial_scales[o]
            
            for k in range(size):
                if idx >= N:
                    break
                angle = k * (2 * np.pi / n_sym) + rot_angle
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                
                if r > 1e-12:
                    base_point = dir_vec * scaled_r
                else:
                    base_point = np.zeros(2)
                
                all_points[idx] = base_point @ R.T
                idx += 1
        
        return all_points
    
    def _compute_score(points):
        D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        L = np.linalg.norm(D, axis=2)
        L_safe = np.where(L > 1e-12, L, 1.0)
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        x_pairs = L_safe[i_idx, j_idx] / 2.0
        
        L_wall = np.linalg.norm(points, axis=1)
        x_walls = 1.0 - L_wall
        
        x_all = np.concatenate([x_pairs, x_walls])
        
        min_x = np.min(x_all)
        shifted_x = x_all - min_x
        exp_terms = np.exp(-beta * shifted_x)
        Z = np.sum(exp_terms)
        
        return min_x - (1.0 / beta) * np.log(Z)
    
    points = _build_points(x)
    val = _compute_score(points)
    return -val

def _optimize_symmetric(points_init, time_limit):
    """Optimize using the symmetric parameterization."""
    n_sym = 12
    
    # Build orbits from the initial lattice
    orbit_reps, orbit_members = _build_orbits_from_lattice(points_init, n_sym)
    orbit_sizes = [len(m) for m in orbit_members]
    
    # Verify total count
    total = sum(orbit_sizes)
    if total != N:
        # Fall back to full optimization
        return points_init
    
    n_orbits = len(orbit_reps)
    
    # Initial parameters: rotation=0, all radial scales=1
    x0 = np.zeros(1 + n_orbits)
    x0[1:] = 1.0
    
    start_time = time.time()
    best_x = x0.copy()
    best_score = _compute_true_score(points_init)
    
    # Try different beta values
    betas = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() - start_time > time_limit - 10:
            break
        
        def obj(x):
            return _soft_min_obj_numeric(x, beta, orbit_reps, orbit_sizes, n_sym)
        
        try:
            res = minimize(obj, current_x, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < obj(current_x):
                current_x = res.x
            
            # Evaluate the true score
            rot_angle = current_x[0]
            radial_scales = current_x[1:]
            
            all_points = np.zeros((N, 2))
            idx = 0
            for o, (rep, size) in enumerate(zip(orbit_reps, orbit_sizes)):
                r = np.linalg.norm(rep)
                if r > 1e-12:
                    dir_vec = rep / r
                else:
                    dir_vec = rep
                
                scaled_r = r * radial_scales[o]
                
                for k in range(size):
                    if idx >= N:
                        break
                    angle = k * (2 * np.pi / n_sym) + rot_angle
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    
                    if r > 1e-12:
                        base_point = dir_vec * scaled_r
                    else:
                        base_point = np.zeros(2)
                    
                    all_points[idx] = base_point @ R.T
                    idx += 1
            
            score = _compute_true_score(all_points)
            if score > best_score:
                best_score = score
                best_x = current_x.copy()
                
        except Exception as e:
            pass
    
    # Build final points from best parameters
    rot_angle = best_x[0]
    radial_scales = best_x[1:]
    
    all_points = np.zeros((N, 2))
    idx = 0
    for o, (rep, size) in enumerate(zip(orbit_reps, orbit_sizes)):
        r = np.linalg.norm(rep)
        if r > 1e-12:
            dir_vec = rep / r
        else:
            dir_vec = rep
        
        scaled_r = r * radial_scales[o]
        
        for k in range(size):
            if idx >= N:
                break
            angle = k * (2 * np.pi / n_sym) + rot_angle
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            if r > 1e-12:
                base_point = dir_vec * scaled_r
            else:
                base_point = np.zeros(2)
            
            all_points[idx] = base_point @ R.T
            idx += 1
    
    return all_points

def _full_soft_min_optimize(points, time_limit):
    """Full soft-min optimization without symmetry constraint."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _compute_true_score(points)
    
    def _soft_min_obj_and_grad_full(x, beta):
        C = x.reshape(N, 2)
        
        D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        L = np.linalg.norm(D, axis=2)
        L_safe = np.where(L > 1e-12, L, 1.0)
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        x_pairs = L_safe[i_idx, j_idx] / 2.0
        
        L_wall = np.linalg.norm(C, axis=1)
        x_walls = 1.0 - L_wall
        
        x_all = np.concatenate([x_pairs, x_walls])
        
        min_x = np.min(x_all)
        shifted_x = x_all - min_x
        exp_terms = np.exp(-beta * shifted_x)
        Z = np.sum(exp_terms)
        P = exp_terms / Z
        
        val = min_x - (1.0 / beta) * np.log(Z)
        
        # Gradient
        grad = np.zeros_like(C)
        
        P_pairs = P[:len(i_idx)]
        D_flat = D[i_idx, j_idx]
        L_flat = L[i_idx, j_idx]
        L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
        U = D_flat / L_flat_safe[:, np.newaxis]
        W = P_pairs * 0.5
        
        np.add.at(grad, i_idx, W[:, np.newaxis] * U)
        np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
        
        P_walls = P[len(i_idx):]
        L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
        U_walls = -C / L_wall_safe[:, np.newaxis]
        grad += P_walls[:, np.newaxis] * U_walls
        
        return -val, -grad
    
    beta = 10.0
    max_beta = 3000.0
    beta_factor = 1.5
    
    current_points = points.copy()
    
    while time.time() - start_time < time_limit:
        def obj_and_grad(x):
            return _soft_min_obj_and_grad_full(x, beta)
        
        try:
            res = minimize(obj_and_grad, current_points.flatten(), method='L-BFGS-B', jac=True,
                          options={'maxiter': 100, 'gtol': 1e-5, 'ftol': 1e-12})
            
            new_points = res.x.reshape(N, 2)
            new_score = _compute_true_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
            
            if np.all(np.isfinite(new_points)):
                current_points = new_points
                
        except Exception:
            pass
        
        beta *= beta_factor
        if beta > max_beta:
            break
    
    return best_points

def _polish(points, time_limit):
    """Final high-precision polish pass."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _compute_true_score(points)
    
    # Use very high beta for near-exact min
    beta = 10000.0
    
    def _obj_grad(x):
        C = x.reshape(N, 2)
        
        D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        L = np.linalg.norm(D, axis=2)
        L_safe = np.where(L > 1e-12, L, 1.0)
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        x_pairs = L_safe[i_idx, j_idx] / 2.0
        
        L_wall = np.linalg.norm(C, axis=1)
        x_walls = 1.0 - L_wall
        
        x_all = np.concatenate([x_pairs, x_walls])
        
        min_x = np.min(x_all)
        shifted_x = x_all - min_x
        exp_terms = np.exp(-beta * shifted_x)
        Z = np.sum(exp_terms)
        P = exp_terms / Z
        
        val = min_x - (1.0 / beta) * np.log(Z)
        
        grad = np.zeros_like(C)
        
        P_pairs = P[:len(i_idx)]
        D_flat = D[i_idx, j_idx]
        L_flat = L[i_idx, j_idx]
        L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
        U = D_flat / L_flat_safe[:, np.newaxis]
        W = P_pairs * 0.5
        
        np.add.at(grad, i_idx, W[:, np.newaxis] * U)
        np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
        
        P_walls = P[len(i_idx):]
        L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
        U_walls = -C / L_wall_safe[:, np.newaxis]
        grad += P_walls[:, np.newaxis] * U_walls
        
        return -val, -grad
    
    try:
        res = minimize(_obj_grad, best_points.flatten(), method='L-BFGS-B', jac=True,
                      options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14})
        
        new_points = res.x.reshape(N, 2)
        new_score = _compute_true_score(new_points)
        
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
            
    except Exception:
        pass
    
    return best_points

def construct():
    start_time = time.time()
    time_limit = 135.0  # Leave 15s margin for 150s total
    
    # Initial points from hexagonal lattice
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_true_score(points)
    
    # Try symmetric optimization first (faster, lower dimensionality)
    elapsed = time.time() - start_time
    if elapsed < time_limit - 60:
        sym_points = _optimize_symmetric(points, time_limit - 60)
        sym_score = _compute_true_score(sym_points)
        if sym_score > best_score:
            best_score = sym_score
            best_points = sym_points
    
    # Full soft-min optimization
    elapsed = time.time() - start_time
    if elapsed < time_limit - 20:
        full_points = _full_soft_min_optimize(best_points, time_limit - 20)
        full_score = _compute_true_score(full_points)
        if full_score > best_score:
            best_score = full_score
            best_points = full_points
    
    # Final polish
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        polished = _polish(best_points, time_limit - 5)
        polished_score = _compute_true_score(polished)
        if polished_score > best_score:
            best_points = polished
    
    # Ensure validity
    if not np.all(np.isfinite(best_points)):
        best_points = _hex_lattice_init(N)
    
    # Verify all points are inside the unit circle
    dists = np.linalg.norm(best_points, axis=1)
    if np.max(dists) > 1.0:
        # Scale down if needed
        scale = 0.99 / np.max(dists)
        best_points = best_points * scale
    
    return best_points
