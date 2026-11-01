"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.optimize import NonlinearConstraint


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _energy(params, temperature=0.02):
    """Smooth energy function to maximize minimum distance."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Use pdist for all pairwise distances
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    T = max(temperature, 1e-8)
    
    # Soft min of pairwise distances - only consider close pairs for speed
    threshold = min_dist * 1.15 
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        # Numerically stable soft min
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    
    # For boundary distances: soft min
    b_min = np.min(boundary_dists)
    b_threshold = b_min * 1.15
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
    
    # Combined soft min of r_pair and r_bound
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    
    return -soft_min_combined


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 20):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        elif len(points) > 0 and best_c is None:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _local_optimize(c0, max_time=10.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    if temperature_schedule is None:
        configs = [0.02, 0.01, 0.005, 0.002]
    else:
        configs = list(temperature_schedule)
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 1:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 100}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def _slsqp_critical_polish(c0, r_target, max_time=10.0, num_critical_pairs=50):
    """
    Polish using SLSQP on a subset of critical constraints.
    This is faster than full SLSQP and helps tighten the packing.
    """
    start_time = time.time()
    n = c0.shape[0]
    m = n * 2
    
    # Identify critical pairs: those with distance close to 2*r_target
    dists = squareform(pdist(c0, metric='euclidean'))
    
    # Get indices of pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Select the smallest 'num_critical_pairs' distances
    boundary_dists = 0.5 - np.abs(c0)
    min_boundary_dist = np.min(boundary_dists)
    
    # Sort pairs by distance
    sorted_indices = np.argsort(pair_dists)
    critical_pair_indices = sorted_indices[:num_critical_pairs]
    
    ci_idx = i_idx[critical_pair_indices]
    cj_idx = j_idx[critical_pair_indices]
    
    # Identify critical boundary points: points where boundary dist is small
    b_thresh = min_boundary_dist * 1.1 + 1e-6
    critical_boundary_points = np.where(np.min(boundary_dists, axis=1) < b_thresh)[0]
    
    n_crit_pairs = len(ci_idx)
    n_crit_bounds = len(critical_boundary_points)
    
    if n_crit_pairs == 0 and n_crit_bounds == 0:
        return c0, r_target

    # Define constraints only for critical parts
    def pairwise_constraints(params):
        c = params.reshape(n, 2)
        ci = c[ci_idx]
        cj = c[cj_idx]
        diff = ci - cj
        dists_sub = np.sqrt(np.sum(diff**2, axis=1))
        return dists_sub - 2.0 * r_target
    
    def grad_pairwise_constraints(params):
        c = params.reshape(n, 2)
        ci = c[ci_idx]
        cj = c[cj_idx]
        diff = ci - cj
        dists_sub = np.sqrt(np.sum(diff**2, axis=1))
        dists_safe = np.where(dists_sub > 1e-12, dists_sub, 1e-12)
        inv_d = 1.0 / dists_safe
        
        J = np.zeros((n_crit_pairs, m))
        for k in range(n_crit_pairs):
            i = ci_idx[k]
            j = cj_idx[k]
            d = dists_sub[k]
            if d < 1e-12: continue
            inv_dk = 1.0/d
            dx = (ci[k, 0] - cj[k, 0]) * inv_dk
            dy = (ci[k, 1] - cj[k, 1]) * inv_dk
            
            J[k, 2*i] += dx
            J[k, 2*i+1] += dy
            J[k, 2*j] -= dx
            J[k, 2*j+1] -= dy
        return J

    def boundary_constraints(params):
        c = params.reshape(n, 2)
        x = c[:, 0]
        y = c[:, 1]
        
        res = []
        for idx in critical_boundary_points:
            res.append(x[idx] + 0.5 - r_target)
            res.append(0.5 - r_target - x[idx])
            res.append(y[idx] + 0.5 - r_target)
            res.append(0.5 - r_target - y[idx])
        return np.array(res)
    
    def grad_boundary_constraints(params):
        J = np.zeros((4 * n_crit_bounds, m))
        for k, idx in enumerate(critical_boundary_points):
            J[4*k, 2*idx] = 1.0
            J[4*k+1, 2*idx] = -1.0
            J[4*k+2, 2*idx+1] = 1.0
            J[4*k+3, 2*idx+1] = -1.0
        return J

    constraints = []
    if n_crit_pairs > 0:
        constraints.append(NonlinearConstraint(pairwise_constraints, 0.0, np.inf, jac=grad_pairwise_constraints))
    if n_crit_bounds > 0:
        constraints.append(NonlinearConstraint(boundary_constraints, 0.0, np.inf, jac=grad_boundary_constraints))
    
    def objective(params):
        return 0.0
    def grad_objective(params):
        return np.zeros(m)
    
    bounds = [(-0.5 + r_target - 0.01, 0.5 - r_target + 0.01)] * m
    
    try:
        result = minimize(
            objective,
            c0.flatten(),
            method='SLSQP',
            jac=grad_objective,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        c_new = np.clip(result.x.reshape(n, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        
        if r_new > r_target:
            return c_new, r_new
        else:
            return c0, r_target
    except Exception:
        return c0, r_target


def _soft_contact_dynamics(c0, r_target, max_time=8.0, stiffness=500.0, damping=5.0, dt_init=0.002):
    """
    Rigid-body dynamics with soft-contact model to settle circles into equilibrium.
    Uses symplectic Euler with adaptive time-stepping.
    """
    start_time = time.time()
    n = c0.shape[0]
    c = c0.copy().astype(float)
    v = np.zeros_like(c)
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    dt = dt_init
    min_dt = 1e-5
    max_dt = 0.005
    
    # Boundary walls
    half_side = 0.5
    
    for step in range(20000):
        if time.time() > start_time + max_time:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion for overlapping or close pairs
        diff = c[i_idx] - c[j_idx]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        dists_safe = np.maximum(dists, 1e-10)
        
        # Soft contact: force = stiffness * (2*r_target - dist) for dist < 2*r_target
        overlap = 2.0 * r_target - dists
        # Only apply force where overlapping (positive overlap)
        mask = overlap > 0
        if np.any(mask):
            # Force magnitude proportional to overlap (linear spring)
            f_mag = stiffness * overlap[mask]
            # Direction: away from each other
            dir_vec = diff[mask] / dists_safe[mask, None]
            # Apply to both particles
            np.add.at(forces, i_idx[mask], -f_mag[:, None] * dir_vec)
            np.add.at(forces, j_idx[mask], f_mag[:, None] * dir_vec)
        
        # Boundary forces (soft walls)
        # For each particle, check distance to each wall
        for dim in range(2):
            # Left wall: x + 0.5 < r_target
            overlap_left = r_target - (c[:, dim] + 0.5)
            mask_left = overlap_left > 0
            if np.any(mask_left):
                forces[mask_left, dim] += stiffness * overlap_left[mask_left]
            
            # Right wall: 0.5 - x < r_target
            overlap_right = r_target - (0.5 - c[:, dim])
            mask_right = overlap_right > 0
            if np.any(mask_right):
                forces[mask_right, dim] -= stiffness * overlap_right[mask_right]
        
        # Symplectic Euler integration
        v += forces * dt
        v *= np.exp(-damping * dt)  # Damping
        c += v * dt
        
        # Clip to stay within bounds (hard constraint)
        c = np.clip(c, -half_side + 1e-12, half_side - 1e-12)
        
        # Adaptive time-stepping: reduce dt if velocity is large
        max_speed = np.max(np.sqrt(np.sum(v**2, axis=1)))
        if max_speed > 0.5:
            dt = max(min_dt, dt * 0.5)
        elif max_speed < 0.05 and dt < max_dt:
            dt = min(max_dt, dt * 1.2)
        
        # Check convergence: if all velocities are small
        if max_speed < 1e-4 and step > 100:
            break
    
    c = np.clip(c, -half_side + 1e-12, half_side - 1e-12)
    r_new = _compute_radius(c)
    
    if r_new > r_target:
        return c, r_new
    else:
        return c0, r_target


def _bundle_method(c0, r_target, max_time=8.0, num_subproblems=5):
    """
    Bundle method: solve subproblems on subsets of critical constraints
    to tighten the packing.
    """
    start_time = time.time()
    n = c0.shape[0]
    best_c = c0.copy()
    best_r = r_target
    
    # Get all pair distances
    dists = squareform(pdist(c0, metric='euclidean'))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Sort pairs by distance (most critical first)
    sorted_indices = np.argsort(pair_dists)
    
    # Divide into subproblems
    chunk_size = max(10, len(sorted_indices) // num_subproblems)
    
    for sp in range(num_subproblems):
        if time.time() > start_time + max_time:
            break
        
        # Select a chunk of critical pairs
        start_idx = sp * chunk_size
        end_idx = min(start_idx + chunk_size, len(sorted_indices))
        if start_idx >= len(sorted_indices):
            break
        
        sub_indices = sorted_indices[start_idx:end_idx]
        ci_idx = i_idx[sub_indices]
        cj_idx = j_idx[sub_indices]
        
        # Also include boundary constraints for points near boundaries
        boundary_dists = 0.5 - np.abs(best_c)
        min_boundary_dist = np.min(boundary_dists)
        b_thresh = min_boundary_dist * 1.1 + 1e-6
        critical_boundary_points = np.where(np.min(boundary_dists, axis=1) < b_thresh)[0]
        
        n_crit_pairs = len(ci_idx)
        n_crit_bounds = len(critical_boundary_points)
        
        if n_crit_pairs == 0 and n_crit_bounds == 0:
            continue
        
        m = n * 2
        
        def pairwise_constraints(params):
            c = params.reshape(n, 2)
            ci = c[ci_idx]
            cj = c[cj_idx]
            diff = ci - cj
            dists_sub = np.sqrt(np.sum(diff**2, axis=1))
            return dists_sub - 2.0 * best_r
        
        def grad_pairwise_constraints(params):
            c = params.reshape(n, 2)
            ci = c[ci_idx]
            cj = c[cj_idx]
            diff = ci - cj
            dists_sub = np.sqrt(np.sum(diff**2, axis=1))
            dists_safe = np.where(dists_sub > 1e-12, dists_sub, 1e-12)
            
            J = np.zeros((n_crit_pairs, m))
            for k in range(n_crit_pairs):
                i = ci_idx[k]
                j = cj_idx[k]
                d = dists_sub[k]
                if d < 1e-12: continue
                inv_dk = 1.0/d
                dx = (ci[k, 0] - cj[k, 0]) * inv_dk
                dy = (ci[k, 1] - cj[k, 1]) * inv_dk
                
                J[k, 2*i] += dx
                J[k, 2*i+1] += dy
                J[k, 2*j] -= dx
                J[k, 2*j+1] -= dy
            return J
        
        def boundary_constraints(params):
            c = params.reshape(n, 2)
            x = c[:, 0]
            y = c[:, 1]
            
            res = []
            for idx in critical_boundary_points:
                res.append(x[idx] + 0.5 - best_r)
                res.append(0.5 - best_r - x[idx])
                res.append(y[idx] + 0.5 - best_r)
                res.append(0.5 - best_r - y[idx])
            return np.array(res)
        
        def grad_boundary_constraints(params):
            J = np.zeros((4 * n_crit_bounds, m))
            for k, idx in enumerate(critical_boundary_points):
                J[4*k, 2*idx] = 1.0
                J[4*k+1, 2*idx] = -1.0
                J[4*k+2, 2*idx+1] = 1.0
                J[4*k+3, 2*idx+1] = -1.0
            return J
        
        constraints = []
        if n_crit_pairs > 0:
            constraints.append(NonlinearConstraint(pairwise_constraints, 0.0, np.inf, jac=grad_pairwise_constraints))
        if n_crit_bounds > 0:
            constraints.append(NonlinearConstraint(boundary_constraints, 0.0, np.inf, jac=grad_boundary_constraints))
        
        def objective(params):
            return 0.0
        def grad_objective(params):
            return np.zeros(m)
        
        bounds = [(-0.5 + best_r - 0.01, 0.5 - best_r + 0.01)] * m
        
        try:
            result = minimize(
                objective,
                best_c.flatten(),
                method='SLSQP',
                jac=grad_objective,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            c_new = np.clip(result.x.reshape(n, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        except Exception:
            continue
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization from best initial
    if time.time() < deadline - 20:
        max_opt_time = min(15.0, deadline - time.time() - 5)
        c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: A few random restarts
    np.random.seed(42)
    n_restarts = 3
    
    for trial in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Jittered grid initialization
        m = int(np.ceil(np.sqrt(N)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(-0.45, 0.45, m),
            np.linspace(-0.45, 0.45, m)
        )
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        np.random.shuffle(grid)
        c_init = grid[:N].copy()
        
        # Add random perturbation
        c_init += np.random.normal(0, 0.01, size=(N, 2))
        c_init = np.clip(c_init, -0.4999, 0.4999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimize
        remaining = deadline - time.time() - 5
        if remaining > 5:
            max_opt_time = min(10.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 4: Final polish with very low temperatures for high precision
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 10
        if remaining > 5:
            c_pol, r_pol = _local_optimize(best_c, max_time=min(10.0, remaining), 
                                           temperature_schedule=[0.001, 0.0005, 0.0002, 0.0001])
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Step 5: Soft-contact dynamics polish
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 3:
            c_dyn, r_dyn = _soft_contact_dynamics(best_c, best_radius, max_time=min(8.0, remaining))
            if r_dyn > best_radius:
                best_radius = r_dyn
                best_c = c_dyn.copy()
    
    # Step 6: Critical Pair SLSQP Polish
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        
        c_pol, r_pol = _slsqp_critical_polish(best_c, best_radius, max_time=5.0, num_critical_pairs=100)
        
        if r_pol > best_radius:
            best_radius = r_pol
            best_c = c_pol.copy()
        else:
            break
    
    # Step 7: Bundle method polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            c_bundle, r_bundle = _bundle_method(best_c, best_radius, max_time=min(6.0, remaining))
            if r_bundle > best_radius:
                best_radius = r_bundle
                best_c = c_bundle.copy()
    
    # Step 8: Ultra-precise final refinement (L-BFGS-B with very low T)
    if time.time() < deadline - 2:
        remaining = deadline - time.time() - 0.5
        if remaining > 2:
            def _ultra_polish(c_init, max_time):
                start_t = time.time()
                best_c_local = c_init.copy()
                best_r_local = _compute_radius(c_init)
                
                temps = [0.0001, 0.00005, 0.00002, 0.00001]
                current_c = c_init.copy()
                
                for T in temps:
                    if time.time() > start_t + max_time - 0.3:
                        break
                    
                    def obj(params, _T=T):
                        return _energy(params, temperature=_T)
                    
                    try:
                        result = minimize(
                            obj,
                            current_c.flatten(),
                            method='L-BFGS-B',
                            bounds=[(-0.4999, 0.4999)] * (2 * N),
                            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 200}
                        )
                        
                        c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
                        r_new = _compute_radius(c_new)
                        
                        if r_new > best_r_local:
                            best_c_local = c_new.copy()
                            best_r_local = r_new
                            current_c = c_new
                        else:
                            current_c = c_new
                    except Exception:
                        continue
                
                return best_c_local, best_r_local
            
            c_final, r_final = _ultra_polish(best_c, max_time=remaining)
            if r_final > best_radius:
                best_radius = r_final
                best_c = c_final.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
