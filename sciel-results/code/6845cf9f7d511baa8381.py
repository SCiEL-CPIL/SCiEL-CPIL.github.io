import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Generate initial hexagonal lattice
    def gen_lattice(spacing):
        points = []
        range_val = int(1.5 / spacing) + 2
        for j in range(-range_val, range_val + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            if abs(y) > 1.2:
                continue
            offset = 0.0
            if j % 2 != 0:
                offset = spacing / 2.0
            for i in range(-range_val, range_val + 1):
                x = i * spacing + offset
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return None
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists)
        return points[indices[:N]]

    x = None
    for sp in [0.12, 0.11, 0.10, 0.09]:
        x = gen_lattice(sp)
        if x is not None:
            break
    if x is None:
        x = np.random.randn(N, 2) * 0.5

    # Helper to calculate score (radius)
    def calc_radius(pts):
        # Pairwise distances
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_dist = np.min(d) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_dist, min_boundary)

    best_x = x.copy()
    best_r = calc_radius(x)

    # 2. Phase 1: Soft-min gradient ascent (from current program)
    # This provides a good global structure quickly.
    def compute_gradient(pts, beta):
        grads = np.zeros_like(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        norms = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - norms

        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_dists = dists[mask]
        pair_weights = np.exp(-beta * pair_dists / 2.0)
        bnd_dists = boundary_dists
        bnd_weights = np.exp(-beta * bnd_dists)

        total_weight = np.sum(pair_weights) + np.sum(bnd_weights)
        if total_weight == 0:
            return grads

        W = np.zeros((N, N))
        W[mask] = pair_weights

        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        S = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
        pair_grads = -beta / 2.0 * S

        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        bnd_unit = pts / safe_norms[:, np.newaxis]
        bnd_grads = beta * bnd_weights[:, np.newaxis] * bnd_unit

        total_grad = pair_grads + bnd_grads
        grad_F = (1.0 / (beta * total_weight)) * total_grad
        return -grad_F

    step_size = 0.05
    beta = 10.0
    x_curr = x.copy()

    while time.time() < deadline - 40.0:
        grad = compute_gradient(x_curr, beta)
        if not np.all(np.isfinite(grad)):
            beta *= 0.5
            continue

        x_new = x_curr + step_size * grad
        if np.any(np.linalg.norm(x_new, axis=1) > 1.0):
            step_size *= 0.5
            x_new = x_curr + step_size * grad
            if np.any(np.linalg.norm(x_new, axis=1) > 1.0):
                continue

        r_new = calc_radius(x_new)
        if r_new > best_r:
            best_r = r_new
            best_x = x_new
            x_curr = x_new
            beta *= 1.1
            step_size = min(step_size * 1.05, 0.1)
        else:
            step_size *= 0.9
            beta *= 1.05

        if step_size < 1e-6:
            break

    # 3. Phase 2: Local Refinement using Nelder-Mead on subsets
    # The soft-min method gets stuck in local minima where the "soft" minimum
    # doesn't perfectly align with the "hard" minimum constraints.
    # We identify the most critical constraints (tightest pairs and boundary points)
    # and optimize only those points using a local solver.

    def identify_critical_points(pts, r_target):
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        # Find pairs with distance < r_target * 1.05
        crit_pairs = np.argwhere(d < r_target * 1.05)
        crit_indices = set()
        for i, j in crit_pairs:
            if i != j:
                crit_indices.add(int(i))
                crit_indices.add(int(j))
        
        # Find points close to boundary
        norms = np.linalg.norm(pts, axis=1)
        crit_boundary = np.argwhere(1.0 - norms < r_target * 1.05)
        for idx in crit_boundary:
            crit_indices.add(int(idx[0]))
            
        return list(crit_indices)

    def local_objective(pts, fixed_indices, active_indices):
        # pts is the full array (view or copy)
        # We only optimize active_indices, but objective depends on all
        # This function is for scipy.minimize which expects a flat vector of variables
        pass # Not used directly, we wrap it

    # We will use scipy.optimize.minimize with Nelder-Mead on the critical points
    # We need to map the flat vector back to the 2D array
    
    def objective_and_grads_for_local(pts_flat, active_indices, fixed_pts):
        # Reconstruct full points
        pts = fixed_pts.copy()
        pts[active_indices] = pts_flat.reshape(-1, 2)
        
        # Calculate radius
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_dist = np.min(d) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        r = min(min_dist, min_boundary)
        
        # We want to maximize r, so minimize -r
        return -r

    # Time left for local refinement
    time_left = deadline - time.time()
    
    # Try a few rounds of local refinement on the best solution found
    # We'll pick the top K critical points and optimize them
    K_local = 20
    
    current_x = best_x.copy()
    current_r = best_r
    
    # Sort critical points by "tightness" to prioritize the most constrained ones
    for round_num in range(5):
        if time.time() > deadline - 5.0:
            break
            
        crit_indices = identify_critical_points(current_x, current_r)
        
        # If no critical points (unlikely), break
        if not crit_indices:
            break
            
        # Select up to K_local points
        # Sort by distance to boundary or min pair distance to pick the worst offenders
        # For simplicity, just take the first K found, or sort by norm (boundary)
        # Better: sort by how "tight" they are.
        
        # Calculate tightness for each critical point
        # Tightness = min(distance to other points, distance to boundary)
        d_mat = cdist(current_x, current_x)
        np.fill_diagonal(d_mat, np.inf)
        min_pair_dist = np.min(d_mat, axis=1) / 2.0
        min_bound_dist = 1.0 - np.linalg.norm(current_x, axis=1)
        
        # For critical points, we care about the min of these
        tightness = np.zeros(N)
        for idx in crit_indices:
            tightness[idx] = min(min_pair_dist[idx], min_bound_dist[idx])
            
        # Sort critical indices by tightness (ascending, so most tight first)
        crit_indices_sorted = sorted(crit_indices, key=lambda i: tightness[i])
        
        # Take the worst K_local
        active_indices = np.array(crit_indices_sorted[:K_local])
        fixed_indices = np.array([i for i in range(N) if i not in active_indices])
        
        # Initial guess for active points
        x0 = current_x[active_indices].flatten()
        fixed_pts = current_x.copy()
        
        # Define objective for scipy
        def obj_local(x_flat):
            pts = fixed_pts.copy()
            pts[active_indices] = x_flat.reshape(-1, 2)
            
            # Check if any point went outside (penalty)
            norms = np.linalg.norm(pts, axis=1)
            if np.any(norms > 1.0):
                return 1e6 + np.sum(norms[norms > 1.0] - 1.0)
                
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            min_dist = np.min(d) / 2.0
            min_boundary = np.min(1.0 - norms)
            r = min(min_dist, min_boundary)
            return -r
            
        # Run Nelder-Mead
        try:
            res = minimize(
                obj_local, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res.success or res.fun < obj_local(x0):
                new_x = fixed_pts.copy()
                new_x[active_indices] = res.x.reshape(-1, 2)
                
                # Verify validity
                if np.all(np.linalg.norm(new_x, axis=1) <= 1.0):
                    r_new = calc_radius(new_x)
                    if r_new > current_r:
                        current_r = r_new
                        current_x = new_x
                        if current_r > best_r:
                            best_r = current_r
                            best_x = current_x
        except Exception:
            continue

    # 4. Final Polish: Random perturbations + small step gradient
    if time.time() < deadline - 2.0:
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            # Small random perturbation
            noise = np.random.randn(N, 2) * 0.005
            x_pert = best_x + noise
            if np.any(np.linalg.norm(x_pert, axis=1) > 1.0):
                # Project back
                norms = np.linalg.norm(x_pert, axis=1)
                mask = norms > 1.0
                x_pert[mask] /= norms[mask, np.newaxis]
            
            r_pert = calc_radius(x_pert)
            if r_pert > best_r:
                best_r = r_pert
                best_x = x_pert

    return best_x

# Keep the original function name and signature
def construct():
    return _construct_impl()

def _construct_impl():
    # The logic above is in the local scope of the previous draft. 
    # Let's wrap it properly.
    import time
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 300
    
    def gen_lattice(spacing):
        points = []
        range_val = int(1.5 / spacing) + 2
        for j in range(-range_val, range_val + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            if abs(y) > 1.2:
                continue
            offset = 0.0
            if j % 2 != 0:
                offset = spacing / 2.0
            for i in range(-range_val, range_val + 1):
                x = i * spacing + offset
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
             return None
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists)
        return points[indices[:N]]

    x = None
    for sp in [0.12, 0.11, 0.10, 0.09]:
        x = gen_lattice(sp)
        if x is not None:
            break
            
    if x is None:
        x = np.random.randn(N, 2) * 0.5

    def calc_radius(pts):
        # Pairwise distances
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_dist = np.min(d) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_dist, min_boundary)

    best_x = x.copy()
    best_r = calc_radius(x)
    
    def compute_gradient(pts, beta):
        grads = np.zeros_like(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        norms = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - norms
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_dists = dists[mask]
        pair_weights = np.exp(-beta * pair_dists / 2.0)
        bnd_dists = boundary_dists
        bnd_weights = np.exp(-beta * bnd_dists)
        
        total_weight = np.sum(pair_weights) + np.sum(bnd_weights)
        if total_weight == 0:
             return grads
             
        W = np.zeros((N, N))
        W[mask] = pair_weights
        
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        S = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
        pair_grads = -beta / 2.0 * S
        
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        bnd_unit = pts / safe_norms[:, np.newaxis]
        bnd_grads = beta * bnd_weights[:, np.newaxis] * bnd_unit
        
        total_grad = pair_grads + bnd_grads
        grad_F = (1.0 / (beta * total_weight)) * total_grad
        return -grad_F

    step_size = 0.05
    beta = 10.0
    
    x_curr = x.copy()
    
    while time.time() < deadline - 40.0:
        grad = compute_gradient(x_curr, beta)
        
        if not np.all(np.isfinite(grad)):
            beta *= 0.5
            continue
            
        x_new = x_curr + step_size * grad
        
        if np.any(np.linalg.norm(x_new, axis=1) > 1.0):
             step_size *= 0.5
             x_new = x_curr + step_size * grad
             if np.any(np.linalg.norm(x_new, axis=1) > 1.0):
                 continue
                 
        r_new = calc_radius(x_new)
        
        if r_new > best_r:
            best_r = r_new
            best_x = x_new
            x_curr = x_new
            beta *= 1.1
            step_size = min(step_size * 1.05, 0.1)
        else:
            step_size *= 0.9
            beta *= 1.05
            
        if step_size < 1e-6:
             break

    # Local Refinement Phase
    def identify_critical_points(pts, r_target):
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        crit_pairs = np.argwhere(d < r_target * 1.05)
        crit_indices = set()
        for i, j in crit_pairs:
            if i != j:
                crit_indices.add(int(i))
                crit_indices.add(int(j))
        
        norms = np.linalg.norm(pts, axis=1)
        crit_boundary = np.argwhere(1.0 - norms < r_target * 1.05)
        for idx in crit_boundary:
            crit_indices.add(int(idx[0]))
            
        return list(crit_indices)

    current_x = best_x.copy()
    current_r = best_r
    K_local = 20
    
    for round_num in range(5):
        if time.time() > deadline - 5.0:
            break
            
        crit_indices = identify_critical_points(current_x, current_r)
        if not crit_indices:
            break
            
        d_mat = cdist(current_x, current_x)
        np.fill_diagonal(d_mat, np.inf)
        min_pair_dist = np.min(d_mat, axis=1) / 2.0
        min_bound_dist = 1.0 - np.linalg.norm(current_x, axis=1)
        
        tightness = np.zeros(N)
        for idx in crit_indices:
            tightness[idx] = min(min_pair_dist[idx], min_bound_dist[idx])
            
        crit_indices_sorted = sorted(crit_indices, key=lambda i: tightness[i])
        active_indices = np.array(crit_indices_sorted[:K_local])
        fixed_indices = np.array([i for i in range(N) if i not in active_indices])
        
        x0 = current_x[active_indices].flatten()
        fixed_pts = current_x.copy()
        
        def obj_local(x_flat):
            pts = fixed_pts.copy()
            pts[active_indices] = x_flat.reshape(-1, 2)
            
            norms = np.linalg.norm(pts, axis=1)
            if np.any(norms > 1.0):
                return 1e6 + np.sum(norms[norms > 1.0] - 1.0)
                
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            min_dist = np.min(d) / 2.0
            min_boundary = np.min(1.0 - norms)
            r = min(min_dist, min_boundary)
            return -r
            
        try:
            res = minimize(
                obj_local, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res.success or res.fun < obj_local(x0):
                new_x = fixed_pts.copy()
                new_x[active_indices] = res.x.reshape(-1, 2)
                
                if np.all(np.linalg.norm(new_x, axis=1) <= 1.0):
                    r_new = calc_radius(new_x)
                    if r_new > current_r:
                        current_r = r_new
                        current_x = new_x
                        if current_r > best_r:
                            best_r = current_r
                            best_x = current_x
        except Exception:
            continue

    if time.time() < deadline - 2.0:
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            noise = np.random.randn(N, 2) * 0.005
            x_pert = best_x + noise
            if np.any(np.linalg.norm(x_pert, axis=1) > 1.0):
                norms = np.linalg.norm(x_pert, axis=1)
                mask = norms > 1.0
                x_pert[mask] /= norms[mask, np.newaxis]
            
            r_pert = calc_radius(x_pert)
            if r_pert > best_r:
                best_r = r_pert
                best_x = x_pert

    return best_x

# The task requires keeping function names. The seed had `construct`.
# I will define `construct` to call the implementation.
