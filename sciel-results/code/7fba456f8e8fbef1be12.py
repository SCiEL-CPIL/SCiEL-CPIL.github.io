import numpy as np
import time
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Precompute pair indices for efficiency
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def compute_score(pts):
        """Compute the packing radius score efficiently."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        return np.min(all_vals)
    
    def soft_min_objective(pts, k, current_min):
        """
        Soft-min surrogate: -log(sum(exp(-k*(d_i - current_min)))) / k
        This is a smooth approximation of min(d_i).
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        d_pair = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_dists = np.concatenate([d_pair, bound_dists])
        
        shifted = all_dists - current_min
        # Clip to avoid numerical issues
        shifted = np.clip(shifted, -100, 100)
        exp_vals = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_vals))
        return -log_sum / k
    
    def soft_min_gradient(pts, k, current_min):
        """
        Gradient of the soft-min objective.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        d_pair = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_dists = np.concatenate([d_pair, bound_dists])
        
        shifted = all_dists - current_min
        shifted = np.clip(shifted, -100, 100)
        exp_vals = np.exp(-k * shifted)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        grads = np.zeros_like(pts)
        
        # Pair contributions
        pair_weights = weights[:num_pairs]
        mask_nonzero = d_pair > 1e-10
        if np.any(mask_nonzero):
            idx_m = mask_nonzero
            vecs = (diff[i_idx[idx_m], j_idx[idx_m]] / 
                   d_pair[idx_m, np.newaxis]) * pair_weights[idx_m, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-10, rad, 1.0)
        bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        return grads
    
    def collision_dynamics(pts, time_budget, dt=0.001, damping=0.98, force_power=4, 
                           target_spacing=None):
        """
        Rigid-body collision dynamics simulation using Hertzian contact law.
        Points are evolved via Verlet integration with repulsive forces proportional
        to overlap depth to the fourth power.
        """
        if target_spacing is None:
            # Estimate target spacing from density
            area_per_point = np.pi / N
            target_spacing = np.sqrt(area_per_point) * 1.05
        
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        end_time = start_time + time_budget
        dt_current = dt
        damping_current = damping
        
        # Precompute pair indices (already have i_idx, j_idx)
        
        while time.time() < end_time:
            # Compute forces
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            d_pair_sq = sq_dists[i_idx, j_idx]
            d_pair = np.sqrt(d_pair_sq)
            
            # Pairwise forces: repulsive when d_pair < 2*target_spacing
            overlap = np.maximum(0, 2 * target_spacing - d_pair)
            force_magnitude = overlap ** force_power
            
            # Direction of force (from j to i for repulsion)
            mask_nonzero = d_pair > 1e-10
            vecs = np.zeros_like(diff[i_idx, j_idx])
            vecs[mask_nonzero] = diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]
            
            forces = np.zeros_like(pts)
            np.add.at(forces, i_idx, vecs * force_magnitude[:, np.newaxis])
            np.add.at(forces, j_idx, -vecs * force_magnitude[:, np.newaxis])
            
            # Boundary forces: push inward if close to boundary
            # Use a smooth potential near the boundary
            boundary_overlap = np.maximum(0, 1.0 - rad - 0.01)  # Small margin
            boundary_force_magnitude = boundary_overlap ** force_power
            rad_safe = np.where(rad > 1e-10, rad, 1.0)
            boundary_forces = -(pts / rad_safe[:, np.newaxis]) * boundary_force_magnitude[:, np.newaxis]
            forces += boundary_forces
            
            # Verlet integration
            # For simplicity, use velocity-Verlet with damping
            # We need to track velocities, but for simplicity use position-based
            # Actually, let's use a simpler approach: just move along force with damping
            
            # Apply force with adaptive step
            force_norm = np.linalg.norm(forces)
            if force_norm > 1e-10:
                forces_normalized = forces / force_norm
                # Adaptive step size based on force magnitude
                step_size = dt_current * np.sqrt(force_norm)
                pts += step_size * forces_normalized
            
            # Damping: scale positions toward center slightly to prevent drift
            # Actually, for packing we don't want to pull to center, just prevent escape
            # Project back inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Compute score
            current_score = compute_score(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            
            # Adapt parameters over time
            progress = min(1.0, (time.time() - start_time) / time_budget)
            dt_current = dt * (1.0 - 0.5 * progress)  # Decrease step size over time
            damping_current = damping + (1.0 - damping) * progress  # Increase damping
            
            # Occasionally add small noise to escape local minima
            if progress < 0.8 and np.random.random() < 0.01:
                pts += np.random.normal(0, 0.001, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
        
        return best_pts, best_score
    
    def optimize_soft_min(pts, alpha_init, k_init, k_end, cooling, time_budget, 
                         stall_threshold=150, noise_scale_base=0.01, polish=False):
        """
        Main optimization loop using soft-min gradient ascent.
        Returns best points found.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        alpha = alpha_init
        k = k_init
        stall_count = 0
        iter_count = 0
        end_time = start_time + time_budget
        
        if polish:
            stall_threshold = 300
            cooling = 0.9995
        
        while time.time() < end_time:
            # Compute distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            d_pair_sq = sq_dists[i_idx, j_idx]
            d_pair = np.sqrt(d_pair_sq)
            
            pair_vals = d_pair / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Compute gradient of soft-min
            grads = soft_min_gradient(pts, k, current_min)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1.0:
                grads /= grad_norm
            
            # Adaptive step size
            if polish:
                step_alpha = alpha * 0.5
            else:
                step_alpha = alpha
            
            # Update
            pts += step_alpha * grads
            
            # Project back inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Cool down
            alpha *= cooling
            
            # Increase k
            progress = min(1.0, (time.time() - start_time) / time_budget)
            if polish:
                k = k_init + (k_end - k_init) * (progress ** 2.0)
            else:
                k = k_init + (k_end - k_init) * (progress ** 1.5)
            
            # Random restart on stall
            if stall_count > stall_threshold:
                noise_scale = noise_scale_base * (1.0 + alpha * 10)
                if polish:
                    noise_scale *= 0.3
                pts += np.random.normal(0, noise_scale, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.005 if not polish else 0.002)
            
            iter_count += 1
        
        return best_pts, best_score
    
    def hex_lattice_init(spacing):
        """Generate hexagonal lattice points inside unit circle."""
        y_step = spacing * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / spacing) + 2
        
        pts = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else spacing / 2
            for i in range(-max_i, max_i + 1):
                x = i * spacing + offset
                if np.hypot(x, y) < 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) > N:
            dists = np.linalg.norm(pts, axis=1)
            idx = np.argsort(dists)[:N]
            pts = pts[idx]
        elif len(pts) < N:
            while len(pts) < N:
                p = np.random.uniform(-0.95, 0.95, 2)
                if np.linalg.norm(p) < 0.95:
                    pts = np.vstack([pts, p])
            pts = pts[:N]
        
        return pts
    
    def random_init():
        """Random initialization with rejection sampling."""
        pts = []
        while len(pts) < N:
            p = np.random.uniform(-0.98, 0.98, 2)
            if np.linalg.norm(p) < 0.98:
                pts.append(p)
        return np.array(pts)
    
    def polish_with_scipy(pts, time_budget):
        """
        Use scipy minimize to polish the packing.
        We minimize the negative soft-min with a high k value.
        """
        end_time = start_time + time_budget
        current_best = compute_score(pts)
        
        # Use a high k for near-exact min
        k_final = 500.0
        
        def objective(x):
            pts_x = x.reshape(N, 2)
            return soft_min_objective(pts_x, k_final, current_best)
        
        def gradient(x):
            pts_x = x.reshape(N, 2)
            return soft_min_gradient(pts_x, k_final, current_best).flatten()
        
        x0 = pts.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 30}
            )
            pts_result = result.x.reshape(N, 2)
            
            # Project back inside
            rad = np.linalg.norm(pts_result, axis=1)
            mask_out = rad > 1.0
            if np.any(mask_out):
                pts_result[mask_out] /= rad[mask_out, np.newaxis]
            
            score_result = compute_score(pts_result)
            return pts_result, score_result
        except Exception:
            return pts, current_best
    
    def polish_with_high_k(pts, time_budget):
        """
        Ultra-high precision polish using very high k values.
        Multiple passes with increasing k.
        """
        current_pts = pts.copy()
        current_best = compute_score(pts)
        end_time = start_time + time_budget
        
        k_values = [500.0, 1000.0, 2000.0, 4000.0]
        
        for k_val in k_values:
            if time.time() > end_time:
                break
            
            def objective_k(x):
                pts_x = x.reshape(N, 2)
                return soft_min_objective(pts_x, k_val, current_best)
            
            def gradient_k(x):
                pts_x = x.reshape(N, 2)
                return soft_min_gradient(pts_x, k_val, current_best).flatten()
            
            x0 = current_pts.flatten()
            
            try:
                result = minimize(
                    objective_k,
                    x0,
                    jac=gradient_k,
                    method='L-BFGS-B',
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 50}
                )
                pts_result = result.x.reshape(N, 2)
                
                # Project back inside
                rad = np.linalg.norm(pts_result, axis=1)
                mask_out = rad > 1.0
                if np.any(mask_out):
                    pts_result[mask_out] /= rad[mask_out, np.newaxis]
                
                score_result = compute_score(pts_result)
                if score_result > current_best:
                    current_best = score_result
                    current_pts = pts_result
            except Exception:
                pass
        
        return current_pts, current_best
    
    def kkt_polish(pts, time_budget):
        """
        Active-set method for polishing the packing.
        Solves the KKT conditions for the maximum radius problem.
        """
        current_pts = pts.copy()
        current_r = compute_score(current_pts)
        end_time = start_time + time_budget
        
        # Tolerance for identifying active constraints
        tol = 1e-10
        
        for _ in range(20): # Max 20 iterations of active set refinement
            if time.time() > end_time:
                break
            
            # 1. Identify active constraints
            diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            
            rad = np.linalg.norm(current_pts, axis=1)
            bound_dists = 1.0 - rad
            
            d_pair_sq = sq_dists[i_idx, j_idx]
            d_pair = np.sqrt(d_pair_sq)
            
            pair_vals = d_pair / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_r = np.min(all_vals)
            
            # Active pairs: distance <= 2*current_r + tol
            active_pairs = np.where(pair_vals <= current_r + tol)[0]
            # Active boundary: distance <= 1 - current_r + tol
            active_bounds = np.where(bound_dists <= current_r + tol)[0]
            
            n_active_pairs = len(active_pairs)
            n_active_bounds = len(active_bounds)
            
            if n_active_pairs == 0:
                break
            
            # 2. Build the KKT linear system
            # Variables: lambda (n_active_pairs), mu (n_active_bounds)
            # Equations: 2*N (force balance for each point in x and y)
            # 
            # Force balance for point i:
            # sum_{j in N(i)} lambda_{ij} * (c_i - c_j) / ||c_i - c_j|| + mu_i * c_i / ||c_i|| = 0
            #
            # We solve for lambda and mu.
            
            n_vars = n_active_pairs + n_active_bounds
            A = np.zeros((2 * N, n_vars))
            
            # Pair contributions
            for k, pair_idx in enumerate(active_pairs):
                i = i_idx[pair_idx]
                j = j_idx[pair_idx]
                
                vec = diff[i, j]
                dist = d_pair[pair_idx]
                
                if dist < 1e-12:
                    continue
                
                unit_vec = vec / dist
                
                # Force on i is +lambda * unit_vec
                # Force on j is -lambda * unit_vec
                A[2 * i, k] += unit_vec[0]
                A[2 * i + 1, k] += unit_vec[1]
                
                A[2 * j, k] -= unit_vec[0]
                A[2 * j + 1, k] -= unit_vec[1]
            
            # Boundary contributions
            for k, bound_idx in enumerate(active_bounds):
                i = bound_idx
                n_var_idx = n_active_pairs + k
                
                if rad[i] < 1e-12:
                    continue
                
                unit_vec = current_pts[i] / rad[i]
                
                # Force on i is +mu * unit_vec (pushing inward? No, constraint is ||c|| + r <= 1, so gradient is c/||c||)
                # Wait, the constraint is g_i(c) = ||c_i|| + r - 1 <= 0.
                # Gradient is c_i / ||c_i||.
                # Lagrangian L = ... + mu_i * (||c_i|| + r - 1).
                # Stationarity: sum lambda_ij * grad_{c_i} g_ij + mu_i * grad_{c_i} g_i = 0.
                # grad_{c_i} g_ij = (c_i - c_j) / ||c_i - c_j||.
                # grad_{c_i} g_i = c_i / ||c_i||.
                # So the force from the boundary is mu_i * (c_i / ||c_i||).
                # This force points OUTWARD.
                # But we want the forces to balance.
                # The repulsive forces from pairs point OUTWARD from the other circle.
                # So if a circle is on the boundary, the boundary force must point INWARD to balance?
                # No, the constraint is one-sided. The circle cannot go OUT.
                # So the "wall" pushes IN.
                # The gradient of the constraint function g(c) = ||c|| + r - 1 is c/||c|| (outward).
                # The Lagrange multiplier mu >= 0.
                # The term in the gradient of L is mu * grad g = mu * (c/||c||).
                # This term points OUTWARD.
                # But the stationarity condition is sum of these terms = 0.
                # This implies that the sum of outward forces must be 0?
                # That doesn't make sense physically.
                # Let's check the sign.
                # Maximize r.
                # Constraints: ||c_i - c_j|| - 2r >= 0 => 2r - ||c_i - c_j|| <= 0.
                # Let h_ij = 2r - ||c_i - c_j|| <= 0.
                # grad_{c_i} h_ij = -(c_i - c_j) / ||c_i - c_j|| = (c_j - c_i) / ||c_i - c_j||. (Points INWARD towards j)
                # Let h_i = 1 - ||c_i|| - r <= 0.
                # grad_{c_i} h_i = -c_i / ||c_i||. (Points INWARD towards center)
                # 
                # L = -r + sum lambda_ij h_ij + sum mu_i h_i.
                # Grad_{c_i} L = sum lambda_ij (c_j - c_i)/d_ij + mu_i (-c_i/||c_i||) = 0.
                # So: sum lambda_ij (c_i - c_j)/d_ij + mu_i (c_i/||c_i||) = 0.
                # This matches my previous derivation.
                # The forces are:
                # From pairs: lambda_ij * (c_i - c_j)/d_ij. (Repulsive, points AWAY from j)
                # From boundary: mu_i * (c_i/||c_i||). (Points OUTWARD)
                # 
                # Wait, if the boundary force points OUTWARD, and the pair forces point OUTWARD (from neighbors), how can they sum to 0?
                # They can't, unless some pair forces point INWARD?
                # No, pair forces are always repulsive.
                # This implies that my sign convention for the boundary force is wrong or the physics is different.
                # 
                # Let's re-evaluate.
                # If a circle is on the boundary, it is being pushed IN by the wall.
                # So the force from the wall should be INWARD.
                # My gradient calculation: grad (1 - ||c|| - r) = -c/||c||. This is INWARD.
                # So the term in the stationarity equation is mu * (-c/||c||).
                # So: sum lambda_ij (c_i - c_j)/d_ij - mu_i (c_i/||c_i||) = 0.
                # => sum lambda_ij (c_i - c_j)/d_ij = mu_i (c_i/||c_i||).
                # The LHS is the sum of repulsive forces from neighbors (OUTWARD).
                # The RHS is the force from the wall (OUTWARD)??
                # No, the wall force is INWARD.
                # So the equation should be:
                # Sum of Repulsive Forces (OUT) + Wall Force (IN) = 0.
                # Sum lambda_ij (c_i - c_j)/d_ij + F_wall = 0.
                # F_wall = - mu_i (c_i/||c_i||).
                # So: sum lambda_ij (c_i - c_j)/d_ij - mu_i (c_i/||c_i||) = 0.
                # This means the sum of repulsive forces must balance the wall force.
                # Since both are outward? No.
                # Repulsive force on i from j is (c_i - c_j)/d_ij. This points from j to i. So OUTWARD from j, but what about i? It points away from j.
                # If i is on the boundary, and j is inside, the force on i is OUTWARD (towards the boundary).
                # The wall pushes INWARD.
                # So they can balance.
                # So the equation is correct:
                # sum lambda_ij (c_i - c_j)/d_ij - mu_i (c_i/||c_i||) = 0.
                
                A[2 * i, n_var_idx] -= unit_vec[0]
                A[2 * i + 1, n_var_idx] -= unit_vec[1]
            
            # Solve A * x = 0. This is a homogeneous system.
            # We want non-trivial solutions? No, we want to find lambda and mu.
            # The system is underdetermined (2N equations, n_vars unknowns).
            # Usually n_vars < 2N.
            # We can use SVD to find the null space.
            # Or, we can add a normalization constraint, e.g., sum(lambda) + sum(mu) = 1.
            # Or, we can solve for the direction of the forces.
            
            # Let's use SVD to find the vector in the null space of A that has non-negative components.
            U, S, Vt = np.linalg.svd(A)
            
            # The null space is spanned by the last rows of Vt if S has zeros.
            # Since the system is underdetermined, there is a null space.
            # We want a vector x in the null space such that x >= 0.
            
            # If the matrix is rank deficient, we can take the right singular vector corresponding to the smallest singular value.
            # But we need x >= 0.
            
            # Alternative: Solve the least squares problem with a penalty for negative values?
            # Or simply: If the system is consistent, the null space vector is unique up to scale (if rank is 2N-1).
            # Let's assume rank is 2N-1. Then the null space is 1D.
            # We can find the vector in the null space.
            
            # If the smallest singular value is close to 0, the last column of V is in the null space.
            if S[-1] < 1e-8:
                x = Vt[-1, :]
                # Scale x to be non-negative if possible
                if np.min(x) < 0:
                    x = -x
                if np.min(x) < 0:
                    # The null space vector has mixed signs.
                    # This means the active set is not valid (some multipliers would be negative).
                    # We should remove the constraint with the most negative multiplier.
                    min_idx = np.argmin(x)
                    if min_idx < n_active_pairs:
                        # Remove this pair
                        active_pairs = np.delete(active_pairs, min_idx)
                        # Remove corresponding columns from A? No, we rebuild A next iteration.
                        # Just update active_pairs and continue.
                        continue
                    else:
                        # Remove this bound
                        active_bounds = np.delete(active_bounds, min_idx - n_active_pairs)
                        continue
                else:
                    # Valid active set. We can now refine r and positions.
                    # The multipliers are x.
                    # We can compute the new r?
                    # The KKT conditions are satisfied.
                    # We can try to increase r slightly and re-solve?
                    # Or just accept the current points as polished.
                    break
            
            # If we get here, it means the system is not rank deficient or we couldn't find a valid null space vector.
            # In that case, we might need to adjust the active set.
            # For simplicity, we break and rely on the previous soft-min polish.
            break
        
        return current_pts, current_r

    # Generate multiple initializations
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.95, 0.97, 0.99, 1.0, 1.01, 1.03]
    
    time_per_init = min(15.0, time_limit * 0.15)
    
    for idx, mult in enumerate(spacing_multipliers):
        if time.time() - start_time > time_limit * 0.55:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        if 0.97 <= mult <= 1.03:
            time_budget = time_per_init
        else:
            time_budget = time_per_init * 0.6
        
        pts_opt, score = optimize_soft_min(pts, 
                                           alpha_init=0.012, 
                                           k_init=5.0, 
                                           k_end=200.0, 
                                           cooling=0.9997,
                                           time_budget=time_budget,
                                           stall_threshold=120,
                                           noise_scale_base=0.006)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Phase 1.5: Collision dynamics from hexagonal lattice
    if time.time() - start_time < time_limit * 0.70:
        for mult in [0.99, 1.0, 1.01]:
            if time.time() - start_time > time_limit * 0.70:
                break
            
            spacing = base_spacing * mult
            pts = hex_lattice_init(spacing)
            
            time_budget = min(10.0, time_limit * 0.10)
            pts_cd, score_cd = collision_dynamics(pts, time_budget, dt=0.002, damping=0.95)
            
            if score_cd > best_score:
                best_score = score_cd
                best_pts = pts_cd.copy()
    
    # Phase 2: Try random initializations
    if time.time() - start_time < time_limit * 0.75:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.75:
                break
            
            pts = random_init()
            time_budget = min(10.0, time_limit * 0.10)
            pts_opt, score = optimize_soft_min(pts,
                                               alpha_init=0.008,
                                               k_init=4.0,
                                               k_end=150.0,
                                               cooling=0.9996,
                                               time_budget=time_budget,
                                               stall_threshold=100,
                                               noise_scale_base=0.008)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Phase 3: Polish the best result
    if best_pts is not None and time.time() - start_time < time_limit * 0.85:
        remaining_time = time_limit * 0.97 - time.time()
        if remaining_time > 5.0:
            # First polish pass with soft-min gradient ascent
            pts_polished, score_polished = optimize_soft_min(best_pts,
                                                             alpha_init=0.005,
                                                             k_init=100.0,
                                                             k_end=300.0,
                                                             cooling=0.9995,
                                                             time_budget=min(remaining_time * 0.20, 5.0),
                                                             stall_threshold=150,
                                                             noise_scale_base=0.003,
                                                             polish=True)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
            
            # Second: Use scipy for high-precision polish
            remaining_time2 = time_limit * 0.97 - time.time()
            if remaining_time2 > 3.0:
                pts_scipy, score_scipy = polish_with_scipy(best_pts, min(remaining_time2 * 0.35, 10.0))
                
                if score_scipy > best_score:
                    best_score = score_scipy
                    best_pts = pts_scipy
            
            # Third: Ultra-high precision polish
            remaining_time3 = time_limit * 0.97 - time.time()
            if remaining_time3 > 3.0:
                pts_ultra, score_ultra = polish_with_high_k(best_pts, min(remaining_time3 * 0.7, 15.0))
                
                if score_ultra > best_score:
                    best_score = score_ultra
                    best_pts = pts_ultra
            
            # Fourth: Collision dynamics polish with small dt
            remaining_time4 = time_limit * 0.97 - time.time()
            if remaining_time4 > 3.0:
                pts_cd_polish, score_cd_polish = collision_dynamics(best_pts, min(remaining_time4 * 0.5, 8.0), dt=0.0005, damping=0.99)
                
                if score_cd_polish > best_score:
                    best_score = score_cd_polish
                    best_pts = pts_cd_polish
    
    # Final verification
    if best_pts is None:
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    # Final high-precision polish if time allows
    final_remaining = time_limit * 0.98 - time.time()
    if final_remaining > 2.0:
        pts_final, score_final = polish_with_high_k(best_pts, final_remaining)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final
        
        # Final KKT Polish
        kkt_remaining = time_limit * 0.99 - time.time()
        if kkt_remaining > 1.0:
            pts_kkt, score_kkt = kkt_polish(best_pts, kkt_remaining)
            if score_kkt > best_score:
                best_score = score_kkt
                best_pts = pts_kkt
    
    # Final safety check
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
