"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        # Radial constraint
        radial_dists = np.linalg.norm(pts, axis=1)
        if np.any(radial_dists > 1.0):
            return -1.0
        min_radial = 1.0 - np.max(radial_dists)
        
        # Pairwise constraint
        # Vectorized distance calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Get upper triangle
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        min_pair = np.min(pair_dists) / 2.0
        
        return min(min_pair, min_radial)

    def physics_optimize(pts, max_iter=300, dt=0.002, damping=0.97):
        """
        Physics-based optimization using repulsive forces with adaptive timestep.
        """
        pts = pts.copy()
        best_local = compute_score(pts)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Pairwise repulsive forces
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=-1)
            dists = np.sqrt(dists_sq)
            
            # Avoid division by zero
            mask = dists > 1e-10
            dists_safe = np.where(mask, dists, 1.0)
            
            # Force magnitude: inverse square law, stronger for close pairs
            force_mag = 1.0 / (dists_sq + 1e-8)
            force_mag[~mask] = 0.0
            
            # Force vector: direction from j to i
            force_vec = diff / dists_safe[:, :, np.newaxis]
            force_vec *= force_mag[:, :, np.newaxis]
            
            # Sum forces for each particle
            forces += np.sum(force_vec, axis=1)
            
            # Container wall repulsion - push inward when near boundary
            radial_dists = np.linalg.norm(pts, axis=1)
            wall_force = np.zeros_like(pts)
            # Stronger push when close to boundary
            proximity = np.clip(1.0 - radial_dists, 0, 1)
            wall_strength = 10.0 * proximity**2
            radial_dirs = pts / np.maximum(radial_dists[:, np.newaxis], 1e-10)
            wall_force = -radial_dirs * wall_strength[:, np.newaxis]
            forces += wall_force
            
            # Adaptive timestep based on force magnitude
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force > 100:
                dt_eff = dt * 0.1
            elif max_force > 10:
                dt_eff = dt * 0.5
            else:
                dt_eff = dt
            
            # Update positions with velocity damping
            pts += forces * dt_eff
            pts *= damping
            
            # Project back to unit disk if necessary
            radial_dists = np.linalg.norm(pts, axis=1)
            outside_mask = radial_dists > 1.0
            if np.any(outside_mask):
                pts[outside_mask] /= radial_dists[outside_mask, np.newaxis]
            
            # Track best
            if it % 50 == 0:
                local_score = compute_score(pts)
                if local_score > best_local:
                    best_local = local_score
        
        return pts

    def generate_radial_rings():
        """
        Generate points arranged in concentric rings with optimal angular spacing.
        """
        pts = []
        # Center point
        pts.append([0.0, 0.0])
        
        # Ring configuration: (num_circles, radius)
        # Approximate optimal packing
        rings = [
            (6, 0.1),
            (12, 0.2),
            (18, 0.3),
            (24, 0.4),
            (30, 0.5),
            (36, 0.6),
            (42, 0.7),
            (48, 0.8),
            (54, 0.9),
            (60, 0.98)
        ]
        
        total = 1
        for n_circ, radius in rings:
            if total + n_circ > N:
                n_circ = N - total
            for k in range(n_circ):
                angle = 2 * np.pi * k / n_circ + np.pi / n_circ  # Offset for hexagonal alignment
                pts.append([radius * np.cos(angle), radius * np.sin(angle)])
            total += n_circ
            if total >= N:
                break
        
        pts = np.array(pts[:N])
        
        # Add small random perturbation to break symmetry
        perturbation = np.random.normal(0, 0.005, size=pts.shape)
        pts += perturbation
        
        # Project back to unit disk
        radial_dists = np.linalg.norm(pts, axis=1)
        outside_mask = radial_dists > 1.0
        if np.any(outside_mask):
            pts[outside_mask] /= radial_dists[outside_mask, np.newaxis]
        
        return pts

    def generate_hex_grid_perturbed(spacing=0.055):
        """
        Generate points on a hexagonal grid with random perturbation.
        """
        pts = []
        # Generate grid points
        k = int(1.0 / spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x**2 + y**2 <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        if len(pts) < N:
            # If not enough, try smaller spacing
            return generate_hex_grid_perturbed(spacing * 0.95)
        
        # Select N points closest to center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        pts = pts[idx]
        
        # Add small random perturbation to break symmetry
        perturbation = np.random.normal(0, spacing * 0.1, size=pts.shape)
        pts += perturbation
        
        # Project back to unit disk
        radial_dists = np.linalg.norm(pts, axis=1)
        outside_mask = radial_dists > 1.0
        if np.any(outside_mask):
            pts[outside_mask] /= radial_dists[outside_mask, np.newaxis]
        
        return pts

    def softmin_objective(params, tau=0.01):
        """
        Smooth objective function: -softmin of all constraints.
        """
        pts = params.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu] / 2.0
        
        # Radial distances
        radial_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        all_vals = np.concatenate([pair_dists, radial_dists])
        
        # Softmin
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        exp_terms = np.exp(-shifted / tau)
        softmin_val = min_val - tau * np.log(np.sum(exp_terms))
        
        return -softmin_val

    def softmin_gradient(params, tau=0.01):
        """
        Gradient of the softmin objective.
        """
        pts = params.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, 1.0)
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu] / 2.0
        
        # Radial distances
        radial_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        all_vals = np.concatenate([pair_dists, radial_dists])
        
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        exp_terms = np.exp(-shifted / tau)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        grad_z = -weights
        
        grad_pts = np.zeros_like(pts)
        
        i_idx, j_idx = iu
        w_pair = grad_z[:len(pair_dists)]
        
        vec = pts[i_idx] - pts[j_idx]
        norms = dists[iu]
        unit_vec = vec / norms[:, np.newaxis]
        
        grad_i = 0.5 * unit_vec * w_pair[:, np.newaxis]
        grad_j = -0.5 * unit_vec * w_pair[:, np.newaxis]
        
        np.add.at(grad_pts, i_idx, grad_i)
        np.add.at(grad_pts, j_idx, grad_j)
        
        w_radial = grad_z[len(pair_dists):]
        norms_rad = np.linalg.norm(pts, axis=1)
        norms_rad_safe = np.where(norms_rad > 1e-10, norms_rad, 1.0)
        unit_rad = pts / norms_rad_safe[:, np.newaxis]
        grad_pts += -w_radial[:, np.newaxis] * unit_rad
        
        return grad_pts.flatten()

    def adaptive_optimize(pts, max_iter=150):
        """
        Run L-BFGS-B with adaptive tau schedule from coarse to fine.
        """
        x0 = pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        # Adaptive tau schedule: start coarse, end fine
        tau_schedule = [0.05, 0.02, 0.01, 0.005, 0.002]
        iters_per_phase = max(1, max_iter // len(tau_schedule))
        
        current_pts = pts.copy()
        best_score = compute_score(pts)
        
        for tau in tau_schedule:
            if time.time() > deadline - 15:
                break
            
            # Define objective and gradient with current tau
            def obj(params, tau=tau):
                return softmin_objective(params, tau)
            
            def grad(params, tau=tau):
                return softmin_gradient(params, tau)
            
            try:
                result = minimize(
                    obj, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=grad,
                    bounds=bounds,
                    options={'maxiter': iters_per_phase, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.fun < 0:
                    new_pts = result.x.reshape(N, 2)
                    score = compute_score(new_pts)
                    if score > best_score:
                        best_score = score
                        current_pts = new_pts
                        x0 = new_pts.flatten()
                        
            except Exception:
                pass
        
        return current_pts, best_score

    # Strategy: Multiple initializations + physics + adaptive gradient optimization
    best_pts = None
    best_score = -1.0
    
    # Try different initial configurations
    init_strategies = [
        ("radial_rings", generate_radial_rings),
        ("hex_grid", lambda: generate_hex_grid_perturbed(0.055)),
        ("hex_grid_fine", lambda: generate_hex_grid_perturbed(0.052)),
    ]
    
    for name, gen_func in init_strategies:
        if time.time() > deadline - 25:
            break
        
        # Generate initial configuration
        init_pts = gen_func()
        
        # Apply physics optimization
        phys_pts = physics_optimize(init_pts, max_iter=200)
        
        # Evaluate physics result
        phys_score = compute_score(phys_pts)
        if phys_score > best_score:
            best_score = phys_score
            best_pts = phys_pts
        
        # Apply adaptive gradient-based polish
        if time.time() < deadline - 15:
            opt_pts, opt_score = adaptive_optimize(phys_pts, max_iter=100)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
    
    # Final polish if time remains
    if best_pts is not None and time.time() < deadline - 5:
        opt_pts, opt_score = adaptive_optimize(best_pts, max_iter=50)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts

    # Fallback
    if best_pts is None:
        best_pts = generate_radial_rings()

    return best_pts
# EVOLVE-BLOCK-END
