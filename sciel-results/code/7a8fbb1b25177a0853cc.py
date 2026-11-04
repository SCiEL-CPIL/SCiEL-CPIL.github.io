"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True
    
    def compute_radii_from_centers(centers, margin=0.0):
        """
        Given circle centers, compute maximum non-overlapping radii using
        all pairwise distances and boundary distances.
        """
        centers = np.asarray(centers)
        n = len(centers)
        
        # Compute boundary distances vectorized
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        radii = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        
        # Enforce non-overlap for all pairs (vectorized)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist, axis=1)
        pair_constraint = min_pair_dist / 2.0
        radii = np.minimum(radii, pair_constraint)
        
        if margin > 0:
            radii = np.maximum(0.0, radii - margin)
        
        return radii

    def kinetic_annealing(init_centers, init_radii, seed=0, max_steps=5000):
        """
        Kinetic annealing simulation.
        """
        rng = np.random.RandomState(seed)
        centers = init_centers.copy()
        radii = init_radii.copy()
        velocities = rng.normal(0, 0.01, size=(N, 2))
        
        # Parameters
        dt = 0.005
        damping = 0.98
        repulsion_strength = 50.0
        attraction_strength = 1.0
        center = np.array([0.5, 0.5])
        
        best_score = -np.inf
        best_centers = centers.copy()
        best_radii = radii.copy()
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            
            # Compute forces
            forces = np.zeros((N, 2))
            
            # Repulsion between circles
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist and dist > 1e-6:
                        overlap = min_dist - dist
                        # Force is proportional to overlap and inversely to distance to avoid singularity
                        force_mag = repulsion_strength * overlap / max(dist, 0.01)
                        fx = force_mag * (dx / dist)
                        fy = force_mag * (dy / dist)
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                    elif dist <= 1e-6:
                        # Handle exact overlap with random push
                        push = rng.normal(0, 0.1, size=2)
                        forces[i, :] += push
                        forces[j, :] -= push
            
            # Attraction to center
            for i in range(N):
                dx = centers[i, 0] - center[0]
                dy = centers[i, 1] - center[1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 1e-6:
                    forces[i, 0] -= attraction_strength * dx
                    forces[i, 1] -= attraction_strength * dy
            
            # Update velocities
            velocities += forces * dt
            velocities *= damping
            
            # Update positions
            centers += velocities * dt
            
            # Constrain to square
            for i in range(N):
                if centers[i, 0] < radii[i]:
                    centers[i, 0] = radii[i]
                    velocities[i, 0] = abs(velocities[i, 0]) * 0.5
                elif centers[i, 0] > 1 - radii[i]:
                    centers[i, 0] = 1 - radii[i]
                    velocities[i, 0] = -abs(velocities[i, 0]) * 0.5
                    
                if centers[i, 1] < radii[i]:
                    centers[i, 1] = radii[i]
                    velocities[i, 1] = abs(velocities[i, 1]) * 0.5
                elif centers[i, 1] > 1 - radii[i]:
                    centers[i, 1] = 1 - radii[i]
                    velocities[i, 1] = -abs(velocities[i, 1]) * 0.5
            
            # Check validity and score
            # We use the current radii for the simulation, but we want to maximize sum of radii.
            # So we check if the current configuration allows for larger radii.
            # For simplicity in the loop, we just track the sum of current radii.
            # At the end, we will recompute max radii.
            current_score = np.sum(radii)
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
                best_radii = radii.copy()
                
            # Slowly grow radii if the system is stable (low kinetic energy)
            kinetic_energy = np.sum(velocities**2)
            if kinetic_energy < 0.001 and step % 100 == 0:
                # Grow radii slightly
                growth = 0.001
                new_radii = radii + growth
                # Check if new radii are feasible
                valid_growth = True
                for i in range(N):
                    if centers[i, 0] - new_radii[i] < 0 or centers[i, 0] + new_radii[i] > 1:
                        valid_growth = False
                        break
                    if centers[i, 1] - new_radii[i] < 0 or centers[i, 1] + new_radii[i] > 1:
                        valid_growth = False
                        break
                if valid_growth:
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = centers[i, 0] - centers[j, 0]
                            dy = centers[i, 1] - centers[j, 1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < new_radii[i] + new_radii[j]:
                                valid_growth = False
                                break
                        if not valid_growth:
                            break
                if valid_growth:
                    radii = new_radii
                    # Reset velocities to allow re-equilibration
                    velocities = rng.normal(0, 0.005, size=(N, 2))
        
        return best_centers, best_radii

    def initial_grid_centers():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])
    
    def random_centers(seed=0):
        rng = np.random.RandomState(seed)
        centers = rng.uniform(0.1, 0.9, size=(N, 2))
        return centers

    def gradient_polish(centers, max_iters=200):
        """
        Final high-precision polish using L-BFGS-B.
        """
        try:
            def objective_only(centers_flat):
                centers = centers_flat.reshape(-1, 2)
                radii = compute_radii_from_centers(centers)
                return -np.sum(radii)
            
            result = minimize(
                objective_only,
                centers.flatten(),
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * (2 * N),
                options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            opt_centers = result.x.reshape(-1, 2)
            radii = compute_radii_from_centers(opt_centers)
            x_full = np.column_stack([opt_centers, radii])
            if is_valid(x_full, tol=1e-6):
                return x_full
        except Exception:
            pass
        return None

    # Strategy: Multiple initializations with Kinetic Annealing + Gradient Polish
    best_result = None
    best_score = -np.inf
    
    # Multiple initializations
    inits = []
    
    # Grid initialization
    inits.append(initial_grid_centers())
    
    # Random initializations with different seeds
    for seed in [1, 7, 13, 23, 37, 53, 71, 89, 101, 131]:
        inits.append(random_centers(seed))
    
    # Optimize each initialization
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        # Initial radii: small, so they can move freely
        init_radii = np.full(N, 0.02)
        
        # Step 1: Kinetic Annealing
        ka_centers, ka_radii = kinetic_annealing(init_centers, init_radii, seed=init_idx, max_steps=3000)
        
        # Step 2: Compute exact max radii for the resulting centers
        final_radii = compute_radii_from_centers(ka_centers)
        x_full = np.column_stack([ka_centers, final_radii])
        
        if is_valid(x_full, tol=1e-6):
            score = np.sum(final_radii)
            if score > best_score:
                best_score = score
                best_result = x_full.copy()
        
        # Step 3: Gradient Polish
        polished = gradient_polish(ka_centers, max_iters=100)
        if polished is not None:
            p_score = np.sum(polished[:, 2])
            if p_score > best_score:
                best_score = p_score
                best_result = polished.copy()
    
    # If no valid result found, fall back to grid
    if best_result is None:
        centers = initial_grid_centers()
        radii = compute_radii_from_centers(centers)
        best_result = np.column_stack([centers, radii])
    
    # Final refinement: more local search from best result
    if best_result is not None and time.time() < deadline - 20:
        x_ref = best_result.copy()
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            
            # Perturb centers slightly
            rng = np.random.RandomState(100 + trial)
            perturbed = x_ref[:, :2].copy()
            perturbed += rng.normal(0, 0.015, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.02, 0.98)
            
            # Kinetic Annealing from perturbed
            init_radii = np.full(N, 0.02)
            ka_centers, ka_radii = kinetic_annealing(perturbed, init_radii, seed=200 + trial, max_steps=2000)
            
            final_radii = compute_radii_from_centers(ka_centers)
            x_full = np.column_stack([ka_centers, final_radii])
            
            if is_valid(x_full, tol=1e-6):
                score = np.sum(final_radii)
                if score > best_score:
                    best_score = score
                    best_result = x_full.copy()
            
            # Gradient Polish
            polished = gradient_polish(ka_centers, max_iters=100)
            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
    
    # Ensure final validity
    if best_result is None or not is_valid(best_result, tol=1e-6):
        # Last resort: very conservative grid
        centers = initial_grid_centers()
        radii = compute_radii_from_centers(centers, margin=0.001)
        best_result = np.column_stack([centers, radii])
    
    return best_result

# EVOLVE-BLOCK-END
