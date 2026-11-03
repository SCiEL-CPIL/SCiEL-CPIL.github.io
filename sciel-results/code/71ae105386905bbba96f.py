"""Circle packing in a unit square - 200 equal circles.

Strategy: Force-directed relaxation (Molecular Dynamics) with simulated annealing.
Treats circles as particles repelling each other and the walls. Uses velocity Verlet
integration with adaptive temperature cooling to escape local minima and settle
into a dense packing. This is structurally different from gradient-based smooth
minimization as it uses stochastic dynamics and discrete time steps.
"""
import numpy as np
import time
import math

N = 200

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s kill

    def compute_radius(c):
        """Return the largest feasible radius for centers c (N,2)."""
        wall = 0.5 - np.abs(c)
        min_wall = float(np.min(wall))
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = float(np.min(dist))
        return max(0.0, min(min_wall, min_pair * 0.5))

    def hex_grid(n_rows, n_cols, d_scale=1.0):
        """Generate hexagonal grid points."""
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d / 2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d / 2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        return None

    def square_grid(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    def run_md(pos, temp_start, temp_end, n_steps, dt, time_limit):
        """
        Run Molecular Dynamics simulation.
        pos: (N, 2) array of positions.
        temp_start, temp_end: temperature schedule.
        n_steps: number of steps.
        dt: time step.
        time_limit: wall clock time limit for this phase.
        """
        pos = pos.copy()
        vel = np.zeros((N, 2))
        
        # Precompute indices for pairs
        rows, cols = np.triu_indices(N, k=1)
        n_pairs = len(rows)
        
        # Force constants
        # Repulsion: F = k * (1/r^2) * r_hat for r < 1, 0 otherwise? 
        # Better: Use a soft repulsion that is always active but weak when far.
        # F = k * (1/r^4) for r < 1.5, scaled.
        # Actually, let's use a potential U = 1/r^2 for r < 1, and wall repulsion.
        # To make it efficient, we only care about pairs that are close.
        # But for N=200, full O(N^2) is 40k pairs, which is fast in NumPy.
        
        k_repel = 10.0  # Repulsion strength
        k_wall = 10.0   # Wall repulsion strength
        r_cut = 1.0     # Cutoff for repulsion (we want to push apart if < 1)
        # Actually, we want to maximize the minimum distance.
        # So we should repel when distance is small.
        # Let's use a force that pushes apart when r < 1.
        # F = k * (1 - r) * r_hat ? No, that's linear.
        # F = k / r^2 * r_hat is standard.
        
        # Let's use a potential that is high when overlapping.
        # U = 0.5 * k * (1 - r)^2 for r < 1, 0 otherwise.
        # F = k * (1 - r) * r_hat.
        # This is smooth and zero at r=1.
        
        t_start = time.time()
        
        for step in range(n_steps):
            if time.time() > time_limit:
                break
            
            # Compute forces
            forces = np.zeros((N, 2))
            
            # Pairwise forces
            diff = pos[rows] - pos[cols]  # (M, 2)
            dist_sq = np.sum(diff * diff, axis=1)
            dist = np.sqrt(dist_sq)
            
            # Only apply force if dist < 1.0 (and > 0 to avoid div by 0)
            mask = (dist < 1.0) & (dist > 1e-8)
            if np.any(mask):
                # F = k * (1 - dist) * r_hat
                # r_hat = diff / dist
                # F_vec = k * (1 - dist) * diff / dist = k * (1/dist - 1) * diff
                # Let's stick to F = k * (1 - dist) * (diff / dist)
                factor = k_repel * (1.0 - dist[mask]) / dist[mask]
                f_vec = factor[:, None] * diff[mask]
                
                # Accumulate forces
                np.add.at(forces, rows[mask], f_vec)
                np.add.at(forces, cols[mask], -f_vec)
            
            # Wall forces
            # If x < -0.5 + r, push right. If x > 0.5 - r, push left.
            # Since we don't know r yet, let's use a fixed "target" radius or just keep them inside.
            # Better: Repel from walls if |x| > 0.5 - r_target.
            # But r_target is what we're optimizing.
            # Alternative: Just constrain them to be inside [-0.5, 0.5] with a soft penalty.
            # U_wall = 0.5 * k_wall * (0.5 - |x|)^2 if |x| > 0.5 - r_min, else 0.
            # This is tricky without knowing r.
            
            # Simpler approach: 
            # 1. Repel pairs if dist < 1.
            # 2. Repel from walls if |coord| > 0.5 - 0.03 (estimate).
            # Or better: Just use a hard constraint projection after each step?
            # No, MD needs smooth forces.
            
            # Let's use a wall repulsion that activates when |coord| > 0.45.
            # This keeps them well inside.
            wall_limit = 0.45
            # x forces
            mask_x_pos = pos[:, 0] > wall_limit
            mask_x_neg = pos[:, 0] < -wall_limit
            if np.any(mask_x_pos):
                forces[mask_x_pos, 0] -= k_wall * (pos[mask_x_pos, 0] - wall_limit)
            if np.any(mask_x_neg):
                forces[mask_x_neg, 0] += k_wall * (-wall_limit - pos[mask_x_neg, 0])
            
            # y forces
            mask_y_pos = pos[:, 1] > wall_limit
            mask_y_neg = pos[:, 1] < -wall_limit
            if np.any(mask_y_pos):
                forces[mask_y_pos, 1] -= k_wall * (pos[mask_y_pos, 1] - wall_limit)
            if np.any(mask_y_neg):
                forces[mask_y_neg, 1] += k_wall * (-wall_limit - pos[mask_y_neg, 1])
            
            # Velocity Verlet integration
            # 1. pos += vel * dt + 0.5 * force * dt^2
            pos += vel * dt + 0.5 * forces * dt * dt
            vel_old = vel.copy()
            
            # 2. Compute new forces (reuse logic, but for simplicity in this snippet, 
            #    we might approximate or recompute. Recomputing is safer.)
            # For performance, we'll just use the same forces for the velocity update in this simplified version?
            # No, that's Euler. Let's do proper Verlet.
            # Recompute forces
            forces_new = np.zeros((N, 2))
            diff = pos[rows] - pos[cols]
            dist_sq = np.sum(diff * diff, axis=1)
            dist = np.sqrt(dist_sq)
            mask = (dist < 1.0) & (dist > 1e-8)
            if np.any(mask):
                factor = k_repel * (1.0 - dist[mask]) / dist[mask]
                f_vec = factor[:, None] * diff[mask]
                np.add.at(forces_new, rows[mask], f_vec)
                np.add.at(forces_new, cols[mask], -f_vec)
            
            mask_x_pos = pos[:, 0] > wall_limit
            mask_x_neg = pos[:, 0] < -wall_limit
            if np.any(mask_x_pos):
                forces_new[mask_x_pos, 0] -= k_wall * (pos[mask_x_pos, 0] - wall_limit)
            if np.any(mask_x_neg):
                forces_new[mask_x_neg, 0] += k_wall * (-wall_limit - pos[mask_x_neg, 0])
            
            mask_y_pos = pos[:, 1] > wall_limit
            mask_y_neg = pos[:, 1] < -wall_limit
            if np.any(mask_y_pos):
                forces_new[mask_y_pos, 1] -= k_wall * (pos[mask_y_pos, 1] - wall_limit)
            if np.any(mask_y_neg):
                forces_new[mask_y_neg, 1] += k_wall * (-wall_limit - pos[mask_y_neg, 1])
            
            # 3. vel = vel_old + 0.5 * (force_old + force_new) * dt
            vel = vel_old + 0.5 * (forces + forces_new) * dt
            
            # 4. Damping / Cooling
            # Apply friction to dissipate energy
            vel *= 0.99  # Friction coefficient
            
            # Random kick (Simulated Annealing)
            if temp_end > 0:
                # Temperature decreases linearly
                frac = step / n_steps
                temp = temp_start * (1 - frac) + temp_end * frac
                vel += np.random.normal(0, math.sqrt(temp), (N, 2))
            
            # Hard clip to stay within square (safety)
            pos = np.clip(pos, -0.499, 0.499)
            
        return pos

    # Generate initial configurations
    inits = []
    for n_rows in range(13, 19):
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.90, 0.95, 1.0, 1.05, 1.10]:
            pts = hex_grid(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)
    
    m = int(math.ceil(math.sqrt(N)))
    inits.append(square_grid(m))
    
    # Add random perturbations
    rng = np.random.default_rng(123)
    for base in inits[:4]:
        perturbed = base + rng.normal(0, 0.01, (N, 2))
        perturbed = np.clip(perturbed, -0.499, 0.499)
        inits.append(perturbed)

    # Score and select top initializations
    scored = []
    for pts in inits:
        r = compute_radius(pts)
        scored.append((r, pts))
    scored.sort(key=lambda x: -x[0])
    top_inits = [pts for _, pts in scored[:min(5, len(scored))]]
    
    if not top_inits:
        top_inits = [square_grid(m)]

    best_centers = None
    best_radius = 0.0
    
    # MD Parameters
    # We want to maximize the minimum distance.
    # The repulsion force pushes particles apart.
    # We run multiple MD phases with decreasing temperature.
    
    for init_idx, init in enumerate(top_inits):
        if time.time() > deadline - 10:
            break
            
        r_init = compute_radius(init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init.copy()
        
        current_c = init.copy()
        
        # Phase 1: High temperature to explore
        # Phase 2: Low temperature to settle
        
        # Time allocation: ~20s per init if 5 inits
        phase_time = min(15.0, deadline - time.time() - 5)
        
        # Run MD
        # dt needs to be small enough for stability.
        # Force magnitude ~ k * (1-r). Max force ~ k.
        # Acceleration ~ k/m. Let m=1. a ~ 10.
        # dt < 2/sqrt(k) ~ 0.4. Let's use dt=0.05.
        n_steps = 5000
        dt = 0.05
        
        # Phase 1: Anneal from 0.1 to 0.01
        current_c = run_md(current_c, 0.1, 0.01, n_steps, dt, time.time() + phase_time * 0.5)
        r_eval = compute_radius(current_c)
        if r_eval > best_radius:
            best_radius = r_eval
            best_centers = current_c.copy()
            
        # Phase 2: Anneal from 0.01 to 0.001
        if time.time() < deadline - 5:
            phase_time2 = min(10.0, deadline - time.time() - 2)
            current_c = run_md(current_c, 0.01, 0.001, n_steps, dt, time.time() + phase_time2)
            r_eval = compute_radius(current_c)
            if r_eval > best_radius:
                best_radius = r_eval
                best_centers = current_c.copy()
                
    if best_centers is None:
        best_centers = square_grid(m)

    return best_centers


if __name__ == "__main__":
    result = construct()
    print(f"Shape: {result.shape}")
    print(f"Radius: {compute_radius(result)}")
