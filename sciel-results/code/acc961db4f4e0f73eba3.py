import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute pair indices for vectorized operations
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    def simulate(x, y, r, num_steps=2000, dt=0.005, damping=0.95):
        """
        Runs a simple physics simulation to pack the circles.
        Forces:
        1. Repulsion between circles (Hooke's law for penetration)
        2. Repulsion from boundaries (Hooke's law for penetration)
        3. Attraction to center (weak, to keep them inside)
        4. "Growth" force: slowly increases target radius? 
           Actually, for packing, we usually start with small radii and let them grow,
           or start with large radii and let them shrink/arrange.
           Better strategy: Start with random positions and a reasonable radius.
           The repulsion forces will push them apart. 
           To maximize sum(r), we can slowly increase a "target radius" or 
           simply let the repulsion equilibrium determine the radius?
           No, standard packing via dynamics:
           - If we fix r, we find positions.
           - If we want to maximize sum(r), we can do this:
             1. Start with small r.
             2. Run dynamics to settle positions.
             3. Slightly increase all r.
             4. Repeat.
           This is "Growing Disks" algorithm.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        # We will use a "Growth" strategy embedded in the simulation loop
        # Or better: Run a simulation where radii are fixed, but we do multiple runs
        # with increasing radii? That's slow.
        
        # Alternative: Simulate with fixed radii, but initialize radii to be large
        # and let the system find a configuration. If it can't, they overlap.
        # We need a method that *finds* the max radius.
        
        # Let's use the "Soft Constraint" approach where the objective is implicit.
        # But the prompt asks for a structural change.
        # The "Growing Disks" is a classic structural approach distinct from black-box opt.
        
        # Strategy:
        # 1. Initialize positions randomly.
        # 2. Start with a small radius r_min.
        # 3. Iteratively:
        #    a. Run short dynamics to settle positions for current r.
        #    b. Increase r slightly.
        #    c. Check if stable. If not, revert or reduce growth rate.
        
        # However, doing this for 26 circles in 120s is feasible if dynamics are fast.
        # Let's implement a fast Verlet-like integrator.
        
        # Actually, a more robust "Structural" change is **Lloyd's Algorithm (Centroidal Voronoi Tessellation)** 
        # adapted for weighted circles, or simply **Iterative Projection**.
        
        # Let's stick to **Physics-based Growing Disks** as it is distinct from L-BFGS.
        
        # State: x, y are positions. r is current radius (scalar for all? or vector?).
        # For max sum(r), usually circles have different sizes.
        # But "Growing Disks" often assumes equal radii or grows them uniformly.
        # If we grow uniformly, we get a specific packing.
        # The optimal packing of 26 circles likely has varied radii (corners smaller, center larger).
        
        # To handle varied radii, we can use a "temperature" annealing on radii?
        # Or, we can just run the dynamics with **fixed radii** but **randomized radii** in a multi-start fashion?
        # No, that's just multi-start.
        
        # Let's try a different structural approach: **Local Search with Pairwise Swaps and Radii Adjustment**.
        # But the current program already does multi-start local search (implicitly via L-BFGS).
        
        # Let's go with **Soft-Body Dynamics with Adaptive Radius Growth**.
        # We will simulate N particles.
        # Each particle has a target radius r_target.
        # We start with small r_target and increase it.
        # The forces push particles apart.
        # If the system becomes unstable (energy spikes), we slow down growth.
        
        # Implementation details:
        # x, y: (N,) arrays
        # r: (N,) array of current radii.
        
        # We will try to grow all radii uniformly by a factor, or add a constant?
        # Adding a constant is better for corner circles.
        
        # Let's define the simulation step:
        # 1. Calculate forces on each particle.
        # 2. Update velocities.
        # 3. Update positions.
        # 4. Check for "stability" (max velocity < threshold).
        
        # To maximize sum(r), we will:
        # - Start with r_i = 0.01 for all i.
        # - Run dynamics to settle.
        # - Increase all r_i by delta_r.
        # - Repeat until time limit or stability failure.
        
        # This assumes the optimal packing is "monotonic" in radius growth.
        
        pass # Placeholder, will implement below
        
    # --- ACTUAL IMPLEMENTATION OF GROWING DISKS ---
    
    def run_growing_disks(initial_x, initial_y, initial_r, time_limit):
        x = initial_x.copy()
        y = initial_y.copy()
        r = initial_r.copy()
        vx = np.zeros(N)
        vy = np.zeros(N)
        
        # Parameters
        k_repel = 500.0      # Repulsion stiffness
        k_bound = 500.0      # Boundary stiffness
        k_center = 1.0       # Weak centering
        dt = 0.001           # Time step
        damping = 0.99       # Velocity damping
        growth_step = 0.0005 # How much to grow radius each "phase"
        phases_per_growth = 50 # How many dynamics steps between growths
        
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_score = np.sum(r)
        
        t_start = time.time()
        
        # Main loop: Grow and Settle
        while time.time() < time_limit:
            # Phase 1: Settle positions for current radii
            for step in range(phases_per_growth):
                # Calculate forces
                fx = np.zeros(N)
                fy = np.zeros(N)
                
                # 1. Pairwise Repulsion
                dx = x[i_idx] - x[j_idx]
                dy = y[i_idx] - y[j_idx]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-9)
                
                # Penetration depth
                r_sum = r[i_idx] + r[j_idx]
                penetration = r_sum - dist
                mask = penetration > 0
                
                if np.any(mask):
                    # Force magnitude proportional to penetration (Hooke)
                    force_mag = k_repel * penetration[mask]
                    # Direction: from j to i for i, from i to j for j
                    # Normalize vector
                    inv_dist = 1.0 / dist[mask]
                    fx[i_idx[mask]] += force_mag * dx[mask] * inv_dist
                    fy[i_idx[mask]] += force_mag * dy[mask] * inv_dist
                    fx[j_idx[mask]] -= force_mag * dx[mask] * inv_dist
                    fy[j_idx[mask]] -= force_mag * dy[mask] * inv_dist
                
                # 2. Boundary Repulsion
                # Left: x < r
                pen_left = r - x
                mask = pen_left > 0
                if np.any(mask):
                    fx[mask] += k_bound * pen_left[mask]
                
                # Right: x > 1-r
                pen_right = x - (1.0 - r)
                mask = pen_right > 0
                if np.any(mask):
                    fx[mask] -= k_bound * pen_right[mask]
                
                # Bottom: y < r
                pen_bottom = r - y
                mask = pen_bottom > 0
                if np.any(mask):
                    fy[mask] += k_bound * pen_bottom[mask]
                
                # Top: y > 1-r
                pen_top = y - (1.0 - r)
                mask = pen_top > 0
                if np.any(mask):
                    fy[mask] -= k_bound * pen_top[mask]
                
                # 3. Centering (very weak)
                fx += -k_center * (x - 0.5)
                fy += -k_center * (y - 0.5)
                
                # Update velocities
                vx = (vx + fx * dt) * damping
                vy = (vy + fy * dt) * damping
                
                # Update positions
                x += vx * dt
                y += vy * dt
                
                # Hard clamp to boundaries to prevent escape (safety)
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
            
            # Check stability: max velocity should be low
            max_v = np.max(np.sqrt(vx*vx + vy*vy))
            
            # Phase 2: Grow radii
            # We grow all radii uniformly.
            # If the packing is tight, some circles might be blocked.
            # Simple strategy: Grow all by same amount.
            r += growth_step
            
            # Check if we are still "valid" enough to continue?
            # The forces will push them apart. If they can't, they stay overlapped.
            # We should check if the overlap is decreasing or if we've hit a wall.
            
            # Heuristic: If max velocity is very high, we are in a chaotic state.
            # If max velocity is low, we are settled.
            
            # Update best if valid?
            # We need to project to validity to get a score.
            # But during simulation, we might have overlaps.
            # We only update best_score if we can project to a valid state.
            
            # Projection: Shrink radii to remove overlaps?
            # This is expensive to do every step.
            # Instead, we track the "theoretical" sum of radii.
            # But the score is sum of *valid* radii.
            
            # Let's do a quick validity check and projection every few growth steps.
            if step % 5 == 0:
                # Project to valid state
                x_proj = x.copy()
                y_proj = y.copy()
                r_proj = r.copy()
                
                # Boundary projection
                max_r_bound = np.minimum(np.minimum(x_proj, 1-x_proj), np.minimum(y_proj, 1-y_proj))
                r_proj = np.minimum(r_proj, max_r_bound)
                
                # Pairwise projection (iterative shrink)
                # This is a heuristic projection, not exact max radius.
                # It's fast: just shrink all radii by 1% if any overlap exists.
                # Better: Shrink only the overlapping ones? Hard to identify.
                # Global shrink is safe.
                for _ in range(5):
                    dx_p = x_proj[i_idx] - x_proj[j_idx]
                    dy_p = y_proj[i_idx] - y_proj[j_idx]
                    dist_p = np.sqrt(dx_p*dx_p + dy_p*dy_p)
                    r_sum_p = r_proj[i_idx] + r_proj[j_idx]
                    violations = r_sum_p - dist_p
                    if np.max(violations) < 1e-6:
                        break
                    r_proj *= 0.98 # Shrink all slightly
                    r_proj = np.minimum(r_proj, max_r_bound)
                
                score = np.sum(r_proj)
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_r = x_proj, y_proj, r_proj
                    
            # If we've grown too much and velocities are high (chaos), maybe revert growth?
            # Or just continue, the damping will settle it.
            
            # Stop condition: If we can't grow anymore (velocities stay high or radii stop increasing effectively)
            # For now, just run until time limit.
            
        return best_x, best_y, best_r, best_score

    # Multi-Start
    best_global_score = 0.0
    best_global_state = None
    
    num_starts = 5
    time_per_start = (deadline - start_time) / num_starts
    
    for k in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize
        if k == 0:
            # Grid
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.full(N, 0.05)
            count = 0
            for i in range(side):
                for j in range(side):
                    if count < N:
                        x[count] = (i + 0.5) / side
                        y[count] = (j + 0.5) / side
                        count += 1
        else:
            # Random
            x = np.random.rand(N) * 0.8 + 0.1
            y = np.random.rand(N) * 0.8 + 0.1
            r = np.full(N, 0.05)
            
        x += np.random.normal(0, 0.01, N)
        y += np.random.normal(0, 0.01, N)
        x = np.clip(x, 0.1, 0.9)
        y = np.clip(y, 0.1, 0.9)
        
        x_res, y_res, r_res, score = run_growing_disks(x, y, r, time.time() + time_per_start)
        
        if score > best_global_score:
            best_global_score = score
            best_global_state = np.column_stack([x_res, y_res, r_res])
            
    if best_global_state is None:
        # Fallback
        x = np.full(N, 0.5)
        y = np.full(N, 0.5)
        r = np.full(N, 0.01)
        best_global_state = np.column_stack([x, y, r])
        
    return best_global_state

# EVOLVE-BLOCK-END
