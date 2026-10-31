"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    def simulate_packing(W, H, seed, max_steps=5000):
        """
        Discrete Element Method (DEM) simulation.
        1. Initialize circles with small radii and random positions.
        2. Iterate:
           a. Apply forces from wall collisions and circle-circle collisions (Hertzian).
           b. Update velocities and positions (Verlet/Leapfrog integration).
           c. Gradually increase target radii (growth).
           d. Damp velocities to settle.
        """
        np.random.seed(seed)
        
        # Initial state
        # Start with small radii to allow placement, then grow
        # Total area constraint: sum(pi*r^2) <= W*H * packing_density
        # Max possible sum of radii is roughly sqrt(W*H * N * packing_eff / pi)
        # Let's start with a radius that fits N circles in a grid, scaled down.
        # Grid approx: side ~ sqrt(N). r_grid ~ min(W,H) / sqrt(N)
        r_start = min(W, H) / (np.sqrt(N) * 2.0)
        
        # We want to grow to a target. 
        # Heuristic target: 
        # For square 1x1, 21 circles.
        # Optimal packing density ~ 0.84 (hex) or less in rect.
        # Let's aim for a radius that is ~ 1.5x the grid start.
        r_target = min(W, H) / (np.sqrt(N) * 1.2)
        
        xs = np.random.uniform(r_start, W - r_start, N)
        ys = np.random.uniform(r_start, H - r_start, N)
        vs = np.zeros((N, 2))
        rs = np.full(N, r_start)
        
        # Growth schedule
        # We will grow radii linearly from r_start to r_target over the first half of steps,
        # then hold and optimize positions.
        growth_steps = max_steps // 2
        
        # Physics parameters
        k_wall = 1000.0
        k_cc = 1000.0
        damping = 0.9
        dt = 0.01
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # 1. Grow radii
            if step < growth_steps:
                # Linear growth
                growth_factor = 1.0 + (step / growth_steps) * (r_target / r_start - 1.0)
                rs = r_start * growth_factor
            else:
                # Fine tuning growth: very slow increase to push against boundaries
                # This helps "squeeze" circles into corners
                rs = rs * 1.0001
                
            # 2. Calculate Forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Wall Forces
            # If x < r, force pushes right. If x > W-r, force pushes left.
            # Use Hertzian: F ~ delta^1.5 or just linear for stability. Linear is faster.
            for i in range(N):
                if xs[i] < rs[i]:
                    fx[i] += k_wall * (rs[i] - xs[i])
                elif xs[i] > W - rs[i]:
                    fx[i] -= k_wall * (xs[i] - (W - rs[i]))
                
                if ys[i] < rs[i]:
                    fy[i] += k_wall * (rs[i] - ys[i])
                elif ys[i] > H - rs[i]:
                    fy[i] -= k_wall * (ys[i] - (H - rs[i]))
            
            # Circle-Circle Forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = rs[i] + rs[j]
                    
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-10 else 1e-10
                        overlap = min_dist - dist
                        if overlap > 0:
                            # Linear force
                            force_mag = k_cc * overlap
                            nx = dx / dist
                            ny = dy / dist
                            fx[i] += force_mag * nx
                            fy[i] += force_mag * ny
                            fx[j] -= force_mag * nx
                            fy[j] -= force_mag * ny
            
            # 3. Update Velocities and Positions
            vs += dt * np.column_stack((fx, fy))
            vs *= damping  # Damping to settle
            
            xs += dt * vs[:, 0]
            ys += dt * vs[:, 1]
            
            # 4. Hard Constraints (Projection)
            # If a circle goes outside, snap it back to the boundary.
            # This is crucial for stability.
            for i in range(N):
                if xs[i] < rs[i]:
                    xs[i] = rs[i]
                    vs[i, 0] = 0
                elif xs[i] > W - rs[i]:
                    xs[i] = W - rs[i]
                    vs[i, 0] = 0
                
                if ys[i] < rs[i]:
                    ys[i] = rs[i]
                    vs[i, 1] = 0
                elif ys[i] > H - rs[i]:
                    ys[i] = H - rs[i]
                    vs[i, 1] = 0
            
            # Check validity for early exit or tracking
            # We only care about the final state, but we can check if it's valid
            # and keep the best valid sum_r found during the process?
            # No, usually the final settled state is best.
            
        # Final check
        valid = True
        for i in range(N):
            if xs[i] < rs[i] - 1e-5 or xs[i] > W - rs[i] + 1e-5:
                valid = False
            if ys[i] < rs[i] - 1e-5 or ys[i] > H - rs[i] + 1e-5:
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist*min_dist - 1e-5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return None, None, None, 0.0
            
        total_r = np.sum(rs)
        return xs, ys, rs, total_r

    # Main loop
    # Try different aspect ratios
    # W + H = 2
    # W in [0.5, 1.5]
    
    aspect_ratios = np.linspace(0.6, 1.4, 9)
    num_restarts = 3
    
    for W in aspect_ratios:
        H = 2.0 - W
        if W <= 0 or H <= 0:
            continue
            
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            seed = int(W * 1000) + restart * 100 + 42
            xs, ys, rs, total_r = simulate_packing(W, H, seed, max_steps=3000)
            
            if xs is not None:
                if total_r > best_sum_r:
                    best_sum_r = total_r
                    best_config = np.column_stack((xs, ys, rs))
                    best_W = W
                    best_H = H
    
    if best_config is not None:
        return best_config
    
    # Fallback: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
