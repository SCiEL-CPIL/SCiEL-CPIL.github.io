"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    # We want to maximize sum(r_i) subject to perimeter 2(W+H)=4 => W+H=2.
    # Strategy:
    # 1. Iterate over aspect ratios.
    # 2. For each aspect ratio, initialize a grid.
    # 3. Use a "Grow and Relax" loop:
    #    a. Start with small radii.
    #    b. Relax positions using force-directed method (springs for overlaps).
    #    c. Increase radii.
    #    d. Repeat until max radius or time limit.
    
    # To make the force-directed method efficient, we use a simple O(N^2) loop 
    # but with vectorized operations where possible, or just tight loops since N=21 is small.
    # N=21 -> 210 pairs. Very fast.
    
    # We will try a few different initializations and aspect ratios.
    
    aspect_ratios = [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4]
    
    for d in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        W = 1.0 + d
        H = 1.0 - d
        if W <= 0.1 or H <= 0.1:
            continue
            
        # Initialize grid
        # Estimate columns based on aspect ratio
        cols = int(np.sqrt(N * W / H))
        if cols < 1: cols = 1
        rows = int(np.ceil(N / cols))
        
        xs = np.zeros(N)
        ys = np.zeros(N)
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            xs[i] = (c + 0.5) * W / cols
            ys[i] = (r_idx + 0.5) * H / rows
            
        # Initial radii
        # Area based estimate
        r_max_est = np.sqrt(0.9 * W * H / (N * np.pi))
        current_r = 0.5 * r_max_est
        
        rs = np.full(N, current_r)
        
        # Relaxation function
        # Uses a spring-like force model.
        # Force = k * (overlap) * direction
        # Also add wall repulsion.
        
        def relax(xs, ys, rs, W, H, steps=2000, lr=0.05):
            """
            Relax positions to minimize overlaps.
            Returns xs, ys and a flag indicating if valid.
            """
            # Precompute some constants
            # We will update xs, ys in place or return new ones
            # Let's use a simple gradient descent on the violation energy.
            
            # To avoid getting stuck in local minima, we can add a small random perturbation
            # or use simulated annealing-like cooling.
            # For now, simple gradient descent with adaptive step.
            
            temp = 1.0
            for step in range(steps):
                if time.time() > deadline - 5:
                    break
                    
                fx = np.zeros(N)
                fy = np.zeros(N)
                
                # Pairwise forces
                for i in range(N):
                    for j in range(i+1, N):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist_sq = dx*dx + dy*dy
                        min_dist = rs[i] + rs[j]
                        
                        if dist_sq < min_dist*min_dist:
                            dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                            if dist < 1e-6:
                                # Random direction
                                angle = np.random.uniform(0, 2*np.pi)
                                nx = np.cos(angle)
                                ny = np.sin(angle)
                            else:
                                nx = dx / dist
                                ny = dy / dist
                            
                            # Overlap amount
                            overlap = min_dist - dist
                            # Force is proportional to overlap
                            # We use a linear spring model
                            f = 1.0 * overlap
                            
                            fx[i] += nx * f
                            fy[i] += ny * f
                            fx[j] -= nx * f
                            fy[j] -= ny * f
                            
                # Wall forces
                # Stronger repulsion from walls to keep them inside
                wall_k = 5.0
                for i in range(N):
                    # Left
                    if xs[i] < rs[i]:
                        fx[i] += wall_k * (rs[i] - xs[i])
                    # Right
                    if xs[i] > W - rs[i]:
                        fx[i] -= wall_k * (xs[i] - (W - rs[i]))
                    # Bottom
                    if ys[i] < rs[i]:
                        fy[i] += wall_k * (rs[i] - ys[i])
                    # Top
                    if ys[i] > H - rs[i]:
                        fy[i] -= wall_k * (ys[i] - (H - rs[i]))
                        
                # Update
                # Adaptive learning rate?
                # Simple: fixed lr, but maybe decay over time?
                # Let's try fixed lr for simplicity, but ensure stability.
                # If forces are large, we might overshoot.
                # Normalize forces?
                # Let's just use a small lr.
                
                xs += lr * fx
                ys += lr * fy
                
                # Hard projection to keep inside bounds
                # This helps stability
                xs = np.clip(xs, rs, W - rs)
                ys = np.clip(ys, rs, H - rs)
                
            # Check validity
            valid = True
            # Check walls
            for i in range(N):
                if xs[i] < rs[i] - 1e-5 or xs[i] > W - rs[i] + 1e-5:
                    valid = False
                if ys[i] < rs[i] - 1e-5 or ys[i] > H - rs[i] + 1e-5:
                    valid = False
            
            # Check pairs
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = rs[i] + rs[j]
                    if dist_sq < min_dist*min_dist - 1e-5:
                        valid = False
                        
            return xs, ys, valid

        # Grow and Relax Loop
        # Start with small r, increase gradually.
        
        r_min = 0.1 * r_max_est
        r_max = 1.1 * r_max_est # Slightly above estimate to allow for better packing
        
        # Binary search or linear increase?
        # Linear increase with backtracking is robust.
        
        current_r = r_min
        step_size = (r_max - r_min) / 50.0
        
        for step in range(100):
            if time.time() > deadline - 5:
                break
                
            # Try to increase radius
            new_r = current_r + step_size
            if new_r > r_max:
                new_r = r_max
                
            rs = np.full(N, new_r)
            
            # Relax
            xs, ys, valid = relax(xs, ys, rs, W, H, steps=1000, lr=0.02)
            
            if valid:
                current_r = new_r
                # If we succeeded, we can keep the step size or reduce it slightly
                # to refine
                # step_size = step_size * 0.9 
            else:
                # Failed, reduce step size
                step_size = step_size * 0.5
                if step_size < 1e-4:
                    break
                # Try again with smaller step
                new_r = current_r + step_size
                rs = np.full(N, new_r)
                xs, ys, valid = relax(xs, ys, rs, W, H, steps=1000, lr=0.02)
                if valid:
                    current_r = new_r
                else:
                    break
                    
        total_r = current_r * N
        if total_r > best_sum_r:
            best_sum_r = total_r
            best_config = np.column_stack((xs, ys, rs))
            best_W = W
            best_H = H
            
    # Fallback if no valid config found (should not happen with small r)
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Verify best_config one last time
    xs = best_config[:, 0]
    ys = best_config[:, 1]
    rs = best_config[:, 2]
    
    valid = True
    for i in range(N):
        if xs[i] < rs[i] - 1e-5 or xs[i] > best_W - rs[i] + 1e-5:
            valid = False
        if ys[i] < rs[i] - 1e-5 or ys[i] > best_H - rs[i] + 1e-5:
            valid = False
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            min_dist = rs[i] + rs[j]
            if dist_sq < min_dist*min_dist - 1e-5:
                valid = False
                
    if valid:
        return best_config
    else:
        # If invalid due to numerical errors, try to fix by shrinking radii slightly
        # or return a safe grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
