import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 130  # safety margin before 150s kill
    
    def penalty(pos, r, W, H):
        """Compute penalty for violations. Lower is better (0 = no violation)."""
        n = len(r)
        p = 0.0
        # Wall constraints
        for i in range(n):
            p += max(0, r[i] - pos[i, 0]) ** 2
            p += max(0, r[i] - (W - pos[i, 0])) ** 2
            p += max(0, r[i] - pos[i, 1]) ** 2
            p += max(0, r[i] - (H - pos[i, 1])) ** 2
        # Circle-circle overlap
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2)
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    p += overlap ** 2
        return p
    
    def is_valid(pos, r, W, H):
        """Check if configuration is fully valid."""
        n = len(r)
        eps = 1e-9
        for i in range(n):
            if pos[i, 0] - r[i] < -eps or pos[i, 0] + r[i] > W + eps:
                return False
            if pos[i, 1] - r[i] < -eps or pos[i, 1] + r[i] > H + eps:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2)
                if d < r[i] + r[j] - eps:
                    return False
        return True
    
    def fitness(x):
        """Decode x and compute fitness (negative penalty - sum_r). Higher is better."""
        # x layout: [W, x_0..x_N-1, y_0..y_N-1, r_0..r_N-1]
        W = x[0]
        H = 2.0 - W
        if W <= 0.1 or H <= 0.1:
            return -1e10
        pos = x[1:1 + 2 * N].reshape(N, 2)
        r = x[1 + 2 * N:1 + 3 * N]
        
        # Ensure positive radii
        r = np.maximum(r, 1e-6)
        
        p = penalty(pos, r, W, H)
        return -p - np.sum(r)
    
    # Differential Evolution
    pop_size = 40
    F = 0.8  # scale factor
    CR = 0.9  # crossover rate
    
    # Variable bounds:
    # W: [0.3, 1.7]
    # pos: [0, 2] (will be clipped to [0, W] or [0, H] later, but DE works in fixed space)
    # Actually, let's use normalized coords or just broad bounds.
    # Better: optimize W in [0.3, 1.7], pos in [0, 2], r in [0, 1]
    
    lb = np.array([0.3] + [0.0] * (2 * N) + [0.0] * N)
    ub = np.array([1.7] + [2.0] * (2 * N) + [1.0] * N)
    
    # Initial population: mix of grid-based and random
    pop = np.zeros((pop_size, 3 * N + 1))
    
    # First few individuals: structured grids
    for k in range(5):
        W = np.random.uniform(0.6, 1.4)
        H = 2.0 - W
        # Create a grid
        side = 5
        xs = np.linspace(0.1 * W, 0.9 * W, side)
        ys = np.linspace(0.1 * H, 0.9 * H, side)
        grid_xs, grid_ys = np.meshgrid(xs, ys)
        grid_xs = grid_xs.flatten()
        grid_ys = grid_ys.flatten()
        
        # Take first 21
        idx = np.arange(N)
        pos_x = grid_xs[idx]
        pos_y = grid_ys[idx]
        
        # Estimate radius
        min_gap = np.inf
        for i in range(N):
            min_gap = min(min_gap, pos_x[i], W - pos_x[i], pos_y[i], H - pos_y[i])
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((pos_x[i] - pos_x[j]) ** 2 + (pos_y[i] - pos_y[j]) ** 2)
                min_gap = min(min_gap, d / 2)
        
        r_init = np.full(N, min_gap * 0.9)
        
        pop[k, 0] = W
        pop[k, 1:1 + N] = pos_x
        pop[k, 1 + N:1 + 2 * N] = pos_y
        pop[k, 1 + 2 * N:] = r_init
    
    # Rest: random
    for k in range(5, pop_size):
        W = np.random.uniform(0.4, 1.6)
        H = 2.0 - W
        pos_x = np.random.uniform(0, W, N)
        pos_y = np.random.uniform(0, H, N)
        r_init = np.random.uniform(0.01, 0.08, N)
        
        pop[k, 0] = W
        pop[k, 1:1 + N] = pos_x
        pop[k, 1 + N:1 + 2 * N] = pos_y
        pop[k, 1 + 2 * N:] = r_init
    
    # Evaluate initial fitness
    fit = np.array([fitness(ind) for ind in pop])
    best_idx = np.argmin(fit)  # Since fitness is negative penalty - sum_r, min is best
    best_x = pop[best_idx].copy()
    best_fit = fit[best_idx]
    
    # DE Loop
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        for i in range(pop_size):
            # Select 3 distinct individuals
            mask = np.ones(pop_size, dtype=bool)
            mask[i] = False
            idxs = np.random.choice(pop_size - 1, 3, replace=False)
            a, b, c = pop[idxs]
            
            # Mutation
            v = pop[i] + F * (a - b) + 0.5 * (b - c)
            
            # Clip to bounds
            v = np.clip(v, lb, ub)
            
            # Crossover
            u = np.random.rand(3 * N + 1) < CR
            j_rand = np.random.randint(0, 3 * N + 1)
            u[j_rand] = True
            trial = np.where(u, v, pop[i])
            
            # Evaluate
            trial_fit = fitness(trial)
            
            if trial_fit < fit[i]:
                pop[i] = trial
                fit[i] = trial_fit
                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_x = trial.copy()
        
        # Elitism is inherent in the tournament selection above
    
    # Decode best
    W_best = best_x[0]
    H_best = 2.0 - W_best
    pos_best = best_x[1:1 + 2 * N].reshape(N, 2)
    r_best = best_x[1 + 2 * N:1 + 3 * N]
    
    # Post-processing: Clip positions to be strictly inside, adjust radii to ensure validity
    # This is crucial because DE might leave slight overlaps or wall violations.
    
    # Strategy: 
    # 1. Clip positions to [r, W-r] and [r, H-r]
    # 2. If still invalid, shrink radii until valid.
    
    # We want to maximize sum(r). 
    # Let's try to "expand" radii from a slightly smaller valid state.
    
    # First, ensure a valid baseline.
    # Shrink all radii by 10% to be safe, then try to grow them.
    
    r_safe = r_best * 0.8
    pos_safe = pos_best.copy()
    
    # Clip positions
    for i in range(N):
        pos_safe[i, 0] = np.clip(pos_safe[i, 0], r_safe[i], W_best - r_safe[i])
        pos_safe[i, 1] = np.clip(pos_safe[i, 1], r_safe[i], H_best - r_safe[i])
    
    # Check validity
    if not is_valid(pos_safe, r_safe, W_best, H_best):
        # Fallback to grid if DE failed badly
        W_f = 1.0
        H_f = 1.0
        side = 5
        pos_f = np.zeros((N, 2))
        for i in range(N):
            pos_f[i, 0] = ((i % side) + 0.5) / side
            pos_f[i, 1] = ((i // side) + 0.5) / side
        r_f = np.full(N, 0.09)
        return np.column_stack([pos_f, r_f])
    
    # Now, try to increase radii using a simple local search / gradient ascent
    # We have time left.
    current_pos = pos_safe.copy()
    current_r = r_safe.copy()
    current_W = W_best
    current_H = H_best
    
    # Simple iterative improvement:
    # 1. Try to increase all radii by a small epsilon.
    # 2. If invalid, move circles to resolve conflicts.
    
    for _ in range(2000):
        if time.time() > deadline - 5:
            break
        
        # Try to increase radius
        dr = 1e-4
        new_r = current_r + dr
        
        # Check if feasible
        feasible = True
        for i in range(N):
            if current_pos[i, 0] - new_r[i] < 0 or current_pos[i, 0] + new_r[i] > current_W:
                feasible = False
                break
            if current_pos[i, 1] - new_r[i] < 0 or current_pos[i, 1] + new_r[i] > current_H:
                feasible = False
                break
        if feasible:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((current_pos[i, 0] - current_pos[j, 0]) ** 2 + (current_pos[i, 1] - current_pos[j, 1]) ** 2)
                    if d < new_r[i] + new_r[j] - 1e-9:
                        feasible = False
                        break
                if not feasible:
                    break
        
        if feasible:
            current_r = new_r
        else:
            # Try to move circles to make room for larger radii
            # Simple heuristic: move circles towards center if hitting wall, or apart if overlapping
            moved = False
            for i in range(N):
                # Wall check
                if current_pos[i, 0] - current_r[i] < 1e-6:
                    current_pos[i, 0] += 1e-5
                    moved = True
                elif current_pos[i, 0] + current_r[i] > current_W - 1e-6:
                    current_pos[i, 0] -= 1e-5
                    moved = True
                if current_pos[i, 1] - current_r[i] < 1e-6:
                    current_pos[i, 1] += 1e-5
                    moved = True
                elif current_pos[i, 1] + current_r[i] > current_H - 1e-6:
                    current_pos[i, 1] -= 1e-5
                    moved = True
                
                # Overlap check
                for j in range(i + 1, N):
                    dx = current_pos[i, 0] - current_pos[j, 0]
                    dy = current_pos[i, 1] - current_pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = current_r[i] + current_r[j]
                    if d < min_d - 1e-9:
                        # Move apart
                        if d > 1e-9:
                            push = (min_d - d) / 2 + 1e-6
                            current_pos[i, 0] += (dx / d) * push
                            current_pos[i, 1] += (dy / d) * push
                            current_pos[j, 0] -= (dx / d) * push
                            current_pos[j, 1] -= (dy / d) * push
                        else:
                            # Random kick
                            current_pos[i, 0] += 1e-3 * np.random.randn()
                            current_pos[i, 1] += 1e-3 * np.random.randn()
                        moved = True
            
            # Re-clip
            for i in range(N):
                current_pos[i, 0] = np.clip(current_pos[i, 0], current_r[i], current_W - current_r[i])
                current_pos[i, 1] = np.clip(current_pos[i, 1], current_r[i], current_H - current_r[i])
            
            if not moved:
                # Stuck, try smaller dr or break
                dr /= 10
                if dr < 1e-7:
                    break

    # Final validation
    if not is_valid(current_pos, current_r, current_W, current_H):
        # Fallback to safe state
        current_pos = pos_safe
        current_r = r_safe
        current_W = W_best
        current_H = H_best

    result = np.zeros((N, 3))
    result[:, 0] = current_pos[:, 0]
    result[:, 1] = current_pos[:, 1]
    result[:, 2] = current_r

    return result
