import numpy as np
import time

N = 26

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for overhead and final processing
    
    # Initial state: Grid-like randomization
    # 5x5 grid is 25 circles. We need 26.
    # Let's just use random points to avoid bias, but initialized with small radii.
    # Or a heuristic: 5x5 grid + 1 in center?
    # Let's try a hexagonal-like packing initialization which is denser.
    # For simplicity and robustness, random init with small radii works well with SA.
    
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.full(N, 0.05)
    
    # Ensure initial state is roughly valid (small radii, random positions might overlap)
    # Quick fix: shrink radii if initial overlaps
    for _ in range(10):
        valid = True
        for i in range(N):
            if x[i] < r[i] or x[i] > 1-r[i] or y[i] < r[i] or y[i] > 1-r[i]:
                r[i] = min(x[i], 1-x[i], y[i], 1-y[i]) * 0.9
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i]-x[j]
                dy = y[i]-y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    scale = dist / (r[i] + r[j]) * 0.9
                    r[i] *= scale
                    r[j] *= scale
                    valid = False
        if valid:
            break

    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    best_score = np.sum(best_r)
    
    # SA Parameters
    T = 1.0
    T_min = 1e-6
    cooling = 0.9999
    step_size = 0.05
    step_cooling = 0.9999
    lambda_penalty = 100.0
    
    def calculate_score(x, y, r):
        # Vectorized penalty calculation
        # Distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Overlaps
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - dist)
        # Exclude diagonal
        np.fill_diagonal(overlap, 0)
        penalty_overlap = np.sum(overlap**2)
        
        # Boundaries
        penalty_boundary = 0.0
        # Left
        pen = np.maximum(0, r - x)
        penalty_boundary += np.sum(pen**2)
        # Right
        pen = np.maximum(0, r - (1 - x))
        penalty_boundary += np.sum(pen**2)
        # Bottom
        pen = np.maximum(0, r - y)
        penalty_boundary += np.sum(pen**2)
        # Top
        pen = np.maximum(0, r - (1 - y))
        penalty_boundary += np.sum(pen**2)
        
        total_penalty = penalty_overlap + penalty_boundary
        score = np.sum(r) - lambda_penalty * total_penalty
        return score

    current_score = calculate_score(x, y, r)
    
    steps = 0
    while time.time() - start_time < time_limit:
        steps += 1
        
        # Select random circle
        i = np.random.randint(N)
        
        # Perturb position
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        new_x[i] += dx
        new_y[i] += dy
        
        # Perturb radius (bias towards growth)
        dr = np.random.normal(0, step_size * 0.5)
        new_r[i] = max(1e-5, new_r[i] + dr)
        
        # Keep within [0,1] roughly to avoid huge boundary penalties slowing convergence
        # Actually, let the penalty handle it, but clamp to [0,1] to prevent extreme values
        new_x[i] = np.clip(new_x[i], 0.0, 1.0)
        new_y[i] = np.clip(new_y[i], 0.0, 1.0)
        
        new_score = calculate_score(new_x, new_y, new_r)
        
        # Metropolis acceptance
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            x = new_x
            y = new_y
            r = new_r
            current_score = new_score
            
            # Update best if valid (or just track best score, validity enforced later)
            # It's safer to track best score and ensure final validity
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
        
        # Cooling
        T *= cooling
        step_size *= step_cooling
        
        # If step_size becomes too small, restart or break
        if step_size < 1e-5:
            break

    # Final Validity Enforcement
    # Use the best solution found
    x = best_x
    y = best_y
    r = best_r
    
    # Iteratively shrink to ensure strict disjoint interiors and boundary containment
    # This is crucial for the evaluator
    for _ in range(200):
        valid = True
        # Check boundaries
        for i in range(N):
            min_bound = min(x[i], 1-x[i], y[i], 1-y[i])
            if r[i] >= min_bound:
                r[i] = min_bound * 0.9999
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i]-x[j]
                dy = y[i]-y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    # Shrink both
                    # We want r[i] + r[j] < dist
                    # Scale factor
                    if dist < 1e-9:
                        # They are at same spot, shrink both significantly
                        r[i] *= 0.5
                        r[j] *= 0.5
                    else:
                        scale = dist / (r[i] + r[j]) * 0.9999
                        r[i] *= scale
                        r[j] *= scale
                    valid = False
                    
        if valid:
            break
            
    # Ensure no negative radii
    r = np.maximum(r, 0.0)
    
    return np.column_stack((x, y, r))

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 26

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for overhead and final processing
    
    # Initial state: Random points with small radii
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.full(N, 0.05)
    
    # Ensure initial state is roughly valid
    for _ in range(10):
        valid = True
        for i in range(N):
            if x[i] < r[i] or x[i] > 1-r[i] or y[i] < r[i] or y[i] > 1-r[i]:
                r[i] = min(x[i], 1-x[i], y[i], 1-y[i]) * 0.9
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i]-x[j]
                dy = y[i]-y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    scale = dist / (r[i] + r[j]) * 0.9 if (r[i]+r[j])>0 else 0.5
                    r[i] *= scale
                    r[j] *= scale
                    valid = False
        if valid:
            break

    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    best_score = np.sum(best_r)
    
    # SA Parameters
    T = 1.0
    cooling = 0.9999
    step_size = 0.05
    step_cooling = 0.9999
    lambda_penalty = 100.0
    
    def calculate_score(x, y, r):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - dist)
        np.fill_diagonal(overlap, 0)
        penalty_overlap = np.sum(overlap**2)
        
        penalty_boundary = 0.0
        pen = np.maximum(0, r - x)
        penalty_boundary += np.sum(pen**2)
        pen = np.maximum(0, r - (1 - x))
        penalty_boundary += np.sum(pen**2)
        pen = np.maximum(0, r - y)
        penalty_boundary += np.sum(pen**2)
        pen = np.maximum(0, r - (1 - y))
        penalty_boundary += np.sum(pen**2)
        
        total_penalty = penalty_overlap + penalty_boundary
        score = np.sum(r) - lambda_penalty * total_penalty
        return score

    current_score = calculate_score(x, y, r)
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(N)
        
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        new_x[i] = np.clip(x[i] + dx, 0.0, 1.0)
        new_y[i] = np.clip(y[i] + dy, 0.0, 1.0)
        
        dr = np.random.normal(0, step_size * 0.5)
        new_r[i] = max(1e-5, r[i] + dr)
        
        new_score = calculate_score(new_x, new_y, new_r)
        
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            x = new_x
            y = new_y
            r = new_r
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
        
        T *= cooling
        step_size *= step_cooling
        
        if step_size < 1e-5:
            break

    # Final Validity Enforcement
    x = best_x
    y = best_y
    r = best_r
    
    for _ in range(200):
        valid = True
        for i in range(N):
            min_bound = min(x[i], 1-x[i], y[i], 1-y[i])
            if r[i] >= min_bound:
                r[i] = min_bound * 0.9999
                valid = False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i]-x[j]
                dy = y[i]-y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    if dist < 1e-9:
                        r[i] *= 0.5
                        r[j] *= 0.5
                    else:
                        scale = dist / (r[i] + r[j]) * 0.9999
                        r[i] *= scale
                        r[j] *= scale
                    valid = False
                    
        if valid:
            break
            
    r = np.maximum(r, 0.0)
    
    return np.column_stack((x, y, r))
# EVOLVE-BLOCK-END
