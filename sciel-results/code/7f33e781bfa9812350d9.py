"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _compute_radii(pos, n):
    """
    Compute the maximum valid radius for each circle given positions.
    Uses iterative relaxation to find consistent radii.
    """
    radii = np.zeros(n)
    
    # Initialize with boundary constraints
    for i in range(n):
        x, y = pos[i]
        radii[i] = min(x, y, 1.0 - x, 1.0 - y)
    
    # Iterative relaxation: update each radius to be consistent with all others
    for _ in range(80):
        changed = False
        for i in range(n):
            x, y = pos[i]
            r_bound = min(x, y, 1.0 - x, 1.0 - y)
            r_max = r_bound
            
            for j in range(n):
                if i == j:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_candidate = dist - radii[j]
                if r_candidate < r_max:
                    r_max = r_candidate
            
            r_new = max(0.0, r_max)
            if abs(r_new - radii[i]) > 1e-10:
                radii[i] = r_new
                changed = True
        
        if not changed:
            break
    
    return radii

def _dense_grid_best_fit(pos, radii, n, deadline, grid_size=15, base_step=0.03):
    """
    Dense grid-based best-fit local search:
    For each circle, try moving it to a dense grid of candidate positions
    around its current location and keep the one that maximizes the sum of radii.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    # Progressive refinement: start coarse, get fine
    step_sizes = [base_step, base_step * 0.5, base_step * 0.25, base_step * 0.125, base_step * 0.0625]
    
    for step_size in step_sizes:
        if time.time() > deadline:
            break
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            for i in range(n):
                orig_x, orig_y = best_pos[i]
                best_local_sum = best_sum
                best_local_pos = best_pos.copy()
                best_local_radii = best_radii.copy()
                
                # Generate dense grid of candidate positions
                half_grid = grid_size // 2
                candidates = []
                for di in range(-half_grid, half_grid + 1):
                    for dj in range(-half_grid, half_grid + 1):
                        new_x = orig_x + di * step_size
                        new_y = orig_y + dj * step_size
                        # Keep in bounds with small margin for radius
                        margin = 0.001
                        if margin <= new_x <= 1 - margin and margin <= new_y <= 1 - margin:
                            candidates.append((new_x, new_y))
                
                # Try each candidate
                for cx, cy in candidates:
                    trial_pos = best_local_pos.copy()
                    trial_pos[i] = [cx, cy]
                    trial_radii = _compute_radii(trial_pos, n)
                    trial_sum = np.sum(trial_radii)
                    
                    if trial_sum > best_local_sum + 1e-8:
                        best_local_sum = trial_sum
                        best_local_pos = trial_pos
                        best_local_radii = trial_radii
                
                if best_local_sum > best_sum + 1e-8:
                    best_pos = best_local_pos
                    best_radii = best_local_radii
                    best_sum = best_local_sum
                    improved = True
    
    return best_pos, best_radii

def _validate(pos, radii, n, tol=1e-9):
    """Check if packing is valid."""
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _fix_and_validate(pos, radii, n):
    """
    Ensure validity by clamping radii.
    This is a safety net.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Clamp to boundary
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
        
    # Fix overlaps by reducing radii
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                md = radii[i] + radii[j]
                if d < md - 1e-9:
                    excess = md - d
                    reduction = excess / 2.0 + 1e-6
                    radii[i] = max(0.0, radii[i] - reduction)
                    radii[j] = max(0.0, radii[j] - reduction)
                    changed = True
        if not changed:
            break
            
    # Final boundary clamp
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
        
    if _validate(pos, radii, n):
        return pos, radii, np.sum(radii)
    else:
        return None, None, -1.0

def _polish(pos, radii, n, deadline):
    """
    High-precision polish pass using very small steps to squeeze out more value.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    # Very fine steps for polishing
    fine_steps = [0.002, 0.001, 0.0005, 0.0002]
    
    for step in fine_steps:
        if time.time() > deadline:
            break
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            for i in range(n):
                orig_x, orig_y = best_pos[i]
                best_local_sum = best_sum
                best_local_pos = best_pos.copy()
                best_local_radii = best_radii.copy()
                
                # 8-directional fine moves
                directions = [
                    (step, 0), (-step, 0), (0, step), (0, -step),
                    (step*0.707, step*0.707), (-step*0.707, step*0.707),
                    (step*0.707, -step*0.707), (-step*0.707, -step*0.707)
                ]
                
                for dx, dy in directions:
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    if 0.001 <= new_x <= 0.999 and 0.001 <= new_y <= 0.999:
                        trial_pos = best_local_pos.copy()
                        trial_pos[i] = [new_x, new_y]
                        trial_radii = _compute_radii(trial_pos, n)
                        trial_sum = np.sum(trial_radii)
                        
                        if trial_sum > best_local_sum + 1e-10:
                            best_local_sum = trial_sum
                            best_local_pos = trial_pos
                            best_local_radii = trial_radii
                
                if best_local_sum > best_sum + 1e-10:
                    best_pos = best_local_pos
                    best_radii = best_local_radii
                    best_sum = best_local_sum
                    improved = True
    
    return best_pos, best_radii

def _initial_grid(n):
    """Create a grid initial configuration."""
    pos = np.zeros((n, 2))
    # 5x6 grid = 30, take first 26
    idx = 0
    for i in range(5):
        for j in range(6):
            if idx >= n:
                break
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 6.0
            pos[idx] = [x, y]
            idx += 1
    return pos

def _initial_hexagonal(n):
    """Create a hexagonal initial configuration."""
    pos = np.zeros((n, 2))
    idx = 0
    row = 0
    while idx < n:
        offset = 0.1 if row % 2 == 1 else 0.0
        for col in range(5):
            if idx >= n:
                break
            x = (col + 0.5) * 0.2 + offset
            y = (row + 0.5) * 0.2
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pos[idx] = [x, y]
            idx += 1
        row += 1
    return pos

def _initial_random(n, seed=0):
    """Create a random initial configuration with minimum separation."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((n, 2))
    valid_pos = []
    attempts = 0
    while len(valid_pos) < n and attempts < 1000:
        p = rng.uniform(0.1, 0.9, size=2)
        ok = True
        for vp in valid_pos:
            if np.linalg.norm(p - vp) < 0.15:
                ok = False
                break
        if ok:
            valid_pos.append(p)
        attempts += 1
    while len(valid_pos) < n:
        valid_pos.append(rng.uniform(0.1, 0.9, size=2))
    pos[:] = valid_pos
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Multiple initial configurations
    configs = []
    
    # Grid
    pos_grid = _initial_grid(N)
    radii_grid = _compute_radii(pos_grid, N)
    configs.append((pos_grid, radii_grid, "grid"))
    
    # Hexagonal
    pos_hex = _initial_hexagonal(N)
    radii_hex = _compute_radii(pos_hex, N)
    configs.append((pos_hex, radii_hex, "hex"))
    
    # Multiple random starts
    for seed in range(4):
        pos_rand = _initial_random(N, seed)
        radii_rand = _compute_radii(pos_rand, N)
        configs.append((pos_rand, radii_rand, f"rand{seed}"))
    
    # Process each configuration
    for pos, radii, name in configs:
        if time.time() > deadline - 20:
            break
        
        try:
            # Run dense grid best-fit optimization
            opt_pos, opt_radii = _dense_grid_best_fit(pos, radii, N, deadline - 15)
            
            # Polish pass for fine-tuning
            pol_pos, pol_radii = _polish(opt_pos, opt_radii, N, deadline - 10)
            
            # Validate and fix
            valid_pos, valid_radii, score = _fix_and_validate(pol_pos, pol_radii, N)
            
            if valid_pos is not None and score > best_score:
                best_score = score
                result = np.zeros((N, 3))
                result[:, :2] = valid_pos
                result[:, 2] = valid_radii
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: small circles in a grid
        result = np.zeros((N, 3))
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N:
                    break
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                r = 0.08
                result[idx, :] = [x, y, r]
                idx += 1
        if idx < N:
            result[idx, :] = [0.5, 0.5, 0.05]
        return result
        
    return best_result

# EVOLVE-BLOCK-END
