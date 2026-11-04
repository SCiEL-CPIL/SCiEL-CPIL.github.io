"""Circle packing in unit square: pack 26 circles, maximize sum of radii.

Strategy: Simulated Annealing with soft penalty for overlaps.
1. Initialize with a reasonable grid or random layout.
2. Define an energy function: E = -sum(r) + Lambda * (sum of overlaps^2).
   Overlaps are defined as max(0, r_i + r_j - dist_ij) and wall violations.
3. Simulated Annealing:
   - Randomly move a circle or change its radius.
   - Accept if energy decreases, or with probability exp(-dE/T).
   - Cool T down over time.
4. Final phase: Strict feasibility repair (shrink radii to fit) and local polish.
"""
import numpy as np
import time
import random

N = 26
EPS = 1e-9

def _energy(xy, r, lambda_overlap):
    """
    Calculate energy: -sum(r) + penalty for overlaps and wall violations.
    Lower is better.
    """
    n = len(xy)
    total_r = np.sum(r)
    
    # Wall violations
    # Circle must be inside [0,1]x[0,1]
    # dist to wall < r => violation
    # We use a quadratic penalty for smoothness in SA
    wall_pen = 0.0
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        # Violation amount: how much it sticks out
        v_left = max(0, ri - x)
        v_right = max(0, ri - (1.0 - x))
        v_down = max(0, ri - y)
        v_up = max(0, ri - (1.0 - y))
        wall_pen += v_left**2 + v_right**2 + v_down**2 + v_up**2

    # Overlap violations
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            overlap = r[i] + r[j] - d
            if overlap > 0:
                overlap_pen += overlap**2

    return -total_r + lambda_overlap * (wall_pen + overlap_pen)

def _is_valid(xy, r):
    """Check if packing is strictly valid."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -EPS: return False
        if x - ri < -EPS or 1.0 - x - ri < -EPS or y - ri < -EPS or 1.0 - y - ri < -EPS:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
            if d < r[i] + r[j] - EPS:
                return False
    return True

def _repair(xy, r):
    """
    Shrink radii to make packing valid.
    Iteratively reduce radii that are causing overlaps or wall violations.
    """
    xy = xy.copy()
    r = r.copy()
    n = len(xy)
    # Simple iterative repair
    for _ in range(100):
        changed = False
        # Wall check
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            if ri > max_r_wall:
                r[i] = max_r_wall
                changed = True
        # Overlap check
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
                if d < r[i] + r[j]:
                    # Reduce both radii proportionally to their current size
                    # or just reduce the larger one? 
                    # Let's reduce both to make them touch
                    # r_i + r_j = d
                    # scale factor
                    total_r = r[i] + r[j]
                    if total_r > 0:
                        scale = d / total_r
                        r[i] *= scale
                        r[j] *= scale
                        changed = True
        if not changed:
            break
    # Final clip
    for i in range(n):
        r[i] = max(r[i], 0.0001)
    return xy, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initialization: Grid + noise
    # 26 circles. 5x5 grid is 25, add 1? Or 6x5?
    # Let's use a 5x5 grid (25 circles) and one extra, or just 26 random-ish
    # Better: 5 rows, 5 cols + 1? 
    # Actually, 26 = 5*5 + 1. 
    # Let's initialize on a 5x5 grid with one extra in center or random.
    # Or just 26 points on a 6x5 grid with one removed?
    # Simple: 5x5 grid covers most. Let's place 25 on grid, 1 random.
    # Or better: 26 points randomly initialized within [0.1, 0.9]
    
    # Grid init is usually better for packing.
    # 26 circles. Sqrt(26) ~ 5.1.
    # Let's use a 5x6 grid (30 spots) and pick 26?
    # Or 5x5 (25) + 1.
    
    # Let's try a dense random init with some structure
    # Or start with a grid of 25 and add 1 small one?
    # No, let's start with 26 random positions and let SA organize.
    # Random init in [0.1, 0.9]
    xy = np.random.uniform(0.1, 0.9, (N, 2))
    r = np.full(N, 0.1) # Start with reasonable radius
    
    # SA Parameters
    T_start = 1.0
    T_end = 1e-5
    alpha = 0.99 # Cooling factor
    lambda_overlap = 10.0
    
    current_energy = _energy(xy, r, lambda_overlap)
    best_xy = xy.copy()
    best_r = r.copy()
    best_energy = current_energy
    
    # Track best valid packing separately? 
    # No, we just want to minimize energy. The energy function penalizes invalid states.
    # If we reach T_end, we should have a near-valid packing.
    
    t = T_start
    step_size_move = 0.05
    step_size_rad = 0.01
    
    while t > T_end and time.time() < deadline:
        # Random move
        i = random.randint(0, N - 1)
        
        # Choose type of move: move position or change radius
        if random.random() < 0.7:
            # Move position
            dx = random.gauss(0, step_size_move)
            dy = random.gauss(0, step_size_move)
            new_x = xy[i, 0] + dx
            new_y = xy[i, 1] + dy
            
            # Keep within bounds roughly to avoid huge penalties
            new_x = np.clip(new_x, 0.0, 1.0)
            new_y = np.clip(new_y, 0.0, 1.0)
            
            old_x, old_y = xy[i, 0], xy[i, 1]
            xy[i, 0] = new_x
            xy[i, 1] = new_y
            
            new_energy = _energy(xy, r, lambda_overlap)
            dE = new_energy - current_energy
            
            if dE < 0 or random.random() < np.exp(-dE / t):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_xy = xy.copy()
                    best_r = r.copy()
            else:
                # Revert
                xy[i, 0] = old_x
                xy[i, 1] = old_y
        else:
            # Change radius
            dr = random.gauss(0, step_size_rad)
            old_r = r[i]
            new_r = max(0.001, r[i] + dr)
            r[i] = new_r
            
            new_energy = _energy(xy, r, lambda_overlap)
            dE = new_energy - current_energy
            
            if dE < 0 or random.random() < np.exp(-dE / t):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_xy = xy.copy()
                    best_r = r.copy()
            else:
                r[i] = old_r
                
        t *= alpha
        
        # Adapt step size?
        # If we are stuck, maybe increase step size?
        # For now, keep simple.
        
        # Reduce step size as T drops
        step_size_move = 0.05 * (t / T_start) + 0.001
        step_size_rad = 0.01 * (t / T_start) + 0.001

    # Use the best energy state found
    xy = best_xy
    r = best_r
    
    # Repair to ensure validity
    xy, r = _repair(xy, r)
    
    # Local polish: Greedy radius increase
    # Try to increase radii where possible
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            # Calculate max possible radius for circle i given others
            max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            for j in range(N):
                if i == j: continue
                d = np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
                max_r = min(max_r, d - r[j])
            
            if max_r > r[i] + 1e-5:
                # Increase slightly to avoid overshooting due to numerical issues
                r[i] = max_r * 0.999
                improved = True
        if not improved:
            break
            
    # Final validation
    if not _is_valid(xy, r):
        # If still invalid, do a hard shrink
        xy, r = _repair(xy, r)
        
    # Construct result array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = xy[i, 0]
        result[i, 1] = xy[i, 1]
        result[i, 2] = r[i]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(res)
