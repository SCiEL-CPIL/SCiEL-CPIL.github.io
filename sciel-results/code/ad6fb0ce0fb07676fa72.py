"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # Initial guess: Grid layout
    side = int(np.ceil(np.sqrt(N)))
    # We have 26 circles. A 5x5 grid has 25 cells. We need 26.
    # Let's use a 5x6 grid or just place them in a 5x5 grid and one extra?
    # Better: Use a grid that fits. 5x5=25. One circle needs to be placed.
    # Let's try a 5x5 grid of radius 0.19 (slightly less than 0.2) and add one in the center? No, that overlaps.
    # Let's just use a 5x5 grid for the first 25, and place the 26th in a random spot or slightly offset?
    # Actually, let's use a 5x5 grid for 25 circles, and for the 26th, we can just start with a small radius.
    # Or better, use a 5x6 grid? 5*6=30. We take 26.
    # Grid 5x6:
    # x in 0..5, y in 0..5.
    # Spacing: 1/5.0? No, if we have 6 columns, spacing is 1/6.
    # Let's use a 5x6 grid (5 rows, 6 cols).
    # Total 30 positions. We pick 26.
    # This gives a good spread.
    
    rows, cols = 5, 6
    positions = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            r = 0.5 / max(rows, cols) * 0.9  # Slightly smaller to avoid initial overlap
            positions.append([x, y, r])
            
    # We have 30 positions, we need 26. 
    # To make it more interesting, let's just take the first 26.
    # But this might be asymmetric. 
    # Alternative: Random initialization with a fixed seed for reproducibility?
    # The grid is a safe start.
    
    v0 = np.array(positions[:N])
    
    # Define the objective function to minimize
    # We want to maximize sum(r). So we minimize -sum(r) + penalties.
    
    def objective(v):
        if time.time() > deadline:
            return 1e10  # Huge penalty to stop
        
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Penalty for negative radius
        neg_r_pen = np.sum(np.maximum(0, -r)**2)
        
        # Penalty for boundary violations
        # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        bx = np.maximum(0, r - x) + np.maximum(0, r - (1 - x))
        by = np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
        bound_pen = np.sum(bx**2) + np.sum(by**2)
        
        # Penalty for overlaps
        # d_ij >= r_i + r_j
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                # If d < r_i + r_j, penalty is (r_i + r_j - d)^2
                viol = r[i] + r[j] - d
                if viol > 0:
                    overlap_pen += viol**2
        
        # Weights
        # The penalty should be strong enough to prevent violations but not so strong it dominates the objective.
        # As we get closer to the solution, we might want to increase the penalty weight.
        # For simplicity, use a fixed high weight.
        w_overlap = 100.0
        w_bound = 100.0
        w_neg_r = 1000.0
        
        val = -np.sum(r) + w_overlap * overlap_pen + w_bound * bound_pen + w_neg_r * neg_r_pen
        return val

    def gradient(v):
        # Finite difference gradient is slow but robust for 78 vars.
        # L-BFGS-B can use finite differences if jac is not provided.
        # Let's let SciPy handle it with finite differences to save coding time and bugs.
        pass

    # Run optimization
    result = None
    try:
        res = minimize(objective, v0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if time.time() < deadline:
            result = res.x
    except Exception as e:
        print(f"Optimization failed: {e}")
        result = v0

    # If time is running out, we might not have a good result, but we must return something valid.
    # Post-processing: Ensure validity.
    
    def fix_validity(v):
        # v is array of shape (N, 3)
        v = np.array(v, dtype=float)
        
        # 1. Ensure radii are positive and not too large
        for i in range(N):
            v[i, 2] = max(1e-6, min(v[i, 2], 0.5))
        
        # 2. Iteratively resolve overlaps and boundaries
        # We shrink radii if there are violations.
        # This is a conservative approach.
        
        max_iters = 50
        for _ in range(max_iters):
            changed = False
            
            # Check boundaries
            for i in range(N):
                x, y, r = v[i]
                # Max radius allowed by boundaries
                max_r_bound = min(x, 1-x, y, 1-y)
                if r > max_r_bound:
                    v[i, 2] = max_r_bound
                    changed = True
            
            # Check overlaps
            # If overlap, shrink the smaller radius? Or both?
            # Shrink both by half the violation?
            for i in range(N):
                for j in range(i+1, N):
                    dx = v[i, 0] - v[j, 0]
                    dy = v[i, 1] - v[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    sum_r = v[i, 2] + v[j, 2]
                    if d < sum_r:
                        # Overlap exists
                        # Reduce radii. 
                        # Simple heuristic: reduce both by (sum_r - d)/2 + epsilon
                        # But we must also respect boundaries.
                        viol = sum_r - d
                        # Shrink each by viol/2
                        new_ri = v[i, 2] - viol/2
                        new_rj = v[j, 2] - viol/2
                        
                        # Check if new radii are still valid against boundaries
                        max_ri = min(v[i, 0], 1-v[i, 0], v[i, 1], 1-v[i, 1])
                        max_rj = min(v[j, 0], 1-v[j, 0], v[j, 1], 1-v[j, 1])
                        
                        new_ri = max(1e-6, min(new_ri, max_ri))
                        new_rj = max(1e-6, min(new_rj, max_rj))
                        
                        v[i, 2] = new_ri
                        v[j, 2] = new_rj
                        changed = True
            
            if not changed:
                break
        
        return v

    if result is None:
        result = v0
    else:
        # Reshape result to (N, 3)
        v_final = result.reshape(N, 3)
        v_final = fix_validity(v_final)
        result = v_final.flatten()
        
    # Final check: if result is still invalid (shouldn't happen with fix_validity), fallback to grid
    # Re-verify validity one last time
    final_v = result.reshape(N, 3)
    
    # Simple validity check
    valid = True
    for i in range(N):
        x, y, r = final_v[i]
        if x < r - 1e-9 or x > 1 - r + 1e-9 or y < r - 1e-9 or y > 1 - r + 1e-9 or r < 0:
            valid = False
            break
        for j in range(i+1, N):
            dx = x - final_v[j, 0]
            dy = y - final_v[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r + final_v[j, 2] - 1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to the initial grid, which is valid by construction (with the 0.9 factor)
        final_v = np.array(positions[:N])
    
    return final_v

# EVOLVE-BLOCK-END
