import numpy as np
import random
import time

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7

    # Analytic seed: 4 corners (1/3), 4 edges (1/3), 2 center (1/6)
    # Sum = 4*(1/3) + 4*(1/3) + 2*(1/6) = 3.0
    a = 1.0 / 3.0
    b = 1.0 / 3.0
    c = 1.0 / 6.0

    centers = np.zeros((n, 2))
    sides = np.zeros(n)

    # Corners (0-3)
    centers[0] = [a/2, a/2]
    centers[1] = [1-a/2, a/2]
    centers[2] = [a/2, 1-a/2]
    centers[3] = [1-a/2, 1-a/2]
    sides[0:4] = a

    # Edges (4-7)
    centers[4] = [0.5, b/2]      # Bottom
    centers[5] = [0.5, 1-b/2]    # Top
    centers[6] = [b/2, 0.5]      # Left
    centers[7] = [1-b/2, 0.5]    # Right
    sides[4:8] = b

    # Center (8-9)
    # Vertical space between edges: 1 - 2*b = 1/3.
    # Two squares of side c=1/6 fit exactly.
    # Square 8: y in [b, b+c] = [1/3, 1/2]. Center y = 5/12.
    # Square 9: y in [1-b-c, 1-b] = [1/2, 2/3]. Center y = 7/12.
    centers[8] = [0.5, 5.0/12.0]
    centers[9] = [0.5, 7.0/12.0]
    sides[8:10] = c

    # Validation function
    def is_valid(c, s):
        for i in range(n):
            if not (np.isfinite(c[i,0]) and np.isfinite(c[i,1]) and np.isfinite(s[i])):
                return False
            if s[i] <= 0:
                return False
            if c[i,0] - s[i]/2 < -tol or c[i,0] + s[i]/2 > 1 + tol:
                return False
            if c[i,1] - s[i]/2 < -tol or c[i,1] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2 - tol
                if abs(c[i,0] - c[j,0]) < t and abs(c[i,1] - c[j,1]) < t:
                    return False
        return True

    # Objective function for optimization
    # We maximize sum of sides, so minimize -sum(sides) + penalty
    def objective(c, s):
        pen = 0.0
        # Boundary penalties
        for i in range(n):
            # Ensure squares are inside [0, 1]
            # If part of square is outside, penalize the amount outside squared
            if c[i,0] - s[i]/2 < 0:
                pen += (c[i,0] - s[i]/2)**2
            if c[i,0] + s[i]/2 > 1:
                pen += (c[i,0] + s[i]/2 - 1)**2
            if c[i,1] - s[i]/2 < 0:
                pen += (c[i,1] - s[i]/2)**2
            if c[i,1] + s[i]/2 > 1:
                pen += (c[i,1] + s[i]/2 - 1)**2
            
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(c[i,0] - c[j,0])
                dy = abs(c[i,1] - c[j,1])
                t = (s[i] + s[j])/2
                if dx < t and dy < t:
                    # Overlap in x and y
                    ox = t - dx
                    oy = t - dy
                    pen += ox * oy

        return -np.sum(s) + 1000.0 * pen

    # Simulated Annealing
    best_c = centers.copy()
    best_s = sides.copy()
    best_score = objective(best_c, best_s)
    
    # Current state
    curr_c = centers.copy()
    curr_s = sides.copy()
    curr_score = objective(curr_c, curr_s)

    T = 0.1
    T_min = 1e-6
    alpha = 0.999
    max_iter = 50000
    start_time = time.time()
    
    rng = np.random.default_rng(42)

    for i in range(max_iter):
        if time.time() - start_time > 500: # Keep well within 600s limit
            break
            
        # Generate neighbor
        new_c = curr_c.copy()
        new_s = curr_s.copy()
        
        # Randomly pick a square to perturb
        idx = rng.integers(0, n)
        
        # Perturb center
        new_c[idx] += rng.normal(0, 0.05, 2)
        
        # Perturb side (small chance to change size)
        if rng.random() < 0.1:
            new_s[idx] += rng.normal(0, 0.01)
        
        # Clamp side to reasonable range
        new_s[idx] = np.clip(new_s[idx], 1e-4, 0.5)
        
        # Clamp centers to keep squares mostly inside
        # cx must be in [s/2, 1-s/2] ideally, but allow some violation for penalty
        new_c[idx, 0] = np.clip(new_c[idx, 0], -0.1, 1.1)
        new_c[idx, 1] = np.clip(new_c[idx, 1], -0.1, 1.1)

        new_score = objective(new_c, new_s)
        
        # Acceptance criterion
        delta = new_score - curr_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            curr_c = new_c
            curr_s = new_s
            curr_score = new_score
            
            if curr_score < best_score:
                best_c = curr_c.copy()
                best_s = curr_s.copy()
                best_score = curr_score
        
        T *= alpha
        if T < T_min:
            break

    # Final validation and repair
    c_opt = best_c
    s_opt = best_s
    
    # If the SA result is invalid, try to repair it or fallback
    if not is_valid(c_opt, s_opt):
        # Simple repair: shrink sides until valid? 
        # Or just fallback to the known valid seed
        # Let's try to shrink the sides of the SA solution slightly
        scale = 0.99
        for _ in range(100):
            s_test = s_opt * scale
            if is_valid(c_opt, s_test):
                s_opt = s_test
                break
            scale *= 0.99
        
        if not is_valid(c_opt, s_opt):
            # Fallback to analytic seed
            c_opt = centers
            s_opt = sides

    return c_opt, s_opt
