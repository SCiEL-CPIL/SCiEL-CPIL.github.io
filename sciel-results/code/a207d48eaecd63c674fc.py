"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial solution: all +1s
    best = np.ones(N)
    best_score = np.max(np.convolve(best, best[::-1], mode='full'))
    
    # Logistic map parameters
    # r is the control parameter, x0 is the initial condition
    # We want to explore r near the edge of chaos (around 3.57 - 4.0)
    # and x0 in (0, 1)
    
    # Generate a set of candidate parameters
    # The logistic map is x_{n+1} = r * x_n * (1 - x_n)
    # We generate N points from the map.
    
    # We'll use a grid search over r and x0, then a local search.
    # Given the time limit, we need to be efficient.
    
    # First, let's try a few known good chaotic parameters.
    # r = 4.0 is fully chaotic.
    # r = 3.9, 3.8, etc.
    
    # Let's define a function to compute the score for a given logistic map sequence.
    def _score_from_params(r, x0):
        # Generate N points
        x = np.empty(N)
        x[0] = x0
        for i in range(1, N):
            x[i] = r * x[i-1] * (1 - x[i-1])
        
        # The sign function is sign(x - threshold)
        # The threshold can be 0, or the mean, or something else.
        # Let's start with 0.
        signs = np.sign(x)
        # If any are 0, replace with +1 (or -1)
        signs[signs == 0] = 1.0
        
        # Compute the autoconvolution peak
        # This is O(N^2) if done naively, but N=400 is small.
        # We can use np.convolve.
        conv = np.convolve(signs, signs[::-1], mode='full')
        return np.max(conv)
    
    # We will search over r and x0.
    # Let's start with a coarse grid.
    # r in [3.5, 4.0], x0 in [0.1, 0.9]
    
    r_vals = np.linspace(3.5, 4.0, 100)
    x0_vals = np.linspace(0.1, 0.9, 100)
    
    best_r = 4.0
    best_x0 = 0.5
    
    for r in r_vals:
        if time.time() > deadline:
            break
        for x0 in x0_vals:
            if time.time() > deadline:
                break
            score = _score_from_params(r, x0)
            if score < best_score:
                best_score = score
                best_r = r
                best_x0 = x0
                # Update best solution
                x = np.empty(N)
                x[0] = best_x0
                for i in range(1, N):
                    x[i] = best_r * x[i-1] * (1 - x[i-1])
                best = np.sign(x)
                best[best == 0] = 1.0
    
    # Now, do a local search around the best (r, x0)
    # We can refine r and x0 in a smaller range.
    r_range = 0.01
    x0_range = 0.05
    
    r_fine = np.linspace(best_r - r_range, best_r + r_range, 50)
    x0_fine = np.linspace(best_x0 - x0_range, best_x0 + x0_range, 50)
    
    for r in r_fine:
        if time.time() > deadline:
            break
        for x0 in x0_fine:
            if time.time() > deadline:
                break
            score = _score_from_params(r, x0)
            if score < best_score:
                best_score = score
                best_r = r
                best_x0 = x0
                x = np.empty(N)
                x[0] = best_x0
                for i in range(1, N):
                    x[i] = best_r * x[i-1] * (1 - x[i-1])
                best = np.sign(x)
                best[best == 0] = 1.0
    
    # We can also try to optimize the threshold.
    # But for now, let's stick with 0.
    
    # One more refinement step
    r_range = 0.001
    x0_range = 0.01
    
    r_fine2 = np.linspace(best_r - r_range, best_r + r_range, 20)
    x0_fine2 = np.linspace(best_x0 - x0_range, best_x0 + x0_range, 20)
    
    for r in r_fine2:
        if time.time() > deadline:
            break
        for x0 in x0_fine2:
            if time.time() > deadline:
                break
            score = _score_from_params(r, x0)
            if score < best_score:
                best_score = score
                best_r = r
                best_x0 = x0
                x = np.empty(N)
                x[0] = best_x0
                for i in range(1, N):
                    x[i] = best_r * x[i-1] * (1 - x[i-1])
                best = np.sign(x)
                best[best == 0] = 1.0
    
    return best
# EVOLVE-BLOCK-END
