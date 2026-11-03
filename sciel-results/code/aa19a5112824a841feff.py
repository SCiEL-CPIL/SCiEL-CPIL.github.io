# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time
import random

N = 100
CUBE_HALF = 0.5

def _score(c):
    """
    Compute the maximum radius for a given set of centers.
    r = min(min_pairwise_dist / 2, min_boundary_dist)
    """
    # Pairwise distances
    if N > 1:
        dists = pdist(c, metric='euclidean')
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    # Boundary distances
    # Distance from each point to the nearest face
    # For a point (x, y, z) in [-0.5, 0.5]^3
    # dist to x-face = 0.5 - |x|
    abs_c = np.abs(c)
    boundary_dists = CUBE_HALF - np.max(abs_c, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    # The radius is limited by half the minimum distance between centers
    # and the minimum distance to the boundary.
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _initial_grid():
    """Create a dense grid initial placement."""
    # 5x5x4 grid = 100 points
    # Spacing: 1/5 = 0.2 in x,y; 1/4 = 0.25 in z?
    # Let's just use a simple grid of size ceil(N^(1/3)) = 5.
    # 5x5x5 = 125 points. We take the first 100.
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle to break symmetry
    np.random.shuffle(grid)
    return grid[:N].copy()

def _initial_random():
    """Create a random initial placement."""
    return np.random.uniform(-CUBE_HALF, CUBE_HALF, size=(N, 3))

def _simulated_annealing(c, time_limit=50):
    """
    Simulated Annealing to maximize the minimum radius.
    """
    start_time = time.time()
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    # Temperature schedule
    T = 0.02
    T_min = 1e-6
    alpha = 0.999
    
    # Move probabilities
    p_single = 0.6
    p_double = 0.3
    p_swap = 0.1 # Swap two points? Or just move two? Swapping is less useful for position optimization, but moving two is good.
    
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        # Determine move type
        r = random.random()
        c_new = c.copy()
        
        if r < p_single:
            # Move one point
            idx = random.randint(0, N - 1)
            # Gaussian perturbation
            delta = np.random.randn(3) * step_size
            c_new[idx] = c_new[idx] + delta
            # Clamp
            c_new[idx] = np.clip(c_new[idx], -CUBE_HALF, CUBE_HALF)
        elif r < p_single + p_double:
            # Move two points
            i1, i2 = random.sample(range(N), 2)
            delta1 = np.random.randn(3) * step_size
            delta2 = np.random.randn(3) * step_size
            c_new[i1] = c_new[i1] + delta1
            c_new[i2] = c_new[i2] + delta2
            c_new[i1] = np.clip(c_new[i1], -CUBE_HALF, CUBE_HALF)
            c_new[i2] = np.clip(c_new[i2], -CUBE_HALF, CUBE_HALF)
        else:
            # Large jump / Restart-like move for one point to escape local minima
            idx = random.randint(0, N - 1)
            c_new[idx] = np.random.uniform(-CUBE_HALF, CUBE_HALF, size=3)

        new_score = _score(c_new)
        
        # Acceptance
        if new_score > current_score:
            c = c_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        else:
            # Accept with probability exp((new - current) / T)
            # Note: new_score - current_score is negative
            delta_score = new_score - current_score
            # If delta_score is very negative, exp will be 0
            # Use log to avoid underflow issues? exp(-large) is fine in double.
            prob = np.exp(delta_score / T)
            if random.random() < prob:
                c = c_new
                current_score = new_score
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: reheat or restart if stuck? 
            # For now, just keep going with small steps.
            
        # Adaptive step size: reduce step size as we cool down to refine
        # step_size = 0.05 * (T / 0.02) ** 0.5 # Maybe too aggressive
        # Let's just scale linearly with T for simplicity
        step_size = max(0.001, 0.05 * (T / 0.02))

    return best_c, best_score

def construct():
    """
    Construct the optimal arrangement of 100 equal spheres in a unit cube.
    """
    best_c = None
    best_score = -np.inf
    
    # Time budget: 120s total. Leave 5s for overhead.
    # Run multiple restarts.
    # Start with grid, then randoms.
    
    # 1. Grid start
    c_grid = _initial_grid()
    c_best, s_best = _simulated_annealing(c_grid, time_limit=20)
    if s_best > best_score:
        best_score = s_best
        best_c = c_best
        
    # 2. Random starts
    # We have ~95s left. Let's do 3 random restarts of 20s each? 
    # Or one long run?
    # SA benefits from restarts.
    
    for i in range(3):
        c_rand = _initial_random()
        c_curr, s_curr = _simulated_annealing(c_rand, time_limit=20)
        if s_curr > best_score:
            best_score = s_curr
            best_c = c_curr
            
    # 3. Final refinement on the best solution found
    # Run a short SA with very low temperature to polish
    if best_c is not None:
        # Copy the best and run a final polish
        c_final = best_c.copy()
        # Override temperature to be very low for polishing
        # We can just call SA again with a shorter time and lower initial T
        # But our SA function has fixed T start. 
        # Let's just run SA again on the best solution. It will start high T and cool down.
        # To polish, we might want to start with low T. 
        # Since we can't easily pass T_start, we rely on the fact that 
        # if we start from a good solution, the high T moves will be rejected often, 
        # but some might be accepted. 
        # Better: Implement a simple local improvement loop here.
        
        # Local improvement: try moving each point slightly in 6 directions
        # This is slow in Python, so we'll do a few iterations.
        c_polish = best_c.copy()
        current_score = _score(c_polish)
        
        # Simple hill climbing
        for _ in range(50): # 50 passes
            improved = False
            for i in range(N):
                # Try small moves
                for dx in [-0.005, 0.005]:
                    for dy in [-0.005, 0.005]:
                        for dz in [-0.005, 0.005]:
                            c_test = c_polish.copy()
                            c_test[i] = c_polish[i] + np.array([dx, dy, dz])
                            c_test[i] = np.clip(c_test[i], -CUBE_HALF, CUBE_HALF)
                            s_test = _score(c_test)
                            if s_test > current_score:
                                c_polish = c_test
                                current_score = s_test
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_c = c_polish

    return best_c
# EVOLVE-BLOCK-END
