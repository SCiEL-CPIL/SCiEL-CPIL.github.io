"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Objective: Minimize (max_dist / min_dist)^2
    # Equivalent to maximizing min_dist / max_dist
    # We will use a custom annealing loop.
    
    def score(pts):
        # pts: (N, 2)
        # Compute pairwise distances
        # Using broadcasting for speed
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        d_min = d.min()
        d_max = d.max()
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    # Initial setup: Random points in a circle, or a grid perturbed
    # Grid is a good start because it's already somewhat compact
    side = int(np.ceil(N ** (1 / D)))
    grid = []
    for i in range(side):
        for j in range(side):
            grid.append([i, j])
            if len(grid) == N:
                break
        if len(grid) == N:
            break
    
    pts = np.array(grid, dtype=float)
    # Center and normalize
    pts -= pts.mean(axis=0)
    current_score = score(pts)
    best_pts = pts.copy()
    best_score = current_score

    # Annealing Parameters
    temp = 1.0
    cooling = 0.995
    steps_per_iter = 1000
    
    # We want to minimize score.
    # Force calculation:
    # 1. Repulsion: Pairs that are too close (dist < target) push apart.
    #    Pairs that are too far (dist > target) pull together? 
    #    Actually, to minimize max/min, we want to equalize distances.
    #    A simple potential: V = sum (1/d^2) is good for min distance, but blows up max?
    #    Better: Simulated Annealing on the score function directly?
    #    Gradient descent on (max/min)^2 is non-smooth.
    #    
    #    Alternative: Minimize variance of distances + maximize min distance?
    #    Let's stick to Simulated Annealing on the raw score.
    
    # Actually, a physics-based approach using "ideal" distances is often better.
    # Let's try to minimize the spread of distances.
    # Score = (max - min)^2 / min^2 + (max/min)^2 ? No, task is (max/min)^2.
    
    # Let's use a simple Metropolis algorithm on the score.
    # Move: Random Gaussian perturbation.
    # Accept: If new_score < old_score OR random < exp(-(new-old)/temp)
    
    # Since score can be large, we need to scale temp.
    # Initial score for 4x4 grid (16 pts):
    # Min dist = 1. Max dist = sqrt(3^2+3^2) = sqrt(18) ~ 4.24.
    # Ratio ~ 4.24. Squared ~ 18.
    
    # Let's refine with a more specific optimization:
    # Minimize (max_d / min_d)
    # This is equivalent to minimizing max_d subject to min_d >= 1 (scale invariance).
    # Or maximizing min_d subject to max_d <= 1.
    
    # Let's try to maximize min_d / max_d.
    # Let's use scipy.optimize.differential_evolution? It might be slow for 32 dims.
    # 16 points * 2 coords = 32 dimensions.
    # DE is robust but can be slow.
    
    # Let's try a hybrid:
    # 1. Start with grid.
    # 2. Use Nelder-Mead or L-BFGS-B on a smoothed version?
    #    The function (max/min)^2 is non-smooth at max/min transitions.
    #    We can use a soft approximation: log(sum(exp(k*d))) / log(sum(exp(k*d)))?
    
    # Let's stick to a robust Simulated Annealing with larger steps initially.
    
    rng = np.random.default_rng(42)
    
    # Scale points to unit variance to help step size
    # Actually, let's just perturb.
    
    for step in range(50000):
        if time.time() > deadline:
            break
            
        # Generate candidate
        # Step size should adapt to temperature
        step_size = 0.1 * temp
        candidate = pts + rng.normal(0, step_size, pts.shape)
        
        # Re-center candidate to remove drift (translation invariance)
        candidate -= candidate.mean(axis=0)
        
        new_score = score(candidate)
        
        # Acceptance
        if new_score < current_score:
            pts = candidate
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_pts = candidate.copy()
        else:
            # Metropolis criterion
            # delta = new - current
            # If temp is small, exp(-delta/temp) is tiny.
            # We need to scale delta.
            # Score range is ~10 to 100.
            # Let's normalize delta by current_score?
            # Or just use a fixed scale.
            delta = new_score - current_score
            # If delta is positive, accept with prob exp(-delta / temp)
            # temp starts at 1.0. If delta is 1.0, prob is 0.36.
            # If delta is 10.0, prob is 0.000045.
            # This seems reasonable.
            if rng.random() < np.exp(-delta / temp):
                pts = candidate
                current_score = new_score
        
        # Cool down
        temp *= cooling
        
        # If temp is too low, restart? Or just stop.
        if temp < 1e-4:
            break

    return best_pts
# EVOLVE-BLOCK-END
