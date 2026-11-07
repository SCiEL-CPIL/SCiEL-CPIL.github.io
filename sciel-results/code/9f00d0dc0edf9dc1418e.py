"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time
import math
import sys

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import linprog

def construct():
    """
    Strategy:
    1. The objective value is max_k C_k(h) / N * 2, where C_k(h) is the k-th correlation coefficient.
    2. For fixed lag k, C_k(h) = sum_i h[i] * (1 - h[i+k]) is a quadratic function (concave).
       However, we can bound the maximum correlation.
       Actually, let's look at the structure.
       np.correlate(h, 1-h, mode="full") at lag k (0-indexed in 'full' mode, center is N-1)
       C_k = sum_{i} h[i] (1-h[i+k]) for valid i.
       
       Wait, the standard approach for "minimize max of linear functions" is Linear Programming.
       Is C_k linear in h?
       C_k(h) = sum_i h[i] - sum_i h[i]h[i+k].
       The term sum_i h[i]h[i+k] is quadratic. So the objective is not linear.
       
       However, we can use a different perspective.
       We want to minimize t such that C_k(h) <= t * N / 2 for all k.
       Since C_k is concave in h, the set {h | C_k(h) <= t} is not necessarily convex?
       Actually, the hypograph of a concave function is convex.
       So minimizing the maximum of concave functions is a convex optimization problem?
       No, minimizing a convex function is convex. Maximizing a concave function is convex.
       But here we are minimizing the MAX of several functions.
       If the functions were convex, min max would be convex.
       Here the functions (correlations) are concave (due to -h^2 terms).
       So we are minimizing the maximum of concave functions. This is generally NON-convex.
       
       Let's re-evaluate.
       C_k(h) = sum h_i - sum h_i h_{i+k}.
       The term sum h_i h_{i+k} is a quadratic form with 0.5 on off-diagonals of a shift matrix.
       This quadratic form is indefinite?
       Let's check the Hessian.
       It's a sum of rank-1 matrices? No.
       
       Alternative Strategy:
       The previous best score was 1.3120 using "epigraph convex formulation".
       The prompt mentions: "inspiration 3 (score 1.3120): Rewritten: epigraph convex formulation (min t s.t. C_k <= t) + local refinement."
       This implies that the epigraph formulation WAS considered convex or treated as such successfully.
       Why would it be convex?
       If we fix the sum of h to be N/2, and h in [0,1].
       
       Let's look at the specific structure of np.correlate.
       For h constant 0.5, C_k = sum 0.5*0.5 = N/4. Value = (N/4)/N * 2 = 0.5.
       
       If we use a pattern like 0, 1, 0, 1...
       Sum = 50.
       h = [0, 1, 0, 1, ...]
       1-h = [1, 0, 1, 0, ...]
       Correlation at lag 0: sum h_i(1-h_i) = 0.
       Correlation at lag 1: sum h_i(1-h_{i+1}).
       If h_i=0, h_{i+1}=1 -> 0*0=0.
       If h_i=1, h_{i+1}=0 -> 1*1=1.
       So C_1 approx N/2.
       Value = (N/2)/N * 2 = 1.0.
       This is worse than 0.5.
       
       We want to MINIMIZE the max correlation.
       Constant 0.5 gives 0.5.
       Can we go lower?
       If we make h vary slowly?
       If h is a sine wave?
       
       Let's try a different approach: **Random Search with Local Refinement**.
       The current program uses DE + L-BFGS-B.
       Inspiration 2 and 3 achieved 1.3120.
       The current program is 1.0613.
       
       I will implement a **Simulated Annealing** approach which is robust for non-convex landscapes and can escape local minima better than L-BFGS-B alone, combined with a strict projection step.
       
       Key Insight: The objective function is piecewise differentiable (max of smooth functions).
       We can use a subgradient method or simply SA.
       
       Let's try a specific structured pattern first to see if we can beat 1.0613 (which corresponds to value ~0.47).
       Wait, score = seed/value. Seed value = 0.5.
       Score 1.0613 -> value = 0.5 / 1.0613 = 0.4711.
       Score 1.3120 -> value = 0.5 / 1.3120 = 0.3811.
       
       So we need to get the value down to ~0.38 or lower.
       
       Let's use **scipy.optimize.differential_evolution** with a better objective handling and **polish** enabled, or just a very tuned SA.
       
       Actually, the "epigraph convex formulation" inspiration suggests that maybe the problem *is* convex under the constraint sum(h)=N/2?
       Let's check the Hessian of C_k(h) again.
       C_k(h) = sum h_i - sum h_i h_{i+k}.
       Let Q_k be the matrix such that sum h_i h_{i+k} = h^T Q_k h.
       Q_k has 0.5 on the k-th super and sub diagonals.
       The eigenvalues of such a matrix are cos(pi * j / (N+1)) * 0.5?
       Actually, the matrix with 0.5 on off-diagonals of a shift has eigenvalues in [-0.5, 0.5].
       So -h^T Q_k h is a convex quadratic (since -Q_k is positive semi-definite if Q_k is negative semi-definite? No).
       If Q_k has eigenvalues in [-0.5, 0.5], then -Q_k has eigenvalues in [-0.5, 0.5]. It is indefinite.
       So the function is not convex.
       
       However, if we restrict to a specific subspace or use a different metric, it might be easier.
       
       Let's stick to a robust **Simulated Annealing** with a strong local refinement step using **Nelder-Mead** or **Powell** which don't require gradients and handle non-smoothness well.
       
       I will use `scipy.optimize.minimize` with `method='Powell'` on the raw objective, starting from multiple random restarts. Powell is derivative-free and handles non-smooth functions (like max of correlations) better than gradient-based methods.
       
       To ensure speed:
       1. Generate 10-20 random starting points (valid: sum=50, bounds [0,1]).
       2. For each, run Powell optimization.
       3. Keep the best.
       4. Time budget: 150s.
       
       Powell is slower than L-BFGS-B but more robust for non-smooth max functions.
       
       Let's refine the random start generation.
       Instead of uniform random, use "smooth" random starts, e.g., low-frequency sine waves with random phases and amplitudes, scaled to fit [0,1] and sum=50.
       """
    
    start_time = time.time()
    time_limit = 145.0
    
    def project_to_feasible(h):
        """Project h to satisfy sum(h)=N/2 and 0<=h<=1."""
        h = np.array(h, dtype=float)
        h = np.clip(h, 0.0, 1.0)
        diff = N/2.0 - np.sum(h)
        if abs(diff) < 1e-9:
            return h
        
        # Iterative water-filling style projection
        for _ in range(50):
            if diff > 0:
                # Increase elements that are < 1
                mask = h < 1.0 - 1e-12
                if not np.any(mask):
                    break
                count = np.sum(mask)
                add = diff / count
                # Check if adding 'add' pushes any over 1
                # If so, we need to distribute more carefully
                # Simple approach: just add and clip, then re-calculate diff
                h[mask] += add
            else:
                # Decrease elements that are > 0
                mask = h > 1e-12
                if not np.any(mask):
                    break
                count = np.sum(mask)
                sub = -diff / count
                h[mask] -= sub
            
            h = np.clip(h, 0.0, 1.0)
            diff = N/2.0 - np.sum(h)
            if abs(diff) < 1e-10:
                break
        
        # Final exact adjustment on a single element if possible to meet strict tolerance
        # The evaluator requires sum within 1e-9 * N.
        # The iterative method should be close.
        # To be safe, distribute the remaining tiny diff to the element with the most "room"
        if abs(diff) > 1e-12:
             # Find index with max room to move in direction of diff
             if diff > 0:
                 idx = np.argmax(1.0 - h)
             else:
                 idx = np.argmin(h)
             h[idx] += diff
             h = np.clip(h, 0.0, 1.0)
             # If clipping changed it, we might have a tiny error, but it's < 1e-9 usually
             
        return h

    def objective(h):
        """
        Compute the evaluation value.
        """
        # Ensure h is 1D
        h = np.asarray(h).flatten()
        corr = np.correlate(h, 1.0 - h, mode="full")
        return np.max(corr) / N * 2.0

    # Initial best is constant 0.5
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Generate diverse starting points
    starts = [best_h.copy()]
    
    # 1. Sine wave starts
    for freq in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for phase in [0.0, np.pi/4, np.pi/2, 3*np.pi/4]:
            x = np.linspace(0, 2*np.pi*freq, N)
            h_sine = 0.5 + 0.4 * np.sin(x + phase)
            h_sine = project_to_feasible(h_sine)
            starts.append(h_sine)
            
    # 2. Random smooth starts (Gaussian noise filtered)
    rng = np.random.default_rng(42)
    for _ in range(10):
        noise = rng.normal(0, 0.1, N)
        # Smooth it
        kernel = np.array([0.25, 0.5, 0.25])
        h_smooth = np.convolve(noise, kernel, mode='same')
        h_rand = 0.5 + h_smooth
        h_rand = project_to_feasible(h_rand)
        starts.append(h_rand)
        
    # 3. Block starts (e.g. half 0.2, half 0.8)
    for split in [20, 40, 60, 80]:
        h_block = np.zeros(N)
        h_block[:split] = 0.2
        h_block[split:] = 0.8
        # Adjust to sum 50
        # Sum is 0.2*split + 0.8*(100-split) = 0.2s + 80 - 0.8s = 80 - 0.6s
        # We want 50. 80 - 0.6s = 50 -> 0.6s = 30 -> s=50.
        # So 50/50 split of 0.2/0.8 works perfectly for sum.
        # Let's try other values.
        h_block = np.full(N, 0.5)
        h_block[:split] = 0.0
        h_block[split:] = 1.0
        # Sum is 100-split. We want 50. So split=50.
        # This is the binary case which gave value 1.0.
        # Let's try intermediate.
        h_block[:split] = 0.1
        h_block[split:] = 0.9
        # Sum = 0.1*s + 0.9*(100-s) = 0.1s + 90 - 0.9s = 90 - 0.8s
        # 90 - 0.8s = 50 -> 0.8s = 40 -> s=50.
        # So again, 50/50 split.
        # Let's just use random permutations of a few values.
        
    # Remove duplicates (approx)
    unique_starts = []
    for h in starts:
        is_dup = False
        for uh in unique_starts:
            if np.allclose(h, uh, atol=1e-3):
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(h)
    
    starts = unique_starts
    
    # Optimize each start
    from scipy.optimize import minimize
    
    for i, h_init in enumerate(starts):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        try:
            # Use Powell's method for derivative-free, non-smooth robustness
            res = minimize(
                objective, 
                h_init, 
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < best_val:
                h_opt = project_to_feasible(res.x)
                val_opt = objective(h_opt)
                
                # Verify validity
                if np.all(np.isfinite(h_opt)) and np.abs(np.sum(h_opt) - N/2.0) <= 1e-9 * N:
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
                        
        except Exception as e:
            pass
            
    # Final check
    if not np.all(np.isfinite(best_h)) or np.abs(np.sum(best_h) - N/2.0) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
