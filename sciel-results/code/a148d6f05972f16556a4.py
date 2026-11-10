import numpy as np
import time
import scipy.optimize as so

N = 300
MAX_TIME = 110.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    
    # 1. Define the Ring Model
    # We assume the optimal packing consists of concentric rings.
    # Based on N=300, we expect roughly:
    # 1 center
    # 6, 12, 18, 24, 30, 36, 42, 48...
    # Sum: 1+6=7, +12=19, +18=37, +24=61, +30=91, +36=127, +42=169, +48=217, +54=271, +60=331.
    # 300 is close to 271 + 29.
    # Let's try a configuration that sums to 300.
    # Common heuristic for ring counts: k_r = 6*r for r>=1.
    # Let's try rings: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29 (last one partial?)
    # Or maybe 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 54 (too many).
    # Let's look at known packings for N=300.
    # Lai et al. 2021 packing for 300 circles.
    # Often these have "defects" but a ring model is a good approximation.
    # Let's define a fixed set of ring sizes that sum to 300.
    # Sizes: [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29] -> Sum = 300.
    # The last ring having 29 is awkward for symmetry.
    # Alternative: [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 54] is 331.
    # How about removing some from outer rings?
    # Let's try: [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]
    # Or [1, 6, 12, 18, 24, 30, 36, 42, 48, 29, 54]
    # Actually, let's just pick a set of ring counts that sums to 300 and allows for optimization.
    # A safer bet for "ring model" is to assume full rings where possible.
    # 1 + 6 + 12 + 18 + 24 + 30 + 36 + 42 + 48 + 54 = 271.
    # Remaining: 29.
    # Let's add a 10th ring of 29? Or distribute?
    # Let's use rings: [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]
    # Total rings: 11.
    # Variables:
    # - Radial distances for rings 1..10 (r_1 to r_10). r_0 = 0.
    # - Angular offset for rings 1..10. (Ring 0 has no angle).
    # Total variables: 10 radii + 10 angles = 20.
    # This is very low dimensional.
    
    ring_counts = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]
    num_rings = len(ring_counts)
    
    # Initial guess for radii: roughly uniform spacing?
    # Area per ring ~ 2*pi*r*dr.
    # Let's just start with linear spacing from 0.1 to 0.9.
    r_init = np.linspace(0.1, 0.9, num_rings - 1)
    theta_init = np.zeros(num_rings - 1)
    
    # Parameter vector: [r_1, r_2, ..., r_{K-1}, theta_1, theta_2, ..., theta_{K-1}]
    # Note: r_0 = 0 is fixed.
    # We optimize r_1..r_{K-1} and theta_1..theta_{K-1}.
    
    def build_points(params):
        """
        params: array of length 2*(K-1)
        First K-1 are radii r_1..r_{K-1}
        Next K-1 are angles theta_1..theta_{K-1}
        Returns: Nx2 array of points
        """
        K = num_rings
        r_vals = np.concatenate(([0.0], params[:K-1]))
        theta_vals = params[K-1:]
        
        points = []
        for i in range(K):
            count = ring_counts[i]
            r = r_vals[i]
            if count == 1:
                points.append([0.0, 0.0])
            else:
                # Base angle offset
                theta0 = theta_vals[i-1] if i > 0 else 0.0
                # Evenly spaced angles
                angles = theta0 + np.arange(count) * (2 * np.pi / count)
                xs = r * np.cos(angles)
                ys = r * np.sin(angles)
                points.extend(zip(xs, ys))
        
        return np.array(points)

    def score_params(params):
        """
        Calculate the minimum radius for a given set of parameters.
        """
        # Check time
        if time.time() - start_time > MAX_TIME - 5:
            return 0.0 # Return 0 to signal stop/invalid? 
            # Actually, for DE, we want to return a valid score. 
            # If we are close to timeout, we should probably just return a low score 
            # and rely on the outer loop to catch the deadline.
            
        x = build_points(params)
        
        # Check bounds
        # Points must be inside unit circle
        dists_center = np.linalg.norm(x, axis=1)
        if np.any(dists_center > 1.0):
            return 0.0 # Invalid
            
        # Pairwise distances
        # N=300, so 300x300 matrix is fine.
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(1.0 - dists_center)
        
        return min(min_pair, min_bound)

    # Optimization Strategy:
    # 1. Differential Evolution to find good radii and angles.
    # 2. Local refinement using Nelder-Mead or Powell on the best result.
    
    # Bounds for DE
    # Radii must be increasing and < 1.0
    # Angles can be anything (mod 2pi), let's bound them to [0, 2pi)
    
    # We need to enforce r_1 < r_2 < ... < r_{K-1} < 1.0
    # DE doesn't handle inequalities well directly.
    # We can transform variables:
    # Let u_i be unconstrained.
    # r_1 = 0.1 + u_1
    # r_2 = r_1 + 0.01 + u_2
    # ...
    # This is complex.
    # Alternative: Use bounds [0.05, 0.95] for all radii, and check monotonicity in score function?
    # If not monotonic, the geometry is invalid (rings cross), score 0.
    
    bounds = []
    for i in range(num_rings - 1):
        # Radii
        bounds.append((0.05, 0.95))
    for i in range(num_rings - 1):
        # Angles
        bounds.append((0.0, 2 * np.pi))
        
    # Initial population for DE
    # DE requires a population size.
    # Let's run DE for a limited number of generations.
    
    try:
        # Use a smaller population and fewer generations to stay within time
        # DE is stochastic.
        # We can use 'best1bin' strategy.
        
        # To ensure monotonic radii, we can penalize non-monotonic solutions heavily.
        def penalized_score(params):
            # Check monotonicity
            r_vals = params[:num_rings-1]
            for i in range(1, len(r_vals)):
                if r_vals[i] <= r_vals[i-1]:
                    return 0.0
            # Check last radius < 1.0
            if r_vals[-1] >= 1.0:
                return 0.0
            
            return score_params(params)

        # Run DE
        # maxiter=100, popsize=20
        # This might be slow. Let's estimate.
        # Score function is O(N^2) = 90k ops.
        # 20 * 100 * 20 (evals per gen?) ~ 40k evals.
        # 40k * 1ms = 40s. Might be tight.
        # Let's reduce popsize to 10 and maxiter to 50.
        
        de_result = so.differential_evolution(
            penalized_score, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4, 
            seed=42
        )
        
        best_params = de_result.x
        best_score = de_result.fun
        
    except Exception:
        # Fallback to initial guess if DE fails
        best_params = np.concatenate((r_init, theta_init))
        best_score = score_params(best_params)
        
    # 2. Local Refinement
    # Use Powell or Nelder-Mead to polish the result.
    # Powell is good for bound-constrained problems, but we have internal constraints (monotonicity).
    # We can just use Nelder-Mead on the penalized score, hoping it stays in valid region.
    # Or better: Project back to valid region if it goes out?
    # Let's just use Nelder-Mead with the penalized score.
    
    if time.time() - start_time < MAX_TIME - 10:
        try:
            lm_result = so.minimize(
                penalized_score, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if lm_result.fun > best_score:
                best_params = lm_result.x
                best_score = lm_result.fun
        except Exception:
            pass
            
    # 3. Final Check and Return
    final_points = build_points(best_params)
    
    # Verify validity
    if not np.all(np.isfinite(final_points)):
        return np.random.randn(N, 2) * 0.1
        
    dists_center = np.linalg.norm(final_points, axis=1)
    if np.any(dists_center > 1.0):
        # If invalid, try to project?
        # Or just return the best valid one found during optimization?
        # We don't store history. 
        # Let's try a quick fallback: scale down to fit.
        max_r = np.max(dists_center)
        if max_r > 0:
            final_points = final_points * (0.99 / max_r)
            
    return final_points
