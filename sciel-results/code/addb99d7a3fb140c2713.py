"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def get_sq_dists(pts):
        """Calculate squared distances between all pairs."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score(pts):
        """Calculate the raw score: (max_d / min_d)^2."""
        d_sq = get_sq_dists(pts)
        d_min_sq = np.min(d_sq)
        d_max_sq = np.max(d_sq)
        if d_min_sq < 1e-12:
            return 1e10
        return d_max_sq / d_min_sq

    def smooth_objective(x):
        """
        Smooth surrogate objective for optimization.
        Minimizes log(max_d^2) - log(min_d^2) using log-sum-exp approximations.
        This is differentiable and allows gradient-based or quasi-Newton methods,
        but we use DE which doesn't strictly need gradients, yet benefits from
        the smooth landscape over the sharp corners of the raw ratio.
        """
        pts = x.reshape(N, 2)
        d_sq = get_sq_dists(pts)
        
        # Avoid log(0)
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Temperature for log-sum-exp
        # Lower T -> closer to true max/min. Higher T -> smoother.
        # We want a balance.
        T = 0.1 
        
        # log(max) approx
        # max(d^2) = T * log(sum(exp(d^2/T)))
        # log(max(d^2)) = log(T) + logsumexp(d^2/T)
        
        # min(d^2) = -T * log(sum(exp(-d^2/T)))
        # log(min(d^2)) = log(T) + logsumexp(-d^2/T) - log(T) ... wait.
        # log(min(d^2)) = log(-T * logsumexp(-d^2/T)) = log(T) + log(-logsumexp(-d^2/T))
        # Actually, simpler:
        # min_val = np.exp(-T * logsumexp(-d_sq/T))
        # log_min = -T * logsumexp(-d_sq/T) + log(T)
        
        # Let's use scipy.special.logsumexp if available, else manual.
        # Manual logsumexp for stability:
        def logsumexp(v):
            v_max = np.max(v)
            return v_max + np.log(np.sum(np.exp(v - v_max)))
        
        log_max_sq = logsumexp(d_sq / T) + np.log(T)
        
        # For min:
        # min(d^2) = exp(-T * logsumexp(-d^2/T))
        # log(min(d^2)) = -T * logsumexp(-d^2/T) + log(T)
        log_min_sq = -T * logsumexp(-d_sq / T) + np.log(T)
        
        # Objective is log(max/min) = log_max - log_min
        # We want to minimize (max/min)^2, which is equivalent to minimizing 2*(log_max - log_min)
        # So we minimize (log_max_sq - log_min_sq)
        obj_val = log_max_sq - log_min_sq
        
        # Add a penalty for points being too close to each other (distinctness)
        # The log_min term already handles this, but if min gets too small, 
        # the gradient might explode. 
        # The DE algorithm handles bounds, so we rely on that.
        
        return obj_val

    # Define bounds for the optimization.
    # We can normalize the points to fit in a box, e.g., [-1, 1].
    # Since the problem is scale invariant, we can fix the scale or let the optimizer find it.
    # However, fixing the scale reduces the dimensionality of the search space effectively.
    # Let's constrain the points to lie within a box of size [-1, 1].
    # Actually, the optimal configuration will likely span the available space.
    # Let's use bounds [-1, 1] for each coordinate.
    bounds = [(-1.0, 1.0)] * (N * 2)

    # Differential Evolution
    # popsize: number of individuals in the population. 
    # For 32 dimensions, popsize=10 means 320 evaluations per generation.
    # maxiter: number of generations.
    # We have ~140 seconds. 
    # Let's estimate time per evaluation. 
    # get_sq_dists is O(N^2 * D) = 256 * 2 = 512 ops. Very fast.
    # logsumexp is O(N^2) = 256 ops.
    # So one eval is < 1ms.
    # 10000 evals would take < 10s.
    # We can afford a much larger search.
    
    # Strategy:
    # 1. Run DE with a reasonable popsize and maxiter.
    # 2. If time remains, run another DE with different parameters or refine the best result.
    
    best_x = None
    best_f = float('inf')
    
    # Helper to check time
    def time_left():
        return deadline - time.time()

    # Run 1: Global Search
    # popsize=15, maxiter=200
    # tol=1e-6
    try:
        res = differential_evolution(
            smooth_objective,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False, # We will do our own refinement or rely on DE's convergence
            updating='immediate'
        )
        
        if res.success or res.fun < best_f:
            best_f = res.fun
            best_x = res.x
            
    except Exception as e:
        pass

    if time_left() > 20:
        # Run 2: Refinement or Second Global Search with different seed/params
        # If we have a good candidate, we can try to polish it with a local optimizer
        # or run DE again with tighter bounds around the best solution found.
        
        if best_x is not None:
            # Define tighter bounds around the best solution
            # The solution is scale invariant, but DE might have found a specific scale.
            # Let's just run DE again with a different seed to see if we find a better global min.
            try:
                res2 = differential_evolution(
                    smooth_objective,
                    bounds,
                    maxiter=100,
                    popsize=10,
                    tol=1e-7,
                    seed=123,
                    polish=False
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x
            except Exception as e:
                pass
            
            # If time is still left, do a local polish using Nelder-Mead on the raw score
            # starting from the best DE result. This helps jump out of the "smooth" approximation
            # and onto the true sharp minimum.
            if time_left() > 5 and best_x is not None:
                from scipy.optimize import minimize
                try:
                    res3 = minimize(
                        score, 
                        best_x, 
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                    )
                    if res3.fun < best_f:
                        best_f = res3.fun
                        best_x = res3.x
                except Exception as e:
                    pass

    # Fallback if DE failed completely
    if best_x is None:
        # Simple grid or random
        np.random.seed(0)
        best_x = np.random.uniform(-1, 1, (N, 2)).ravel()
        
    return best_x.reshape(N, 2)

# EVOLVE-BLOCK-END
