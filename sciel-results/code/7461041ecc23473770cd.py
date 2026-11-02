"""
Strategy: Global Stochastic Search with Smooth Max Surrogate
Represents the step function h directly as a vector of non-negative heights.
Instead of spectral parameterization (which can be restrictive or hard to optimize locally), 
we use Differential Evolution (DE) to search the high-dimensional space of h directly.
To handle the non-differentiable nature of the 'max' in the objective function 
(minimize max(autoconv)), we use a soft-max approximation (log-sum-exp) which is 
smooth and allows the optimizer to focus on the 'hottest' parts of the convolution 
while remaining differentiable. This approach is structurally different from the 
IDCT-based method as it operates directly in the time domain with a global search 
heuristic rather than local gradient descent on a transformed parameter space.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def _smooth_max_surrogate(h, beta):
    """
    Compute a smooth approximation of the normalized max autoconvolution.
    Uses log-sum-exp to approximate the max, scaled by beta.
    Higher beta -> closer to true max.
    """
    s = np.sum(h)
    if s == 0:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    # Normalize convolution by (sum h)^2 to keep values O(1)
    norm_conv = conv / (s * s)
    
    # Soft max approximation: max(x) ≈ (1/beta) * log(sum(exp(beta * x)))
    # We want to minimize the max, so we minimize the soft max.
    # To avoid overflow, subtract max before exp.
    max_val = np.max(norm_conv)
    if max_val == 0:
        return 0.0
        
    # beta controls the sharpness. 
    # If beta is too small, it averages out the peaks. 
    # If beta is too large, it becomes non-smooth and DE struggles.
    # Let's use a moderate beta.
    shifted = beta * (norm_conv - max_val)
    # Clip to prevent underflow issues, though exp(-large) is just 0
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted))
    
    # The soft max value is max_val + (1/beta) * log_sum
    # We return this as the objective to minimize.
    return max_val + (1.0 / beta) * log_sum

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit

    def check_time():
        return time.time() > deadline

    # Initial best: Uniform distribution
    # h_i = 1 for all i
    # Autoconv of uniform is triangular. Peak is N. Sum is N.
    # Score = N / N^2 = 1/N = 1/600 ≈ 0.00166
    h_uniform = np.ones(N)
    best_score = compute_score(h_uniform)
    best_h = h_uniform.copy()

    # Define objective for DE
    # DE requires a function f(x) where x is the vector to optimize.
    # We optimize h directly. h >= 0.
    
    # We need to be careful with the number of function evaluations.
    # N=600 is high for DE. 
    # Standard DE population size is 15*N = 9000. 
    # Max iterations 50. Total evals ~ 450,000.
    # Each eval: convolve(600,600) -> 1199 elements. 
    # Numpy convolve is fast (C backend). 
    # 450k * (convolve + sum + logsumexp) might be too slow.
    
    # Let's reduce the search space or use a smaller population.
    # Or, we can use a smaller N for the search and then refine? 
    # No, the task requires 600 intervals.
    
    # Alternative: Use a lower-dimensional parameterization that is still global?
    # The prompt asks for a "step function with 600 equal intervals".
    # We must output 600 values.
    
    # Let's try to speed up the evaluation or reduce iterations.
    # If we use maxiter=10 and popsize=5 (5*600=3000 individuals), 
    # total evals = 30,000. This is feasible.
    
    # However, DE with popsize=5 is very weak.
    # Let's try popsize=10, maxiter=5. Total evals = 15,000.
    
    # Objective function for DE
    beta = 50.0 # Sharpness of soft max
    
    def objective(h):
        if check_time():
            return 1e10
        # Ensure non-negativity (DE bounds handle this, but just in case)
        h = np.maximum(0, h)
        return _smooth_max_surrogate(h, beta)

    # Bounds: [0, 1] for each height. 
    # Scaling doesn't matter for the score, so [0,1] is a good canonical range.
    bounds = [(0.0, 1.0)] * N
    
    # DE Options
    # popsize: multiplier for N. 
    # maxiter: number of generations.
    # tol: convergence tolerance.
    # mutation: (lo, hi) tuple.
    # crossover: CR probability.
    
    try:
        # Use a smaller population to fit in time budget
        # popsize=5 -> 3000 individuals. 
        # maxiter=10 -> 30,000 evaluations.
        # Each evaluation is ~ O(N^2) for conv? No, np.convolve is O(N^2) in worst case but highly optimized.
        # For N=600, convolve is very fast (<1ms).
        # 30,000 * 1ms = 30s. This is safe.
        
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=10, 
            popsize=5, 
            tol=1e-4, 
            mutation=(0.5, 1.5), 
            crossover=0.9,
            seed=42,
            polish=True, # Try to polish the result with L-BFGS-B at the end? 
                         # Polish might be slow or fail if gradient is bad.
                         # Let's set polish=False to be safe, or try it if time permits.
                         # Actually, polish uses L-BFGS-B which needs gradients. 
                         # Our objective is smooth, so gradients are estimated by finite diff? 
                         # No, scipy DE polish uses L-BFGS-B which uses finite diff if no jac provided.
                         # Finite diff on N=600 is 600 extra evals per polish step. 
                         # Might be too slow. Let's disable polish for now.
                         # We can do a manual local refinement if time allows.
                         polish=False
        )
        
        if not check_time():
            h_opt = result.x
            # Compute the TRUE score (not surrogate) for the best candidate
            true_score_opt = compute_score(h_opt)
            
            if true_score_opt < best_score:
                best_score = true_score_opt
                best_h = h_opt

        # If we have time, try a local refinement on the best DE result
        # using a smoother objective (higher beta) or just L-BFGS-B on the true max?
        # True max is not differentiable. 
        # We can use the soft max with high beta as a proxy for local refinement.
        
        if not check_time():
            # Local refinement with L-BFGS-B on the soft max objective
            # Start from best_h
            # Use higher beta for sharper approximation
            beta_refine = 100.0
            
            def objective_refine(h):
                if check_time():
                    return 1e10
                h = np.maximum(0, h)
                return _smooth_max_surrogate(h, beta_refine)
            
            from scipy.optimize import minimize
            
            # L-BFGS-B with bounds
            bounds_refine = [(0.0, 1.0)] * N
            
            res_refine = minimize(
                objective_refine, 
                best_h, 
                method='L-BFGS-B', 
                bounds=bounds_refine,
                options={'maxiter': 50, 'maxls': 10}
            )
            
            if not check_time():
                h_refined = res_refine.x
                true_score_refined = compute_score(h_refined)
                
                if true_score_refined < best_score:
                    best_score = true_score_refined
                    best_h = h_refined

    except Exception:
        # Fallback to uniform if anything fails
        pass

    # Return the best h found
    # Ensure it's a list of floats
    return best_h.tolist()

# EVOLVE-BLOCK-END
