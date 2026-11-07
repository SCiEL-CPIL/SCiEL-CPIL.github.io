import numpy as np
import time

# Global state to track elapsed time and number of calls
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    Uses a pure Lower Confidence Bound (LCB) approach with an adaptive kappa.
    """
    global _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 330s limit, leave a safety margin of 10s
    if time.time() - _start_time > 320:
        # Return a random or simple score if we are about to timeout
        # This is unlikely to happen as the function is fast, but just in case
        return np.random.rand(len(mu))
    
    _call_count += 1
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-8)
    
    # Adaptive kappa for LCB
    # In the early stages, we want more exploration (higher kappa).
    # As we get more evaluations, we can shift towards exploitation (lower kappa).
    # The total number of evaluations is 30.
    # We have 5 initial points, then 25 more.
    # So _call_count goes from 1 to 25 (for the optimization loop part).
    
    # A simple adaptive strategy:
    # Start with a higher kappa to explore the domain, then decrease it.
    # The ST function is multimodal with 5 minima. We need to find the global one.
    # Global minimum is around -39.166.
    
    # Let's use a schedule that decays kappa.
    # kappa = max_kappa * (1 - call_count / total_calls) + min_kappa
    # But we don't know total_calls exactly inside, we know it's 25 steps.
    # Let's assume the loop runs for 25 times.
    
    # Alternatively, a fixed kappa often works well for 1D problems.
    # Let's try a slightly high kappa to ensure we explore all basins.
    # For 1D, LCB with kappa ~ 1.0 to 2.0 is common.
    
    # Let's try an adaptive kappa based on the number of calls.
    # Early: kappa = 2.0 (more exploration)
    # Late: kappa = 0.5 (more exploitation)
    
    # Since we don't know the exact total, let's assume 25 steps.
    # If this is the first call, it's likely step 1 of 25.
    
    # A robust choice for 1D ST function:
    # The function has 5 local minima. The global one is at x ~ -1.717.
    # The others are at x ~ 1.283, -3.283, 3.283, -0.283 (approx).
    # Actually, let's look at the structure.
    # ST(x) = 0.5 * x * (x^2 - 5)^2
    # Minima at x = -1.717, 1.717 (local max? no), let's check.
    # f'(x) = 0.5 * (x^2 - 5)^2 + 0.5 * x * 2 * (x^2 - 5) * 2x
    #       = 0.5 (x^2 - 5) [ (x^2 - 5) + 4x^2 ]
    #       = 0.5 (x^2 - 5) (5x^2 - 5)
    #       = 2.5 (x^2 - 5) (x^2 - 1)
    # Roots: x = +/- 1, +/- sqrt(5) ~ +/- 2.236
    # f(-2.236) = 0.5 * (-2.236) * (5 - 5)^2 = 0
    # f(-1) = 0.5 * (-1) * (1 - 5)^2 = 0.5 * (-1) * 16 = -8
    # f(0) = 0
    # f(1) = 0.5 * 1 * (1 - 5)^2 = 8
    # f(2.236) = 0
    # Wait, the global minimum is -39.166?
    # Let's re-calculate.
    # Styblinski-Tang 1D: f(x) = 0.5 * x * (x^2 - 5)^2
    # Let's check x = -1.717.
    # f(-1.717) = 0.5 * (-1.717) * (2.948 - 5)^2 = 0.5 * (-1.717) * (-2.052)^2
    #           = 0.5 * (-1.717) * 4.21 = -3.61
    # This is not -39.
    
    # Ah, the standard ST function is often defined in N dimensions.
    # The 1D version might be different or I am misremembering the minimum.
    # The task says "minimum value -39.1662".
    # Let's check the 5D ST function minimum.
    # f(x) = 0.5 * sum(x_i * (x_i^2 - 5)^2)
    # If x_i = -1.717 for all i, f = 5 * 0.5 * (-1.717) * (2.948 - 5)^2 = 5 * (-3.61) = -18.05.
    # Still not -39.
    
    # Let's check the definition again.
    # Maybe the domain is different or the function is scaled?
    # Or maybe the function is:
    # f(x) = sum(0.5 * x_i * (x_i^2 - 5)^2)
    # Global min is at x_i = -1.717.
    # Value is -18.05 for 5D.
    # For 1D, min is -3.61.
    
    # Why does the task say -39.1662?
    # Perhaps the function is:
    # f(x) = sum(x_i^2 - 5)^2 ? No.
    
    # Let's trust the task description: "minimizes styblinski tang 1d on [-5, 5]^1, minimum value -39.1662".
    # This implies the function provided by the evaluator has this minimum.
    # I will not try to guess the function form, but rely on the GP.
    
    # Since I don't know the exact function, I must rely on the GP's mu and sigma.
    # The LCB approach is robust.
    
    # Let's use a fixed kappa that balances exploration and exploitation well for 1D.
    # A common choice is kappa = 1.0 or 1.5.
    # Given the multimodal nature, a slightly higher kappa is better to escape local minima.
    
    kappa = 1.5
    
    # LCB score = mu - kappa * sigma
    # We want to MINIMIZE the function, so we want to MAXIMIZE the acquisition score.
    # The acquisition score is typically defined as:
    # a(x) = -mu(x) + kappa * sigma(x)  (to maximize)
    # or
    # a(x) = mu(x) - kappa * sigma(x)   (to minimize)
    
    # The task says: "Return 2000 finite scores, higher meaning evaluate next".
    # So we want to return scores where higher is better.
    # Since we are minimizing f, a low mu is good. A high sigma is good (uncertainty).
    # So we want to maximize: -mu + kappa * sigma.
    
    scores = -mu + kappa * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
