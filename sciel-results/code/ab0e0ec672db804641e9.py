import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600
W = 1.0 / 1200.0

# Precompute fine grid mapping for fast ratio evaluation
# We use a finer grid to accurately estimate the max of the autoconvolution
FINE_N = 6000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map each fine grid point to the corresponding coarse step index
# x is in [-0.25, 0.25). Step i covers [-0.25 + i*W, -0.25 + (i+1)*W)
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of 600 nonnegative step heights.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    # Interpolate coarse steps to fine grid
    f_fine = h[CELL]
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX  # Approximate integral
    
    max_conv = np.max(conv)
    
    # Normalize by L2 norm squared (which is the value of autoconvolution at 0 if continuous, 
    # but for discrete steps, we use sum(h^2)*W as a proxy for the L2 norm squared of the step function)
    # The task asks for "normalized maximum of its autoconvolution". 
    # Usually this means max(f*f) / ||f||_2^2.
    # ||f||_2^2 = sum(h_i^2 * W)
    l2_sq = np.sum(h ** 2) * W
    if l2_sq < 1e-15:
        return float('inf')
    
    return max_conv / l2_sq

def objective(params):
    """
    Objective function for scipy.optimize.differential_evolution.
    params: array of 600 values.
    """
    h = params
    return ratio_from_h(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Initial best guess: constant function
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    # Define bounds for Differential Evolution
    # We can scale the heights. The ratio is scale-invariant, so absolute scale doesn't matter.
    # However, DE needs finite bounds. Let's constrain heights to [0, 10].
    # Since the ratio is invariant to scaling, the optimizer will find the shape, and scale is irrelevant.
    # To avoid numerical issues with very large/small numbers, we keep them in a reasonable range.
    bounds = [(0.0, 10.0) for _ in range(N)]

    # Differential Evolution is a global stochastic optimizer.
    # It is robust to non-convexity and doesn't require gradients.
    # maxiter and tol control convergence.
    # 'seed' ensures reproducibility if needed, but random is fine.
    
    try:
        # Run Differential Evolution
        # popsize: size of population (default 15). Larger is more robust but slower.
        # maxiter: max number of generations.
        # tol: convergence tolerance.
        # The time limit is the main constraint. DE will stop early if it hits maxiter or tol.
        # We rely on the deadline check in the wrapper if we were to do manual loop, 
        # but scipy's DE doesn't have a time-based callback easily.
        # So we must estimate maxiter such that it finishes within time.
        # Each evaluation takes ~ (FFT size) operations. FINE_N=6000 -> N_FFT=16384.
        # FFT is fast. Let's benchmark roughly or just set a conservative maxiter.
        # A single DE generation evaluates popsize * N * something.
        # Let's try maxiter=100, popsize=10.
        
        # To be safe with the 140s limit, we might need to run multiple smaller DE runs 
        # or one large one if it's fast enough.
        # Let's check time per function call.
        t0 = time.time()
        ratio_from_h(best_h)
        t1 = time.time()
        eval_time = t1 - t0
        
        # Estimate how many evaluations we can afford.
        # 140s / eval_time.
        # DE with popsize P and maxiter M does ~ P*M evaluations (roughly).
        # Actually DE does more due to crossover/mutation, but P*M is a lower bound.
        # Let's aim for ~2000-5000 evaluations total.
        
        max_affordable_evals = int((deadline - time.time()) / (eval_time * 2)) # Factor 2 for safety
        if max_affordable_evals < 100:
            max_affordable_evals = 100
            
        # Choose popsize and maxiter
        popsize = 10
        maxiter = max(10, max_affordable_evals // popsize)
        
        # Cap maxiter to avoid running too long if eval_time is underestimated
        maxiter = min(maxiter, 50) # Conservative cap
        
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=maxiter, 
            popsize=popsize,
            tol=1e-4,
            seed=42
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_h = res.x.copy()
            
    except Exception:
        pass

    # If we have time left, try a local polish with Nelder-Mead on the best found
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            # Use the best h from DE as start point for local refinement
            # Nelder-Mead can handle the non-differentiability of the max operation better than L-BFGS sometimes,
            # or we can just do a few random restarts with Nelder-Mead.
            
            # Since the landscape is likely rugged, let's do a few short Nelder-Mead runs from slightly perturbed best_h
            for i in range(3):
                if time.time() > deadline - 2.0:
                    break
                perturbation = np.random.normal(0, 0.1, N)
                h_start = np.clip(best_h + perturbation, 0, None)
                
                res_nm = minimize(
                    objective, 
                    h_start, 
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-6}
                )
                
                if res_nm.fun < best_val:
                    best_val = res_nm.fun
                    best_h = res_nm.x.copy()
        except Exception:
            pass

    return list(best_h)
