"""Rewritten program: Low-dim global search + High-dim local refinement."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

N = 400
# Domain is [-1/4, 1/4] mapped to indices 0..N-1
# x[i] = -0.25 + i * (0.5 / N)
x = -0.25 + np.arange(N) * (0.5 / N)

def _value(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s * s))

def _h_from_gaussians(params):
    """
    Construct h from parameters defining a sum of Gaussians.
    params: [amp1, center1, width1, amp2, center2, width2, ...]
    Centers are in [-1/4, 1/4]. Widths are positive.
    """
    h = np.zeros(N)
    n_gaussians = len(params) // 3
    for i in range(n_gaussians):
        amp = params[3*i]
        center = params[3*i + 1]
        width = abs(params[3*i + 2]) + 1e-6 # Ensure positive width
        
        # Gaussian: amp * exp(-0.5 * ((x - center) / width)^2)
        h += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    np.random.seed(42)
    
    # --- Phase 1: Global Search in Low-Dim Space ---
    # We use 5 Gaussians (15 parameters) to capture general shapes.
    n_gaussians = 5
    n_params = 3 * n_gaussians
    
    # Bounds for parameters:
    # Amplitude: [-10, 10]
    # Center: [-0.25, 0.25]
    # Width: [0.01, 0.5] (absolute width in x-domain)
    bounds = []
    for _ in range(n_gaussians):
        bounds.append((-10.0, 10.0)) # Amp
        bounds.append((-0.25, 0.25)) # Center
        bounds.append((0.01, 0.5))   # Width
    
    def objective_lowdim(params):
        h = _h_from_gaussians(params)
        s = np.sum(h)
        if s == 0:
            return 1e6
        return _value(h)
    
    # Run Differential Evolution for a fixed portion of time
    # DE is stochastic and can be slow, so we limit it.
    time_budget_de = 60.0
    de_end_time = start_time + time_budget_de
    
    best_params = None
    try:
        # Use a small population to keep it fast
        result_de = differential_evolution(
            objective_lowdim, 
            bounds, 
            maxiter=100, 
            popsize=10, 
            tol=1e-4,
            seed=42,
            polish=False # We will polish later
        )
        if result_de.success or result_de.fun < 1e6:
            best_params = result_de.x
    except Exception as e:
        print(f"DE failed: {e}")
        
    if best_params is None:
        # Fallback to random initialization if DE fails or times out badly
        best_params = np.array([
            np.random.uniform(-10, 10), 
            np.random.uniform(-0.25, 0.25), 
            np.random.uniform(0.01, 0.5)
        ] * n_gaussians)
        
    h_initial = _h_from_gaussians(best_params)
    best_h = h_initial.copy()
    best_val = _value(h_initial)
    
    # --- Phase 2: Local Refinement in High-Dim Space ---
    # Use L-BFGS-B on the full 400-dim vector h
    # This allows the shape to deviate from the Gaussian sum if beneficial
    
    def objective_highdim(h):
        return _value(h)
    
    # We need to be careful with time here. L-BFGS-B can take many steps.
    # We will run it with a time limit by checking inside a callback or just limiting maxiter.
    # Since _value is fast, we can afford a decent number of iterations.
    
    # To ensure we don't exceed time, we run minimize in a loop or just set a strict maxiter.
    # Let's try a single minimize call with a reasonable maxiter.
    
    current_h = h_initial.copy()
    
    # Check if we have time left
    if time.time() < deadline:
        try:
            result_lbfgs = minimize(
                objective_highdim, 
                current_h, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False}
            )
            if result_lbfgs.fun < best_val:
                best_val = result_lbfgs.fun
                best_h = result_lbfgs.x
        except Exception as e:
            print(f"L-BFGS-B failed: {e}")
            
    # If we still have time, we might try a few more random restarts of local optimization
    # from slightly perturbed versions of the best solution found by DE
    if time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() >= deadline - 5.0:
                break
            # Perturb the best_h
            h_perturbed = best_h + 0.1 * np.random.randn(N)
            try:
                res = minimize(
                    objective_highdim, 
                    h_perturbed, 
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'disp': False}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_h = res.x
            except Exception:
                pass
                
    return best_h
