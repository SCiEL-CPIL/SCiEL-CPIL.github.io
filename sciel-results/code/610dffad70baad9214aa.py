"""
Strategy: Deterministic Parametric Search with Nelder-Mead and Coordinate Polish.

Rationale:
1. The current GA operates in 50D, which is a high-dimensional non-convex space.
   The objective function (involving convolution and norms) likely favors smooth,
   unimodal or simple bimodal structures due to the quadratic form in the numerator
   and the norm constraints in the denominator.
2. We restrict the search to a low-dimensional family of parametric shapes:
   - Super-Gaussian: h(x) = exp(-|x|^alpha)
   - Generalized Triangle: h(x) = (1 - |x|^beta)^gamma
   - Sine Bump: h(x) = sin(pi * x)^delta
   - Bimodal Super-Gaussian: h(x) = exp(-|x-c|^alpha) + exp(-|x+c|^alpha)
3. We use Nelder-Mead to optimize the parameters of these shapes. This is
   derivative-free and robust for low-dimensional problems.
4. After finding the best parametric shape, we perform a final coordinate-wise
   refinement in the 50D space to fine-tune individual heights, which can
   break symmetry and improve the value slightly.
5. This approach is much faster per evaluation than a full GA generation and
   allows us to explore many more distinct "basins" of the shape manifold.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
X = np.linspace(-0.5, 0.5, N) # Grid points for h

def evaluate(h):
    """
    Compute the value for a given set of step heights.
    h: array of shape (N,)
    Returns: float value
    """
    h = np.clip(h, 0, None)
    if np.all(h == 0):
        return 0.0
        
    g = np.convolve(h, h)
    m = len(g)
    
    # y construction: [0, g[0], ..., g[m-1], 0]
    # y has length m+2
    # We need sum over i in 0..m of (w/3) * (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
    # w = 1/(m+1)
    
    w = 1.0 / (m + 1)
    
    # Vectorized calculation of the quadratic form
    # y_i = y[:m+1]
    # y_ip1 = y[1:m+2]
    # Instead of building y explicitly, we can compute the sum directly from g.
    # y[0]=0, y[1]=g[0], ..., y[m]=g[m-1], y[m+1]=0
    
    # Terms:
    # i=0: y[0]=0, y[1]=g[0] -> 0 + 0 + g[0]^2 = g[0]^2
    # i=m: y[m]=g[m-1], y[m+1]=0 -> g[m-1]^2 + 0 + 0 = g[m-1]^2
    # i in 1..m-1: y[i]=g[i-1], y[i+1]=g[i]
    # Sum = g[0]^2 + g[m-1]^2 + sum_{k=0}^{m-2} (g[k]^2 + g[k]g[k+1] + g[k+1]^2)
    
    # Let's just construct y for clarity and safety, it's small (101 elements)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    
    y_i = y[:m+1]
    y_ip1 = y[1:m+2]
    
    terms = y_i**2 + y_i * y_ip1 + y_ip1**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
    else:
        return l2 / (norm_1 * norm_inf)

def parametric_shape(params, shape_type):
    """
    Generate a height vector from parameters.
    params: array of parameters
    shape_type: int identifier
    """
    if shape_type == 0:
        # Super-Gaussian: exp(-|x|^alpha)
        alpha = np.exp(params[0]) # Ensure positive
        h = np.exp(-np.abs(X)**alpha)
        # Scale to max 1
        if np.max(h) > 0:
            h /= np.max(h)
        return h
    elif shape_type == 1:
        # Generalized Triangle: (1 - |x|^beta)^gamma
        beta = np.exp(params[0])
        gamma = np.exp(params[1])
        base = 1 - np.abs(X)**beta
        base = np.clip(base, 0, None)
        h = base**gamma
        if np.max(h) > 0:
            h /= np.max(h)
        return h
    elif shape_type == 2:
        # Sine Bump: sin(pi * x)^delta
        delta = np.exp(params[0])
        base = np.sin(np.pi * X)
        base = np.clip(base, 0, None)
        h = base**delta
        if np.max(h) > 0:
            h /= np.max(h)
        return h
    elif shape_type == 3:
        # Bimodal Super-Gaussian: exp(-|x-c|^alpha) + exp(-|x+c|^alpha)
        alpha = np.exp(params[0])
        c = np.tanh(params[1]) * 0.5 # Center in [-0.5, 0.5]
        h = np.exp(-np.abs(X - c)**alpha) + np.exp(-np.abs(X + c)**alpha)
        if np.max(h) > 0:
            h /= np.max(h)
        return h
    else:
        # Fallback: Uniform
        return np.ones(N)

def objective(params, shape_type):
    """
    Objective function for scipy.optimize.minimize (minimizes negative value).
    """
    h = parametric_shape(params, shape_type)
    val = evaluate(h)
    return -val

def local_refine_50d(h, time_limit):
    """
    Coordinate-wise refinement in the full 50D space.
    """
    h = np.array(h, dtype=np.float64)
    best_h = h.copy()
    best_val = evaluate(best_h)
    t0 = time.time()
    
    # Simple coordinate descent with random order and step sizes
    step_sizes = [0.1, 0.05, 0.01, 0.005]
    
    while time.time() < time_limit:
        # Randomly pick a coordinate
        i = random.randint(0, N-1)
        improved = False
        
        # Try a few step sizes
        for step in step_sizes:
            for delta in [step, -step]:
                new_h = best_h.copy()
                new_h[i] = max(0, new_h[i] + delta)
                val = evaluate(new_h)
                if val > best_val:
                    best_val = val
                    best_h = new_h
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # If no improvement with large steps, try smaller
            # (handled by the loop above trying smaller steps)
            pass
            
    return best_h, best_val

def construct():
    deadline = time.time() + 140.0 # Safety margin
    
    best_h = np.ones(N)
    best_val = evaluate(best_h)
    
    # List of shape types to explore
    shape_types = [0, 1, 2, 3]
    
    # Initial parameter guesses for each shape type
    # We will run Nelder-Mead for each shape type from multiple starting points
    
    all_candidates = []
    
    # Pre-allocate time budget per shape type
    # 4 shape types, so ~30s each?
    # Let's do 20s per shape type for global search, then refine.
    
    for st in shape_types:
        if time.time() > deadline - 20:
            break
            
        # Multiple random initializations for each shape
        n_starts = 5
        for k in range(n_starts):
            if time.time() > deadline - 20:
                break
                
            if st == 0:
                # alpha in [1, 10] -> log(alpha) in [0, 2.3]
                init_params = np.array([np.log(random.uniform(1, 10))])
            elif st == 1:
                # beta in [0.5, 5], gamma in [1, 5]
                init_params = np.array([np.log(random.uniform(0.5, 5)), np.log(random.uniform(1, 5))])
            elif st == 2:
                # delta in [1, 10]
                init_params = np.array([np.log(random.uniform(1, 10))])
            elif st == 3:
                # alpha in [1, 10], c in [-0.5, 0.5] -> tanh^-1(c)
                c = random.uniform(-0.4, 0.4)
                init_params = np.array([np.log(random.uniform(1, 10)), np.arctanh(c)])
            else:
                continue
                
            try:
                # Use Nelder-Mead
                res = minimize(
                    objective, 
                    init_params, 
                    args=(st,), 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6}
                )
                
                if res.success:
                    h_candidate = parametric_shape(res.x, st)
                    val_candidate = evaluate(h_candidate)
                    all_candidates.append((h_candidate, val_candidate))
                    
                    if val_candidate > best_val:
                        best_val = val_candidate
                        best_h = h_candidate.copy()
                        
            except Exception:
                pass

    # If we found any candidates, pick the best one from all_candidates
    if all_candidates:
        # Sort by value
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        best_h, best_val = all_candidates[0]
    
    # Final refinement in 50D space
    # Allocate remaining time
    remaining_time = deadline - time.time() - 5
    if remaining_time > 2:
        best_h, best_val = local_refine_50d(best_h, time.time() + remaining_time)
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
