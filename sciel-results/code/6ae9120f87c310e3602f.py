import numpy as np
import time
from scipy.optimize import differential_evolution

def _union_length_1d(f, c_samples):
    """
    Compute the total length of the union of intervals for 1D problem.
    f: array of shape (8,)
    c_samples: array of shape (M,)
    Returns: array of shape (M,) with union lengths.
    """
    N = 8
    i_vals = np.arange(N, dtype=np.float64)
    # Intervals: [f[i] + c*i/8, f[i] + c*i/8 + 1/8]
    # starts: (M, 8)
    # ends: (M, 8)
    starts = f[None, :] + c_samples[:, None] * (i_vals[None, :] / N)
    ends = starts + 1.0 / N
    
    M = len(c_samples)
    union_lens = np.empty(M)
    
    for m in range(M):
        s = starts[m]
        e = ends[m]
        # Sort by start
        idx = np.argsort(s)
        s_sorted = s[idx]
        e_sorted = e[idx]
        
        total = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] <= cur_end:
                cur_end = max(cur_end, e_sorted[k])
            else:
                total += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
        total += cur_end - cur_start
        union_lens[m] = total
        
    return union_lens

def _volume_objective(params):
    """
    Objective function for optimization.
    params: array of shape (16,) -> f[0:8], g[8:16]
    """
    f = params[:8]
    g = params[8:16]
    
    # Use a fixed set of c samples for the integral approximation
    # 500 points should be accurate enough for optimization
    c_samples = np.linspace(0, 1, 500)
    
    lx = _union_length_1d(f, c_samples)
    ly = _union_length_1d(g, c_samples)
    
    # Volume is integral of lx(c) * ly(c) dc
    # Using trapezoidal rule
    V = np.trapz(lx * ly, c_samples)
    
    return V

def kakeya_needle_3d8() -> np.ndarray:
    deadline = time.time() + 580
    
    # Initial guess
    # Based on the seed program, f(i) = -i/16 was a starting point.
    # Let's provide a reasonable initial population or bounds.
    # The optimal f(i) is likely linear: f(i) = A * i + B.
    # Due to symmetry, B might be 0 or -0.5 (to center).
    # Let's use bounds [-1, 1] for each parameter.
    
    bounds = [(-1.0, 1.0)] * 16
    
    # Differential Evolution
    # maxiter and popsize control the budget.
    # Each evaluation of the objective takes ~500 * 8 operations, which is very fast.
    # We can afford many iterations.
    
    try:
        result = differential_evolution(
            _volume_objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20, 
            tol=1e-8,
            polish=True
        )
        best_params = result.x
    except Exception:
        # Fallback to a simple heuristic if DE fails or is too slow
        best_params = np.zeros(16)
        # f(i) = -i/16, g(j) = -j/16
        i_vals = np.arange(8)
        best_params[:8] = -i_vals / 16.0
        best_params[8:] = -i_vals / 16.0

    f = best_params[:8]
    g = best_params[8:16]
    
    # Construct positions
    N = 8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    x = f[i_flat]
    y = g[j_flat]
    
    positions = np.stack([x, y], axis=1)
    
    # Check time and validity
    # The problem states V must be > 0 and finite.
    # Our optimization minimizes volume, so it should be positive.
    # If the result is invalid (e.g. all zeros leading to 0 volume? No, width is 1/8), 
    # we might need a fallback. But width is fixed, so volume > 0 always.
    
    return positions
