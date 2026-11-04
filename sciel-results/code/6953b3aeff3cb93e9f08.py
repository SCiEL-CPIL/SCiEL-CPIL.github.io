"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 400
DEADLINE = 150.0  # wall-clock limit from evaluator; keep margin

# EVOLVE-BLOCK-START

def _autoconv(x):
    """Full linear autoconvolution of x."""
    return np.convolve(x, x, mode='full')

def _c3prime(x):
    """Maximum absolute autoconvolution divided by squared sum."""
    ac = _autoconv(x)
    denom = np.sum(x) ** 2
    if denom == 0:
        return np.inf
    return np.max(np.abs(ac)) / denom

def _score(x):
    """The objective to minimize (lower is better)."""
    return _c3prime(x)

def _zadoff_chu_real_part(n, q):
    """
    Real part of a Zadoff-Chu-like sequence: Re(exp(1j * (pi * q * n^2 / N))).
    Here n is 0..N-1.
    """
    n = np.arange(n)
    phase = np.pi * q * (n ** 2) / n
    return np.cos(phase)

def construct():
    t_start = time.time()
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Strategy: 1D scalar search over chirp parameter q
    # The real part of a complex Zadoff-Chu sequence has near-zero autocorrelation
    # for the complex sequence, but the real part may have lower peaks than ones.
    
    def objective(q):
        x = _zadoff_chu_real_part(N, q[0])
        return _score(x)
    
    # First, do a coarse grid search to find a good region
    t_grid_start = time.time()
    q_values = np.linspace(-5, 5, 200)
    grid_scores = []
    for q in q_values:
        if time.time() - t_start > DEADLINE - 30:
            break
        x = _zadoff_chu_real_part(N, q)
        s = _score(x)
        grid_scores.append(s)
    
    if grid_scores:
        grid_scores = np.array(grid_scores)
        min_idx = np.argmin(grid_scores)
        q_init = q_values[min_idx]
        best_grid_score = grid_scores[min_idx]
        if best_grid_score < best_score:
            best_score = best_grid_score
            best_x = _zadoff_chu_real_part(N, q_init)
    
    # Now do a local refinement around the best grid point
    if time.time() - t_start < DEADLINE - 10:
        try:
            result = minimize(objective, np.array([q_init]), method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.success or result.fun < best_score:
                q_opt = result.x[0]
                x_opt = _zadoff_chu_real_part(N, q_opt)
                s_opt = _score(x_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_x = x_opt
        except Exception:
            pass
    
    # Also try differential evolution over a small range if time permits
    if time.time() - t_start < DEADLINE - 15:
        try:
            # Determine a reasonable range around the best found q
            q_lower = max(-5, q_init - 2)
            q_upper = min(5, q_init + 2)
            
            def de_objective(q):
                x = _zadoff_chu_real_part(N, q[0])
                return _score(x)
            
            de_result = differential_evolution(de_objective, 
                                               bounds=[(q_lower, q_upper)],
                                               maxiter=50, popsize=20,
                                               tol=1e-8, seed=42,
                                               polish=True)
            if de_result.fun < best_score:
                q_de = de_result.x[0]
                x_de = _zadoff_chu_real_part(N, q_de)
                best_score = de_result.fun
                best_x = x_de
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if time.time() - t_start < DEADLINE - 5:
        try:
            q_current = best_x  # need to recover q... let's store it
            # We need to track q separately
            # For now, just return best_x
            pass
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
