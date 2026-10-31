import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h)."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _normalize(h):
    """Enforce exact sum N/2 and bounds [0,1]."""
    h = h.copy()
    for _ in range(100):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(x, alpha=500.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _two_block_init():
    """
    Analytic 2-block initial guess.
    The optimal 2-step function has heights A and B such that A+B=1 
    and the overlap max is minimized. 
    For a 2-step function with period 2 (A, B), the overlaps are:
    A(1-B) = A^2
    B(1-A) = B^2
    To minimize max(A^2, B^2) subject to A+B=1, we set A=B=0.5.
    However, the "2-block" structure in the context of the whole interval [0,2] 
    often refers to a step function that is low for half and high for the other half.
    Let's use the specific analytic solution for the 2-periodic case which is just 0.5, 0.5.
    But a better initial guess for general search is often derived from the 
    continuous relaxation. 
    Actually, the known optimal value for the continuous case is related to 
    the golden ratio. Let's try a pattern that approximates the optimal 
    distribution. 
    A common strong initial guess is alternating values close to 0.5 but 
    slightly perturbed or using the values from the 2-step optimal of the 
    continuous problem.
    
    For the discrete case, let's just use a uniform 0.5 as a safe baseline, 
    but the DE will handle the exploration.
    """
    return np.full(N, 0.5)

def _construct_h_from_base(b, p):
    """Construct full h vector from base vector b of length p."""
    h = np.tile(b, N // p + 1)[:N]
    return h

def _de_objective_periodic(b, p):
    """
    Objective for Differential Evolution on the base vector b.
    We project b to a valid h, then compute the score.
    To make DE work well, we use a smooth approximation or the raw score.
    Raw score is non-differentiable but DE doesn't need gradients.
    However, we must ensure the sum constraint.
    """
    h = _construct_h_from_base(b, p)
    h = _normalize(h)
    return _score(h)

def _slsqp_polish(h, deadline):
    """
    Polish a given h using SLSQP with epigraph formulation.
    """
    if time.time() > deadline:
        return h, _score(h)
    
    def obj(params):
        h_loc = params[:N]
        t = params[N]
        shifted = np.zeros(N)
        shifted[:-1] = h_loc[1:]
        overlap = h_loc * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha=1000.0)
        # Penalty for sum constraint to help SLSQP if it drifts, 
        # though we use equality constraint.
        s = np.sum(h_loc)
        sum_pen = 100.0 * (s - N/2.0)**2
        # Epigraph penalty: we want t >= max(overlap)
        # Minimize t + penalty(t < smax)
        pen = 1000.0 * np.maximum(0.0, smax - t)**2
        return t + sum_pen + pen

    def grad(params):
        h_loc = params[:N]
        t = params[N]
        shifted = np.zeros(N)
        shifted[:-1] = h_loc[1:]
        overlap = h_loc * (1.0 - shifted)
        
        x_max = np.max(overlap)
        if x_max == 0:
            alpha = 1000.0
            exps = np.exp(alpha * (overlap - x_max))
            weights = exps / np.sum(exps)
            smax = 0.0
        else:
            alpha = 1000.0
            exps = np.exp(alpha * (overlap - x_max))
            weights = exps / np.sum(exps)
            smax = x_max + (1.0/alpha) * np.log(np.sum(exps))
            
        val = t + 1000.0 * np.maximum(0.0, smax - t)**2 + 100.0 * (np.sum(h_loc)-N/2.0)**2
        
        g = np.zeros(N+1)
        
        # Gradient of sum penalty
        g[:N] += 200.0 * (np.sum(h_loc)-N/2.0)
        
        # Gradient of epigraph penalty
        if smax > t:
            dp = 2000.0 * (smax - t)
            # d(smax)/d(h_i)
            # overlap_i = h_i (1-h_{i+1})
            # d/dh_i = (1-h_{i+1})
            # d/dh_{i+1} = -h_i
            for i in range(N-1):
                if weights[i] > 1e-15:
                    g[i] += dp * weights[i] * (1.0 - h_loc[i+1])
                    g[i+1] += dp * weights[i] * (-h_loc[i])
            if weights[N-1] > 1e-15:
                g[N-1] += dp * weights[N-1] * 1.0
            g[N] -= dp
        else:
            g[N] = 1.0
            
        return val, g

    def con_sum(params):
        return np.sum(params[:N]) - N/2.0

    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    x0 = np.concatenate([h, [_score(h) + 0.01]])
    
    try:
        res = minimize(
            obj, 
            x0, 
            jac=grad,
            method='SLSQP', 
            bounds=bounds, 
            constraints=[{'type': 'eq', 'fun': con_sum}],
            options={'maxiter': 200, 'ftol': 1e-14, 'disp': False}
        )
        if res.success or res.fun < 10.0:
            h_opt = res.x[:N]
            h_opt = _normalize(h_opt)
            return h_opt, _score(h_opt)
    except Exception:
        pass
    return h, _score(h)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_h = _two_block_init()
    best_score = _score(best_h)
    
    # 1. Try periodic structures with DE + SLSQP
    # We search for periods p. 
    # Small p (like 2, 3, 4) are likely candidates for simple structures.
    # Larger p allows more complex structures.
    # We limit p to avoid too many variables for DE.
    
    periods_to_try = [2, 3, 4, 5, 10, 20, 25, 50]
    
    # Add N itself (full freedom) as a "period" handled separately or in DE loop
    # But DE on N=100 is expensive. We'll do a quick DE on N=100 later if time permits.
    
    for p in periods_to_try:
        if time.time() > deadline - 10.0:
            break
            
        # Initial guess for base vector b
        # Uniform 0.5
        b0 = np.full(p, 0.5)
        
        # Run Differential Evolution on b
        # popsize=10, maxiter=50 -> 500 * p evaluations.
        # For p=50, 25000 evals. Each eval is O(N). 2.5M ops. Fast.
        try:
            de_res = differential_evolution(
                _de_objective_periodic, 
                bounds=[(0.0, 1.0)] * p, 
                args=(p,),
                popsize=10,
                maxiter=50,
                tol=1e-8,
                seed=42,
                polish=False # We will polish with SLSQP
            )
            
            b_opt = de_res.x
            h_candidate = _construct_h_from_base(b_opt, p)
            h_candidate = _normalize(h_candidate)
            score_candidate = _score(h_candidate)
            
            # Polish with SLSQP
            h_polished, score_polished = _slsqp_polish(h_candidate, deadline)
            
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished.copy()
                
        except Exception:
            pass
            
    # 2. If time remains, try a global DE on the full N-vector
    # This is more expensive but can find non-periodic optima.
    if time.time() < deadline - 15.0:
        try:
            def de_objective_full(h):
                h = _normalize(h)
                return _score(h)
                
            # Use a smaller population to fit in time
            de_res_full = differential_evolution(
                de_objective_full, 
                bounds=[(0.0, 1.0)] * N, 
                popsize=5,
                maxiter=20,
                tol=1e-6,
                seed=123,
                polish=False
            )
            
            h_candidate = _normalize(de_res_full.x)
            score_candidate = _score(h_candidate)
            
            if score_candidate < best_score:
                # Polish it
                h_polished, score_polished = _slsqp_polish(h_candidate, deadline)
                if score_polished < best_score:
                    best_score = score_polished
                    best_h = h_polished.copy()
        except Exception:
            pass

    # Final safety check: ensure return is valid
    best_h = _normalize(best_h)
    
    return best_h
