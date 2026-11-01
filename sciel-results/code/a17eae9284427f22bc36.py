"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 110  # Safety margin for 120s limit

# EVOLVE-BLOCK-START

def _compute_c3(h):
    """
    Compute the C3 upper bound for a given height vector h.
    Metric: max_k |conv(h, h)_k| / (sum_i |h_i|)^2
    """
    if np.all(h == 0):
        return 0.0
    conv = np.convolve(h, h, mode='full')
    peak = np.max(np.abs(conv))
    norm_sq = np.sum(np.abs(h)) ** 2
    if norm_sq == 0:
        return 0.0
    return peak / norm_sq

def _objective(x, h_scale):
    """
    Objective function for optimization.
    h = tanh(x) * h_scale
    We want to minimize max |conv(h,h)| / (sum |h|)^2.
    Since h_scale cancels out in the ratio (if we scale h, num scales by s^2, denom by s^2),
    we can fix h_scale or just optimize the shape.
    However, to avoid 0/0, we assume h is non-zero.
    Let's optimize x directly.
    Objective: log-sum-exp of (conv^2) / (sum |tanh(x)|)^2 ?
    Actually, the ratio is scale invariant.
    Let S = sum |h_i|. Ratio = Peak / S^2.
    If we define g = h / S, then sum |g_i| = 1.
    Ratio = max |conv(g, g)|.
    So we can constrain sum |h_i| = 1 and minimize max |conv(h, h)|.
    
    To handle the non-differentiability of |.| and max, we use smooth approximations.
    Smooth max: log-sum-exp of squares.
    Smooth abs: sqrt(x^2 + eps).
    
    Let's try minimizing log-sum-exp(conv^2) + penalty * (sum |h| - 1)^2.
    """
    h = np.tanh(x)
    # Ensure non-trivial
    abs_h = np.abs(h)
    S = np.sum(abs_h)
    if S < 1e-6:
        return 1e10
    
    # Normalize to sum |h| = 1
    g = h / S
    
    conv = np.convolve(g, g, mode='full')
    
    # Smooth max of conv^2
    # conv values are small since sum |g|=1. Max conv <= 1.
    # conv^2 <= 1.
    T = 5.0  # Temperature for log-sum-exp
    conv_sq = conv**2
    max_val = np.max(conv_sq)
    # log-sum-exp for numerical stability
    lse = T * (np.log(np.sum(np.exp((conv_sq - max_val) / T))) + max_val / T)
    
    # We want to minimize this.
    # Add a small penalty to encourage binary-like solutions? 
    # Actually, continuous might be better.
    # Let's just return lse.
    
    return lse

def _gradient_step(h, lr, iters=100):
    """
    Perform gradient descent steps on h directly.
    Objective: max |conv(h,h)| subject to sum |h| = 1.
    Using subgradient or smooth approximation.
    """
    h = h.copy()
    for _ in range(iters):
        # Compute gradient of max |conv|
        conv = np.convolve(h, h, mode='full')
        # Gradient of max |conv| w.r.t h
        # This is complex. Let's stick to scipy minimize.
        pass
    return h

def construct():
    start_time = time.time()
    best_h = np.ones(N)  # Initial seed
    best_score = _compute_c3(best_h)
    
    # Strategy 1: Random restarts with local optimization
    # We optimize x where h = tanh(x).
    # We fix sum |h| = 1 by normalization inside objective?
    # No, better to optimize h directly with a constraint?
    # scipy minimize doesn't handle sum |h|=1 well directly.
    # Let's use the ratio objective directly on h, but h can be scaled.
    # Ratio is scale invariant. So we can just minimize max |conv(h,h)| / (sum |h|)^2.
    
    def objective_ratio(x):
        h = np.tanh(x)
        S = np.sum(np.abs(h))
        if S < 1e-6:
            return 1e10
        conv = np.convolve(h, h, mode='full')
        peak = np.max(np.abs(conv))
        return peak / (S * S)
    
    # To get gradients, we need to differentiate this.
    # peak is non-differentiable.
    # Use log-sum-exp approximation for peak.
    # peak ~ log-sum-exp(|conv|).
    # But |conv| is also non-diff.
    # Use log-sum-exp(conv^2) as proxy for peak^2.
    # Then minimize log(peak^2) - 2*log(S).
    
    def objective_smooth(x):
        h = np.tanh(x)
        S = np.sum(np.abs(h))
        if S < 1e-6:
            return 1e10
        
        conv = np.convolve(h, h, mode='full')
        conv_sq = conv**2
        
        T = 10.0
        max_val = np.max(conv_sq)
        lse = T * (np.log(np.sum(np.exp((conv_sq - max_val) / T))) + max_val / T)
        
        # log(S^2) = 2 log S
        # We want to minimize lse / S^2.
        # Minimizing log(lse) - 2 log(S) is equivalent.
        # But lse is an approximation of peak^2.
        
        # Let's just minimize lse + penalty * (S - 1)^2 ?
        # If we fix S=1, we minimize lse.
        # Let's try to push S towards 1.
        penalty = 1.0 * (S - 1.0)**2
        
        return lse + penalty

    np.random.seed(42)
    
    for i in range(20):  # 20 restarts
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        # Random initialization
        x0 = np.random.uniform(-5, 5, N)
        
        try:
            res = minimize(objective_smooth, x0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
            
            h_opt = np.tanh(res.x)
            
            # Evaluate actual score
            score = _compute_c3(h_opt)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            continue

    # Refine best solution with more iterations
    if time.time() - start_time < TIME_LIMIT - 20:
        x_best = np.arctanh(np.clip(best_h, -0.9999, 0.9999))
        try:
            res = minimize(objective_smooth, x_best, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            h_refined = np.tanh(res.x)
            score = _compute_c3(h_refined)
            if score < best_score:
                best_score = score
                best_h = h_refined
        except Exception:
            pass

    # Try a different strategy: Random walk parameterization?
    # The directive suggests: "Formulate the signed step function as a sequence of partial sums of a random walk..."
    # Let's try a simple local search on the binary/continuous h directly.
    
    if time.time() - start_time < TIME_LIMIT - 5:
        # Simple local search: flip signs or adjust magnitudes
        # Since we allow continuous heights, let's try gradient-free local search.
        # Or just stick with the best found.
        
        # Let's try to enforce a "low variance" profile as suggested.
        # Low variance in h might reduce peak autoconvolution?
        # Actually, if h is constant, conv is triangular, peak is N.
        # If h is random +/-1, conv is roughly sqrt(N).
        # Random +/-1 is much better.
        
        # Let's generate a random +/-1 sequence and see its score.
        # Then try to improve it.
        
        # Check if best_h is already good.
        # Random +/-1 score:
        # conv peak ~ sqrt(N) * constant.
        # sum |h| = N.
        # score ~ sqrt(N) / N^2 = 1/N^1.5.
        # For N=400, 1/400^1.5 = 1/8000 = 0.000125.
        # Current best_score is likely higher if optimization failed.
        
        # Let's explicitly test a random binary sequence.
        h_rand = np.random.choice([-1, 1], N)
        score_rand = _compute_c3(h_rand)
        if score_rand < best_score:
            best_score = score_rand
            best_h = h_rand
        
        # Try to improve h_rand using the optimizer
        if time.time() - start_time < TIME_LIMIT - 10:
            x_rand = np.arctanh(np.clip(best_h, -0.9999, 0.9999))
            try:
                res = minimize(objective_smooth, x_rand, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
                h_opt = np.tanh(res.x)
                score = _compute_c3(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt
            except Exception:
                pass

    return best_h

# EVOLVE-BLOCK-END
