"""Combined program: Kronecker/quadratic-irrational construction + soft-min correlation optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 138.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _overlap_max(h):
    """Compute max over shifts k of sum_i h[i]*(1-h[i+k]), circular."""
    n = len(h)
    max_val = 0.0
    for k in range(n):
        shifted = np.roll(h, -k)
        val = np.sum(h * (1.0 - shifted))
        if val > max_val:
            max_val = val
    return max_val

def _cross_corrs(h):
    """Compute all non-trivial circular cross-correlations R_k = sum h[i]*h[i+k]."""
    n = len(h)
    corrs = np.zeros(n)
    for k in range(1, n):
        shifted = np.roll(h, -k)
        corrs[k] = np.sum(h * shifted)
    return corrs

def _soft_min_corr(h, beta=100.0):
    """Soft min of cross-correlations using log-sum-exp."""
    corrs = _cross_corrs(h)
    # Only consider non-trivial shifts (k=1..n-1)
    corrs_nontriv = corrs[1:]
    min_val = np.min(corrs_nontriv)
    shifted = corrs_nontriv - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    return soft_min, corrs_nontriv

def _objective_and_grad_soft(x, beta=100.0):
    """Minimize 1 - soft_min_corr, approximating max overlap."""
    h = x
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        shifted = np.roll(h, -k)
        corrs[k-1] = np.sum(h * shifted)
    
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Gradient: d(soft_min)/d(corrs_k) = exp(-beta*(corrs_k - min_val)) / (beta * sum_exp)
    # Actually using the cleaner form: soft_min = -(1/beta)*log(sum exp(-beta*corrs_k))
    # d(soft_min)/d(corrs_k) = exp(-beta*corrs_k) / sum_j exp(-beta*corrs_j)
    exp_neg = np.exp(-beta * corrs)
    sum_exp_neg = np.sum(exp_neg)
    weights = exp_neg / sum_exp_neg  # softmax weights (inverted)
    
    # d(soft_min)/d(h) = sum_k weights_k * d(corrs_k)/d(h)
    # corrs_k = sum_i h[i]*h[i+k] (circular)
    # d/dh[j] corrs_k = h[j+k] + h[j-k] (circular)
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1]
        if w != 0:
            shifted_plus = np.roll(h, -k)   # h[i+k]
            shifted_minus = np.roll(h, k)    # h[i-k]
            grad += w * (shifted_plus + shifted_minus)
    
    # d(obj)/d(h) = -d(soft_min)/d(h)
    grad_obj = -grad
    return obj, grad_obj

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    v = np.clip(v, 0.0, 1.0)
    for _ in range(50):
        diff = target_sum - np.sum(v)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            free = v < 1.0 - 1e-10
        else:
            free = v > 1e-10
        if not np.any(free):
            break
        v[free] += diff / np.sum(free)
        v = np.clip(v, 0.0, 1.0)
    return v

def _kronecker_h(alpha, beta, target_sum):
    """
    Construct h[i] = 1 if {i*alpha + beta} < threshold, else 0,
    then adjust threshold so sum = target_sum.
    This gives a binary (0/1) step function with equidistribution properties.
    """
    n = len(target_sum) if hasattr(target_sum, '__len__') else 0
    # Actually target_sum is a scalar here
    t = target_sum
    n = N
    # For a given alpha, beta, the set {i*alpha + beta mod 1 : i=0..n-1} is equidistributed
    # We want sum h = n/2, so threshold should be 0.5 for large n
    # But we can also use non-binary: h[i] = clip(2*({i*alpha+beta} - 0.5), 0, 1) or similar
    # Let's try binary first, then smooth
    
    # Binary Kronecker: h[i] = 1 if frac(i*alpha + beta) < 0.5
    indices = np.arange(n)
    vals = np.mod(indices * alpha + beta, 1.0)
    h = (vals < 0.5).astype(float)
    h = _project_to_simplex(h, t)
    return h

def _smooth_kronecker_h(alpha, beta, width, target_sum):
    """Smoothed Kronecker: h[i] = sigmoid((0.5 - frac(i*alpha+beta))/width) or similar."""
    n = N
    t = target_sum
    indices = np.arange(n)
    vals = np.mod(indices * alpha + beta, 1.0)
    # Map to [0,1] with a smooth step around 0.5
    # Use a smooth approximation: h = 0.5 + 0.5 * tanh((0.5 - vals) / width)
    # But this doesn't respect the [0,1] bounds well. Let's use a clipped linear.
    # Actually, let's use: h[i] = clip((0.5 - vals) / width + 0.5, 0, 1)
    # This gives a triangular bump
    h = np.clip((0.5 - vals) / width + 0.5, 0.0, 1.0)
    h = _project_to_simplex(h, t)
    return h

def construct():
    deadline = time.time() + 140.0
    target_sum = N / 2.0
    
    best = np.full(N, 0.5)
    best_score = _overlap_max(best)
    
    rng = np.random.RandomState(42)
    
    # Strategy 1: Kronecker/quadratic-irrational constructions
    # Try various quadratic irrationals: sqrt(2)-1, (1+sqrt(5))/2 - 1, etc.
    quadratic_irrationals = [
        np.sqrt(2) - 1,
        (1 + np.sqrt(5)) / 2 - 1,  # golden ratio conjugate
        np.sqrt(3) - 1,
        np.sqrt(5) - 2,
        (np.sqrt(2) + 1) / 2 - 0.5,
        np.sqrt(13) - 3,
        np.sqrt(17) - 4,
        1/np.sqrt(2) - 0.5,
        np.sqrt(2)/2 - 0.3,
        (3 + np.sqrt(5)) / 2 - 2,
    ]
    
    # Add some random irrationals
    for _ in range(10):
        quadratic_irrationals.append(rng.uniform(0.1, 0.9))
    
    for alpha in quadratic_irrationals:
        if time.time() > deadline - 15.0:
            break
        
        # Try several beta offsets
        for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            if time.time() > deadline - 15.0:
                break
            
            # Binary Kronecker
            h_kron = _kronecker_h(alpha, beta, target_sum)
            score = _overlap_max(h_kron)
            if score < best_score - 1e-10:
                best = h_kron
                best_score = score
            
            # Smoothed Kronecker with various widths
            for width in [0.05, 0.1, 0.15, 0.2, 0.3]:
                if time.time() > deadline - 15.0:
                    break
                h_smooth = _smooth_kronecker_h(alpha, beta, width, target_sum)
                score = _overlap_max(h_smooth)
                if score < best_score - 1e-10:
                    best = h_smooth
                    best_score = score
    
    # Strategy 2: Optimize from best Kronecker start using soft-min objective
    if time.time() < deadline - 20.0:
        # Use the best Kronecker result as starting point
        x0 = best.copy()
        
        try:
            res = minimize(
                _objective_and_grad_soft,
                x0,
                args=(50.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = _project_to_simplex(res.x, target_sum)
            score = _overlap_max(h_opt)
            if score < best_score - 1e-10:
                best = h_opt
                best_score = score
        except Exception:
            pass
    
    # Strategy 3: Polish with higher beta
    if time.time() < deadline - 10.0:
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(
                    _objective_and_grad_soft,
                    best,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_opt = _project_to_simplex(res.x, target_sum)
                score = _overlap_max(h_opt)
                if score < best_score - 1e-10:
                    best = h_opt
                    best_score = score
            except Exception:
                pass
    
    # Strategy 4: Random restarts with soft-min optimization
    for trial in range(5):
        if time.time() > deadline - 10.0:
            break
        
        h0 = rng.uniform(0.2, 0.8, size=N)
        h0 = _project_to_simplex(h0, target_sum)
        
        try:
            res = minimize(
                _objective_and_grad_soft,
                h0,
                args=(50.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = _project_to_simplex(res.x, target_sum)
            score = _overlap_max(h_opt)
            if score < best_score - 1e-10:
                best = h_opt
                best_score = score
        except Exception:
            pass
    
    # Final projection
    best = _project_to_simplex(best, target_sum)
    return best
