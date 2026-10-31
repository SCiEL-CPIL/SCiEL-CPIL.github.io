"""
Strategy: Continuous optimization via SLSQP on a smoothed max-overlap objective.
Instead of discrete hill-climbing, we treat the problem as a continuous non-linear
programming problem. We approximate the non-differentiable max() using a log-sum-exp
function and compute analytical gradients to guide the SLSQP solver.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _log_sum_exp(x, beta=10.0):
    """
    Approximate max(x) using log-sum-exp.
    As beta -> infinity, this approaches max(x).
    x: array of shape (N,)
    """
    # Shift for numerical stability
    x_max = np.max(x)
    # If x is very large, x_max might cause overflow in exp, but we subtract x_max
    # exp(beta * (x - x_max)) is <= 1
    # log(beta * sum(exp(...))) / beta
    # Wait, standard log-sum-exp is log(sum(exp(x))).
    # We want (1/beta) * log(sum(exp(beta * x)))
    # = (1/beta) * (beta * x_max + log(sum(exp(beta * (x - x_max)))))
    # = x_max + (1/beta) * log(sum(exp(beta * (x - x_max))))
    
    # Handle case where x is empty or constant
    if len(x) == 0:
        return 0.0, np.zeros_like(x)
        
    x_shifted = beta * (x - x_max)
    # Clip to avoid underflow/overflow issues in exp, though x_shifted <= 0
    # exp(x_shifted) is in (0, 1]
    e = np.exp(x_shifted)
    s = np.sum(e)
    if s < 1e-300:
        return x_max, np.zeros_like(x)
    
    val = x_max + np.log(s) / beta
    
    # Gradient: d/dx_i [ x_max + (1/beta) log(sum_j exp(beta(x_j - x_max))) ]
    # Note: x_max depends on x, but for smooth approximation we often ignore the derivative of x_max
    # or assume x_max is fixed locally. A more rigorous gradient includes the max term,
    # but for optimization, the soft gradient is usually sufficient and smoother.
    # Let's use the standard soft gradient:
    # grad_i = (1/beta) * ( beta * e_i / s ) = e_i / s
    # This sums to 1.
    grad = e / s
    
    return val, grad

def _objective_and_grad(h, beta=10.0):
    """
    Compute the smoothed max overlap and its gradient.
    h: array of shape (N,)
    """
    N = len(h)
    inv_h = 1.0 - h
    
    # Compute all N shifts of the overlap
    # corr[s] = sum_i h[i] * inv_h[(i+s)%N]
    # Using FFT is faster for the values, but for gradient we need the structure.
    # Actually, for SLSQP, we need the gradient with respect to h.
    # Let's compute the values first using FFT for speed, then compute gradient analytically.
    
    H = np.fft.fft(h)
    IH = np.fft.fft(inv_h)
    # corr[s] = sum_k h[k] * inv_h[(k+s)%N]
    # ifft(fft(h) * conj(fft(inv_h))) gives corr[s] = sum_k h[k] * inv_h[(k+s)%N] ?
    # Let's verify:
    # ifft(fft(a) * conj(fft(b)))[s] = sum_k a[k] b[(k+s)%N]
    # So yes, corr[s] = sum_k h[k] (1-h[(k+s)%N])
    corr = np.real(np.fft.ifft(H * np.conj(IH)))
    
    # The objective is max(corr)
    val, grad_corr = _log_sum_exp(corr, beta)
    
    # Now we need grad_h of val.
    # val = LSE(corr)
    # grad_h val = sum_s (grad_corr[s] * grad_h corr[s])
    
    # corr[s] = sum_k h[k] * (1 - h[(k+s)%N])
    # Let A_{s,k} = h[k] * (1 - h[(k+s)%N])
    # d(corr[s])/d(h[j]) = 
    #   if k=j: (1 - h[(j+s)%N])
    #   if (k+s)%N = j => k = (j-s)%N: h[(j-s)%N] * (-1)
    #   else 0
    
    # So d(corr[s])/d(h[j]) = (1 - h[(j+s)%N]) - h[(j-s)%N]
    
    # grad_h val[j] = sum_s grad_corr[s] * ((1 - h[(j+s)%N]) - h[(j-s)%N])
    
    # We can compute this efficiently.
    # term1[j] = sum_s grad_corr[s] * (1 - h[(j+s)%N])
    # term2[j] = sum_s grad_corr[s] * h[(j-s)%N]
    
    # term1[j] = sum_s grad_corr[s] - sum_s grad_corr[s] * h[(j+s)%N]
    # sum_s grad_corr[s] = 1.0 (from LSE gradient property)
    # So term1[j] = 1.0 - sum_s grad_corr[s] * h[(j+s)%N]
    
    # Let g = grad_corr
    # sum_s g[s] * h[(j+s)%N] is a circular cross-correlation of g and h?
    # corr_gh[j] = sum_s g[s] h[(j+s)%N]
    # This is ifft(fft(g) * conj(fft(h)))?
    # ifft(fft(a)*conj(fft(b)))[s] = sum_k a[k] b[(k+s)%N]
    # So if a=g, b=h, result[j] = sum_k g[k] h[(k+j)%N]. Yes.
    
    G = np.fft.fft(grad_corr)
    H_fft = np.fft.fft(h)
    corr_gh = np.real(np.fft.ifft(G * np.conj(H_fft)))
    
    # term1[j] = 1.0 - corr_gh[j]
    
    # term2[j] = sum_s g[s] * h[(j-s)%N]
    # Let k = j-s => s = j-k.
    # sum_k g[j-k] h[k]
    # This is circular convolution of g and h?
    # conv[j] = sum_k g[k] h[j-k].
    # Here we have g[j-k] h[k]. Let l = j-k => k = j-l.
    # sum_l g[l] h[j-l]. This is exactly circular convolution of g and h.
    # Convolution via FFT: ifft(fft(g) * fft(h))
    
    corr_conv = np.real(np.fft.ifft(G * H_fft))
    
    # grad_h val[j] = term1[j] - term2[j]
    # = (1.0 - corr_gh[j]) - corr_conv[j]
    
    grad_h = 1.0 - corr_gh - corr_conv
    
    return val, grad_h

def _score(h):
    """Exact score for final evaluation."""
    s = h.sum()
    if abs(s - N / 2.0) > 1e-8:
        return float('inf')
    if h.min() < -1e-10 or h.max() > 1.0 + 1e-10:
        return float('inf')
    
    inv_h = 1.0 - h
    H = np.fft.fft(h)
    IH = np.fft.fft(inv_h)
    corr = np.real(np.fft.ifft(H * np.conj(IH)))
    return float(np.max(corr))

def _normalize_to_sum(h, target=N/2.0):
    """Adjust h to have sum exactly target, staying in [0,1]."""
    h = h.copy()
    current_sum = h.sum()
    diff = target - current_sum
    
    if abs(diff) < 1e-15:
        return h
    
    # Iteratively adjust
    for _ in range(100):
        if abs(diff) < 1e-15:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        diff = target - h.sum()
    
    # Final exact fix
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        if diff > 0:
            idxs = np.where(h < 1.0)[0]
            if len(idxs) > 0:
                room = 1.0 - h[idxs]
                order = np.argsort(-room)
                for i in order:
                    add = min(diff, room[i])
                    h[idxs[i]] += add
                    diff -= add
                    if abs(diff) < 1e-15:
                        break
        else:
            idxs = np.where(h > 0.0)[0]
            if len(idxs) > 0:
                order = np.argsort(h[idxs])
                for i in order:
                    sub = min(-diff, h[idxs[i]])
                    h[idxs[i]] -= sub
                    diff += sub
                    if abs(diff) < 1e-15:
                        break
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    # Initializations:
    # 1. Constant 0.5
    # 2. Random
    # 3. Sine patterns
    inits = []
    inits.append(np.full(N, 0.5))
    
    for _ in range(5):
        inits.append(rng.random(N))
        
    for freq in [1, 2, 5, 10]:
        x = np.arange(N)
        inits.append(0.5 + 0.4 * np.sin(2 * np.pi * freq * x / N))
        inits.append(0.5 + 0.4 * np.cos(2 * np.pi * freq * x / N))
        
    # Block patterns
    for block_len in [2, 5, 10]:
        pat = np.zeros(N)
        for i in range(N):
            if (i // block_len) % 2 == 0:
                pat[i] = 1.0
            else:
                pat[i] = 0.0
        inits.append(pat)
        
    # For each initialization, run SLSQP
    # We will use a schedule of beta values to anneal the approximation
    # Start with low beta (smoother) and increase to high beta (sharper)
    
    for init in inits:
        if time.time() > deadline:
            break
            
        h_curr = _normalize_to_sum(init)
        
        # Annealing schedule
        betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            # Define objective and gradient for current beta
            def obj(x):
                val, _ = _objective_and_grad(x, beta)
                return val
            
            def grad(x):
                _, g = _objective_and_grad(x, beta)
                return g
                
            # Constraints: sum(x) = N/2
            # Bounds: [0, 1] for all x
            
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0, 
                     'jac': lambda x: np.ones(N)})
            
            bounds = [(0.0, 1.0) for _ in range(N)]
            
            try:
                res = minimize(obj, h_curr, jac=grad, method='SLSQP', 
                               bounds=bounds, constraints=cons,
                               options={'maxiter': 200, 'ftol': 1e-8, 'disp': False})
                
                if res.success or res.fun < best_score:
                    h_candidate = res.x
                    # Ensure constraints are met strictly
                    h_candidate = _normalize_to_sum(h_candidate)
                    sc = _score(h_candidate)
                    if sc < best_score:
                        best_score = sc
                        best_h = h_candidate.copy()
                    h_curr = h_candidate.copy()
            except Exception:
                pass

    # Final polish with direct hill climbing if time permits
    if time.time() < deadline - 10:
        h = best_h.copy()
        no_improve = 0
        max_no_improve = 100
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i == j:
                no_improve += 1
                continue
            eps = rng.uniform(0.001, 0.01)
            if h[i] + eps > 1.0 or h[j] - eps < 0.0:
                no_improve += 1
                continue
            h_new = h.copy()
            h_new[i] += eps
            h_new[j] -= eps
            sc = _score(h_new)
            if sc < best_score:
                best_score = sc
                best_h = h_new.copy()
                h = h_new.copy()
                no_improve = 0
            else:
                no_improve += 1

    best_h = _normalize_to_sum(best_h)
    final_sc = _score(best_h)
    if np.isinf(final_sc):
        return np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
