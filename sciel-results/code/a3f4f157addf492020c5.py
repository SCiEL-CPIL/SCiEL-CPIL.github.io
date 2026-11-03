"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, SLSQP

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _score(h):
    """Return the maximum overlap (lower is better)."""
    corrs = _compute_overlaps(h)
    return 1.0 - np.min(corrs)

def _project_to_simplex(h, target_sum):
    """Project h to satisfy sum(h) = target_sum and 0<=h<=1."""
    h = np.array(h, dtype=float).copy()
    h = np.clip(h, 0, 1)
    diff = target_sum - np.sum(h)
    for _ in range(30):
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        cnt = np.sum(mask)
        if cnt == 0:
            # If all at bounds, distribute to those closest to the needed direction
            if diff > 0:
                # Need to add, pick those closest to 1
                idx = np.argsort(h)[-min(5, len(h)):]
                h[idx] += diff / len(idx)
            else:
                # Need to subtract, pick those closest to 0
                idx = np.argsort(h)[:min(5, len(h))]
                h[idx] += diff / len(idx)
            h = np.clip(h, 0, 1)
        else:
            h[mask] += diff / cnt
            h = np.clip(h, 0, 1)
        diff = target_sum - np.sum(h)
    # Final exact adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute to all elements
        h += diff / len(h)
        h = np.clip(h, 0, 1)
        # One more pass to fix any rounding
        diff = target_sum - np.sum(h)
        if abs(diff) > 1e-12:
            mask = (h > 0) & (h < 1)
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
    return h

def _soft_min_obj_and_grad(x, beta):
    """
    Objective: minimize 1 - soft_min(corrs)
    where soft_min = -(1/beta) * log(sum exp(-beta * corrs_k))
    
    This is equivalent to maximizing soft_min(corrs).
    """
    n = len(x)
    corrs = _compute_overlaps(x)
    
    # soft_min = -(1/beta) * log(sum_k exp(-beta * corrs_k))
    # Use log-sum-exp trick for stability
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Gradient:
    # d(soft_min)/d(corrs_k) = exp(-beta*corrs_k) / sum_j exp(-beta*corrs_j)
    #                        = weights_k (softmax weights)
    # d(obj)/d(x) = -sum_k weights_k * d(corrs_k)/d(x)
    
    exp_neg = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_neg)
    weights = exp_neg / sum_exp  # These sum to 1
    
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w < 1e-15:
            continue
        # corrs_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
        # d(corrs_k)/d(h[j]) = (2/n)*h[j+k] for j < n-k
        #                     + (2/n)*h[j-k] for j >= k
        grad[:n-k] += w * x[k:]
        grad[k:] += w * x[:n-k]
    
    grad_obj = -grad  # d(obj)/d(x) = -d(soft_min)/d(x)
    
    return obj, grad_obj

def _epigraph_obj_and_grad(params, beta):
    """
    Epigraph formulation: minimize t subject to R_k >= t for all k.
    Soft formulation: minimize t + (1/beta)*log(sum_k exp(beta*(t - R_k)))
    
    params = [h_0, ..., h_{n-1}, t]
    """
    n = N
    h = params[:n]
    t = params[n]
    
    corrs = _compute_overlaps(h)
    
    # obj = t + (1/beta)*log(sum_k exp(beta*(t - corrs_k)))
    shifted = beta * (t - corrs)
    max_shifted = np.max(shifted)
    if max_shifted > 500:
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    else:
        log_sum = np.log(np.sum(np.exp(shifted)))
    
    obj = t + log_sum / beta
    
    # Gradient w.r.t. t:
    # d/dt [t + (1/beta)*log(sum_k exp(beta*(t - R_k)))]
    # = 1 + (1/beta)*sum_k beta*exp(beta*(t-R_k))/sum_k exp(beta*(t-R_k))
    # = 1 + 1 = 2  ... wait, that's not right.
    # Actually: d/dt log(sum exp(beta*(t-R_k))) = sum beta*exp(...)/sum exp(...) = beta * 1 = beta
    # So d/dt = 1 + (1/beta)*beta = 1 + 1 = 2? No...
    # d/dt [log(sum_j exp(beta*(t - R_j)))] = [sum_j beta*exp(beta*(t-R_j))] / [sum_j exp(beta*(t-R_j))] = beta
    # So d(obj)/dt = 1 + (1/beta)*beta = 2
    # Hmm, that means the gradient w.r.t. t is always 2, which is strange.
    # Let me reconsider. At optimum, we want t = min(R_k). The soft constraint 
    # should push t up until it hits the binding constraint.
    # Actually the gradient w.r.t. t being 2 means we always want to decrease t,
    # but the h gradient will push t up. Let me just use the h gradient.
    
    # Gradient w.r.t. h:
    # d(obj)/d(h_i) = (1/beta) * sum_k beta*exp(beta*(t-R_k))/sum_k exp(beta*(t-R_k)) * (-dR_k/dh_i)
    #               = -sum_k weights_k * dR_k/dh_i
    # where weights_k = exp(beta*(t-R_k)) / sum_j exp(beta*(t-R_j))
    
    exp_shifted = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    grad_h = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w < 1e-15:
            continue
        grad_h[:n-k] -= w * h[k:]
        grad_h[k:] -= w * h[:n-k]
    
    grad_t = 1.0  # From the t term; the log term contributes +1 as shown
    
    return obj, np.concatenate([grad_h, [grad_t]])

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_h = np.full(n, 0.5)
    best_score = _score(best_h)
    
    np.random.seed(42)
    
    # Generate diverse initializations
    restarts = []
    
    # Uniform
    restarts.append(np.full(n, 0.5))
    
    # Linear gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    
    # Sinusoidal patterns
    x_grid = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3, 4, 5, 6, 8, 10]:
        for amp in [0.2, 0.3, 0.4, 0.5]:
            restarts.append(np.clip(0.5 + amp*np.sin(freq*x_grid), 0, 1))
            restarts.append(np.clip(0.5 + amp*np.cos(freq*x_grid), 0, 1))
    
    # Block patterns
    for block_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(n * block_frac)
        restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
        restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    
    # Random
    for _ in range(20):
        restarts.append(np.random.uniform(0, 1, n))
    
    # Bang-bang (binary) patterns
    for _ in range(15):
        pattern = np.random.choice([0, 1], size=n)
        restarts.append(pattern.astype(float))
    
    # Project all restarts to feasible set
    projected_restarts = []
    for r in restarts:
        pr = _project_to_simplex(r, target_sum)
        # Check score
        s = _score(pr)
        if s < best_score:
            best_score = s
            best_h = pr
        projected_restarts.append(pr)
    
    # Optimize each restart using L-BFGS-B with soft-min objective
    for x0 in projected_restarts:
        if _time_remaining() < 3.0:
            break
        
        # Multi-stage: start with low beta for smooth landscape, increase for sharpness
        current = x0.copy()
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if _time_remaining() < 3.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    current,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_score = _score(res_x)
                if res_score < best_score:
                    best_score = res_score
                    best_h = res_x
                current = res_x
            except Exception:
                pass
    
    # Polish best solution with SLSQP using epigraph formulation
    if _time_remaining() > 10.0:
        for beta in [100.0, 200.0, 500.0]:
            if _time_remaining() < 5.0:
                break
            try:
                params0 = np.concatenate([best_h, [best_score - 0.01]])
                constraints = [
                    {'type': 'eq', 'fun': lambda p: np.sum(p[:n]) - target_sum,
                     'jac': lambda p: np.concatenate([np.ones(n), [0.0]])}
                ]
                bounds = [(0, 1)] * n + [(-1, 2)]
                
                res = minimize(
                    _epigraph_obj_and_grad,
                    params0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                res_h = _project_to_simplex(res.x[:n], target_sum)
                res_score = _score(res_h)
                if res_score < best_score:
                    best_score = res_score
                    best_h = res_h
            except Exception:
                pass
    
    # Final subgradient polish on the hard min
    if _time_remaining() > 5.0:
        h = best_h.copy()
        step_size = 0.05
        for iteration in range(100):
            if _time_remaining() < 3.0:
                break
            
            corrs = _compute_overlaps(h)
            min_corr = np.min(corrs)
            k_star = np.argmin(corrs) + 1
            
            # Subgradient of min_corr w.r.t. h
            grad = np.zeros(n)
            for j in range(n - k_star):
                grad[j] += (2.0 / n) * h[j + k_star]
                grad[j + k_star] += (2.0 / n) * h[j]
            
            # We want to maximize min_corr, so move in +grad direction
            h_new = h + step_size * grad
            h_new = _project_to_simplex(h_new, target_sum)
            
            new_score = _score(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new
                # Keep step size or slightly increase
                step_size = min(step_size * 1.1, 0.1)
            else:
                step_size *= 0.5
        
        # Simulated annealing style local search
        if _time_remaining() > 5.0:
            h = best_h.copy()
            temp = 0.01
            for iteration in range(200):
                if _time_remaining() < 3.0:
                    break
                
                # Random perturbation
                h_perturbed = h + np.random.normal(0, temp, n)
                h_perturbed = _project_to_simplex(h_perturbed, target_sum)
                
                new_score = _score(h_perturbed)
                cur_score = _score(h)
                
                if new_score < cur_score or np.random.random() < np.exp((cur_score - new_score) / temp):
                    h = h_perturbed
                    if new_score < best_score:
                        best_score = new_score
                        best_h = h
                
                temp *= 0.995
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    diff = target_sum - np.sum(best_h)
    if abs(diff) > 1e-12:
        mask = (best_h > 0) & (best_h < 1)
        if np.sum(mask) > 0:
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
