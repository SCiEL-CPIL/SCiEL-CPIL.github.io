import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def smooth_max(conv, beta=100.0):
    """Compute smooth approximation of max using log-sum-exp."""
    max_c = np.max(conv)
    if max_c < 1e-15:
        return 0.0
    shifted = conv - max_c
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    smooth_max_val = max_c + np.log(sum_exp) / beta
    return smooth_max_val

def ratio_smooth(h, beta=100.0):
    """Compute smooth ratio for optimization."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    smax = smooth_max(conv, beta)
    if smax < 1e-15:
        return 0.0
    return h2_sq / smax

def gradient_ratio_smooth(h, beta=100.0):
    """
    Compute gradient of the smooth ratio.
    Ratio R = S / M, where S = sum(h^2) and M = smooth_max(conv).
    dR/dh = (dS/dh * M - S * dM/dh) / M^2
    """
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return np.zeros_like(h), 0.0
    
    h_norm = h / l1
    S = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    max_c = np.max(conv)
    if max_c < 1e-15:
        return np.zeros_like(h), 0.0
    
    shifted = conv - max_c
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    M = max_c + np.log(sum_exp) / beta
    weights = exp_shifted / sum_exp
    
    # Gradient of S w.r.t h
    # S = sum((h/l1)^2) = (1/l1^2) sum(h^2)
    # dS/dh_i = (1/l1^2) * 2 h_i
    dS = (2.0 * h_norm**2) / l1  # Wait, h_norm = h/l1. d(h_norm^2)/dh = 2 h_norm * (1/l1) = 2 h_i / l1^2.
    # Let's re-derive carefully.
    # Let u = h/l1. S = sum(u^2).
    # du_i/dh_j = (delta_ij l1 - h_i) / l1^2.
    # dS/du_i = 2 u_i.
    # dS/dh_j = sum_i 2 u_i * (delta_ij l1 - h_i) / l1^2
    #           = (2 u_j l1 - 2 sum_i u_i h_i) / l1^2
    #           = (2 h_j - 2 l1 sum_i (h_i/l1) h_i) / l1^2
    #           = (2 h_j - 2 S) / l1
    
    # Gradient of M w.r.t h
    # M = max_c + (1/beta) log(sum_exp)
    # dM/dh_j = sum_k weights[k] * d(conv[k])/dh_j
    # conv[k] = sum_i h_norm[i] h_norm[k-i]
    # d(conv[k])/dh_j = d(sum_i h_i/l1 * h_{k-i}/l1)/dh_j
    # = (1/l1^2) * [ (delta_jk l1 - h_j) h_{k-j} + h_j (delta_{j,k-j} l1 - h_{k-j}) ] ... this is messy.
    # Simpler: use finite difference or numerical gradient if analytic is too complex/risky.
    # Given N=50, numerical gradient is fast enough for L-BFGS-B?
    # L-BFGS-B can use numerical gradients. Let's use that to avoid bugs.
    
    # We will return 0 and let scipy use numerical gradient.
    return np.zeros_like(h), S / M

def optimize_on_support(support, h_init=None, beta=100.0, max_iter=200):
    """
    Optimize weights on the given support indices.
    """
    m = len(support)
    
    def objective_and_grad(x):
        # x are the weights on the support
        x = np.maximum(x, 0.0)
        l1 = np.sum(x)
        if l1 < 1e-15:
            return 0.0, np.zeros(m)
        
        h = np.zeros(N)
        h[support] = x / l1
        
        # Calculate ratio
        h2_sq = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(m)
        
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        M = max_c + np.log(sum_exp) / beta
        weights_conv = exp_shifted / sum_exp
        
        ratio = h2_sq / M
        
        # Gradient of ratio w.r.t x (weights on support)
        # We want d(ratio)/dx_j.
        # Let's use the chain rule.
        # h[support] = x / l1.
        # dh_i/dx_j = (delta_ij l1 - h_i) / l1^2 for i in support.
        
        # d(ratio)/dh_i = (dS/dh_i * M - S * dM/dh_i) / M^2
        # S = h2_sq.
        # dS/dh_i = 2 h_i.
        # dM/dh_i = sum_k weights_conv[k] * d(conv[k])/dh_i
        # d(conv[k])/dh_i = h[k-i] + h[i-k] (with bounds 0..N-1 for indices)
        # Note: h is zero outside support, but we treat h as the full vector.
        
        dM = np.zeros(N)
        for i in range(N):
            s = 0.0
            # conv[k] = sum_p h[p] h[k-p]
            # d(conv[k])/dh[i] = h[k-i] + h[i-k] if indices valid
            # We can compute this efficiently by correlation of weights_conv with h
            # dM/dh[i] = sum_k weights_conv[k] (h[k-i] + h[i-k])
            # This is 2 * (weights_conv * h) at lag i?
            # Let's check: sum_k w[k] h[k-i] is correlation.
            # sum_k w[k] h[i-k] is correlation with flipped w.
            # Since h is real, h[i-k] = h[-(k-i)].
            # So dM/dh[i] = 2 * sum_k w[k] h[k-i] (if w is symmetric? No, w is not symmetric).
            # Actually, conv[k] = sum_p h[p] h[k-p].
            # d(conv[k])/dh[i] = h[k-i] + h[i-k].
            # sum_k w[k] h[k-i] is (w * h)[i] (convolution).
            # sum_k w[k] h[i-k] is (w_rev * h)[i] where w_rev[k] = w[-k].
            # So dM/dh[i] = (w * h)[i] + (w_rev * h)[i].
            
            # Let's just compute it directly for N=50, it's fast enough.
            for k in range(2 * N - 1):
                if weights_conv[k] > 1e-15:
                    ki = k - i
                    if 0 <= ki < N:
                        s += weights_conv[k] * h[ki]
                    ik = i - k
                    if 0 <= ik < N:
                        s += weights_conv[k] * h[ik]
            dM[i] = s

        dS = 2.0 * h
        dratio_dh = (dS * M - h2_sq * dM) / (M * M)
        
        # Now map back to x
        # h[support] = x / l1
        # dh_i/dx_j = (delta_ij l1 - h_i) / l1^2
        # dratio/dx_j = sum_{i in support} dratio_dh[i] * (delta_ij l1 - h_i) / l1^2
        #              = (dratio_dh[j] * l1 - sum_{i in support} dratio_dh[i] h_i) / l1^2
        
        sum_term = np.sum(dratio_dh[support] * h[support])
        grad_x = (dratio_dh[support] * l1 - sum_term) / (l1 * l1)
        
        # We want to maximize ratio, so minimize -ratio
        return -ratio, -grad_x

    # Initial guess: equal weights
    if h_init is None:
        x0 = np.ones(m) / m
    else:
        x0 = h_init[support].copy()
        l1 = np.sum(x0)
        if l1 > 0:
            x0 = x0 / l1
        else:
            x0 = np.ones(m) / m
            
    # Bounds
    bounds = [(0.0, 1.0) for _ in range(m)]
    
    # Constraint: sum x = 1 is handled by normalization in objective, 
    # but optimizer might drift. 
    # Better to use equality constraint or just let it optimize on simplex.
    # L-BFGS-B doesn't handle equality constraints well.
    # We can parameterize x = exp(y) / sum(exp(y)) to stay on simplex?
    # Or just use SLSQP which handles constraints.
    
    # Let's use SLSQP for better constraint handling
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    
    try:
        res = minimize(objective_and_grad, x0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-12})
        if res.success:
            x_opt = res.x
            x_opt = np.maximum(x_opt, 0.0)
            x_opt = x_opt / np.sum(x_opt)
            h_opt = np.zeros(N)
            h_opt[support] = x_opt
            return h_opt, -res.fun
        else:
            # Return initial if failed
            h_opt = np.zeros(N)
            h_opt[support] = x0
            return h_opt, compute_ratio(h_opt)
    except Exception:
        h_opt = np.zeros(N)
        h_opt[support] = x0
        return h_opt, compute_ratio(h_opt)

def generate_sidon_sets():
    """
    Generate several Sidon sets of size m within [0, N).
    """
    sets = []
    # Greedy generation for different sizes
    for m in [8, 10, 12, 15, 18]:
        support = []
        used_sums = set()
        for i in range(m):
            found = False
            for cand in range(0, N):
                if cand in support:
                    continue
                # Check sums with existing elements
                valid = True
                new_sums = []
                for j in range(len(support)):
                    s = cand + support[j]
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.append(s)
                # Check sums with itself (cand+cand)
                if valid:
                    s_self = 2 * cand
                    if s_self in used_sums:
                        valid = False
                    else:
                        new_sums.append(s_self)
                
                if valid:
                    support.append(cand)
                    used_sums.update(new_sums)
                    found = True
                    break
            if not found:
                break
        
        if len(support) == m:
            sets.append(support)
    
    # Add some random sparse sets
    np.random.seed(42)
    for _ in range(10):
        m = np.random.randint(8, 20)
        support = np.random.choice(N, m, replace=False)
        sets.append(sorted(support))
        
    return sets

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    sidon_sets = generate_sidon_sets()
    
    # Optimize on each Sidon set
    for support in sidon_sets:
        if time.time() > deadline - 10.0:
            break
        
        # Try a few different betas for smooth max
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 5.0:
                break
            
            h_opt, sc = optimize_on_support(support, beta=beta, max_iter=100)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
                
            # Polish with a local search on the full vector?
            # Maybe just do a few random perturbations on the support
            if time.time() > deadline - 5.0:
                break
            
            # Try a few random restarts on the same support
            for _ in range(3):
                if time.time() > deadline - 5.0:
                    break
                # Random initial weights
                h_init = np.zeros(N)
                h_init[support] = np.random.rand(len(support))
                h_init = h_init / np.sum(h_init)
                
                h_opt2, sc2 = optimize_on_support(support, h_init=h_init, beta=beta, max_iter=50)
                if sc2 > best_score:
                    best_score = sc2
                    best_h = h_opt2.copy()
    
    # Final polish: try a dense random initialization and optimize
    # This is less likely to be better than Sidon, but covers the base case
    if time.time() > deadline - 10.0:
        return list(best_h)
    
    # Try a few dense random starts
    for _ in range(5):
        if time.time() > deadline - 2.0:
            break
        h_rand = np.random.rand(N)
        h_rand = h_rand / np.sum(h_rand)
        # Use a simple coordinate descent or gradient step
        # Let's just compute ratio and keep best
        sc = compute_ratio(h_rand)
        if sc > best_score:
            best_score = sc
            best_h = h_rand.copy()
            
    return list(best_h)
