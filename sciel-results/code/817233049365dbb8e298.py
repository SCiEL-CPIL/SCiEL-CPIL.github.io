import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    """Compute the ratio: (L2 norm squared) / (L1 norm * sup norm of autoconvolution)
    for a vector of nonnegative heights."""
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

def is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set: all pairwise sums are distinct."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def generate_sidon_sets():
    """Generate several Sidon sets of various sizes within [0, N)."""
    sidon_sets = []
    base_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 9, 15, 21],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 4],
        [0, 1, 3],
    ]
    
    for s in base_sets:
        if all(0 <= x < N for x in s) and is_sidon_set(s):
            sidon_sets.append(list(s))
    
    np.random.seed(123)
    for _ in range(5000):
        k = np.random.randint(4, 11)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            found = False
            for existing in sidon_sets:
                if len(existing) == k and set(existing) == set(indices):
                    found = True
                    break
            if not found:
                sidon_sets.append(indices)
                if len(sidon_sets) > 50:
                    break
    
    return sidon_sets

def optimize_weights_for_support(support, beta=100.0, max_iter=2000, tol=1e-10):
    """
    Optimize weights h_i for i in support to minimize max convolution coefficient.
    Uses smooth max surrogate: (1/beta) * log(sum(exp(beta * conv_k)))
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initial guess: uniform weights
    x0 = np.ones(k) / k
    
    def objective(x):
        # x is the vector of weights for the support
        h = np.zeros(N)
        h[support] = x
        # Normalize just in case, though constraints should handle it
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        conv = fftconvolve(h, h, mode='full')
        # Smooth max
        # Use log-sum-exp trick for stability
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0
        shifted = conv - max_c
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted)))
        return log_sum_exp / beta

    def jac(x):
        h = np.zeros(N)
        h[support] = x
        s = np.sum(h)
        if s < 1e-15:
            return np.zeros(k)
        h = h / s
        conv = fftconvolve(h, h, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return np.zeros(k)
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        # Gradient of log_sum_exp w.r.t h
        # d/dh_j (1/beta log sum exp(beta conv_k)) = sum_k weights_k * beta * d conv_k / d h_j / beta
        # = sum_k weights_k * d conv_k / d h_j
        # conv_k = sum_i h_i h_{k-i}
        # d conv_k / d h_j = h_{k-j} + h_{j-k} (if indices valid)
        # Since h is non-zero only on support, we can compute this efficiently.
        
        # grad_h_j = 2 * sum_k weights_k * h_{k-j}
        # Let's compute it using FFT for speed if N is large, but N=50 is small.
        
        # grad_h[j] = 2 * sum_{m in support} h[m] * weights[m+j]
        
        grad_h = np.zeros(N)
        for m in support:
            hm = h[m]
            if hm > 1e-15:
                # weights indices m+j must be in [0, 2N-2]
                # j in [0, N-1], m in [0, N-1] => m+j in [0, 2N-2]
                # weights has length 2N-1
                # We need to add 2 * hm * weights[m+j] to grad_h[j] for all j
                # This is a convolution-like operation
                # Let's just do it with a loop for the support indices
                for j in support:
                    idx = m + j
                    if 0 <= idx < len(weights):
                        grad_h[j] += 2 * hm * weights[idx]
        
        # Now we need the gradient w.r.t x
        # x is the vector of weights for the support
        # h = x (before normalization)
        # The objective is f(x) = g(h(x)) where h(x) = x / sum(x)
        # So we need the chain rule
        # Actually, let's just compute the gradient numerically or use a simpler approach
        
        # For simplicity, let's use numerical gradient
        grad_x = np.zeros(k)
        eps = 1e-7
        f0 = objective(x)
        for i in range(k):
            x_plus = x.copy()
            x_plus[i] += eps
            grad_x[i] = (objective(x_plus) - f0) / eps
        
        return grad_x

    # Use Nelder-Mead for simplicity and robustness
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
        x_opt = np.maximum(res.x, 0.0)
        s = np.sum(x_opt)
        if s > 1e-15:
            x_opt = x_opt / s
        else:
            x_opt = x0
        
        h = np.zeros(N)
        h[support] = x_opt
        sc = compute_ratio(h)
        return h, sc
    except Exception:
        h = np.zeros(N)
        h[support] = x0
        sc = compute_ratio(h)
        return h, sc

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin: kill at 150s, return by 145s

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []

    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))

    x = np.arange(N)
    center = N // 2

    # Phase 1: Structured Families (from current program)
    
    # 1. Gaussian
    for sigma in np.linspace(0.5, 15.0, 30):
        h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        sc = compute_ratio(h)
        update_best(h, sc)
    if time.time() > deadline - 30.0:
        return list(best_h)

    # 2. Triangular
    for width in np.linspace(2, N, 20):
        h = np.maximum(1.0 - np.abs(x - center) / width, 0.0)
        sc = compute_ratio(h)
        update_best(h, sc)
    if time.time() > deadline - 30.0:
        return list(best_h)

    # 3. Bimodal (Two Gaussians)
    for dist in np.linspace(1, N//2, 10):
        for sigma in np.linspace(0.5, 5.0, 10):
            h = np.exp(-0.5 * ((x - (center - dist)) / sigma) ** 2) + np.exp(-0.5 * ((x - (center + dist)) / sigma) ** 2)
            sc = compute_ratio(h)
            update_best(h, sc)
    if time.time() > deadline - 30.0:
        return list(best_h)

    # 4. Box / Top-hat
    for width in np.linspace(1, N, 20):
        h = np.zeros(N)
        start = int(center - width/2)
        end = int(center + width/2) + 1
        if start < 0: start = 0
        if end > N: end = N
        h[start:end] = 1.0
        sc = compute_ratio(h)
        update_best(h, sc)
    if time.time() > deadline - 30.0:
        return list(best_h)

    # 5. Exponential
    for decay in np.linspace(0.1, 0.8, 20):
        h = np.exp(-decay * np.abs(x - center))
        sc = compute_ratio(h)
        update_best(h, sc)
    if time.time() > deadline - 30.0:
        return list(best_h)

    # 6. Cauchy-like (Lorentzian)
    for gamma in np.linspace(0.5, 5.0, 10):
        h = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
        sc = compute_ratio(h)
        update_best(h, sc)
    if time.time() > deadline - 30.0:
        return list(best_h)

    # 7. Symmetric two-point: h[i] = a, h[j] = 1-a
    for i in range(N // 2):
        for j in range(i + 1, N):
            if j - i > 10:
                continue
            for a in np.linspace(0.01, 0.99, 50):
                h = np.zeros(N)
                h[i] = a
                h[j] = 1.0 - a
                sc = compute_ratio(h)
                update_best(h, sc)
            if time.time() > deadline - 20.0:
                return list(best_h)

    # 8. Symmetric three-point: h[i-1]=a, h[i]=b, h[i+1]=a, 2a+b=1
    for c in range(2, N - 2):
        for a in np.linspace(0.01, 0.49, 50):
            b = 1.0 - 2.0 * a
            if b < 0:
                continue
            h = np.zeros(N)
            h[c - 1] = a
            h[c] = b
            h[c + 1] = a
            sc = compute_ratio(h)
            update_best(h, sc)
        if time.time() > deadline - 20.0:
            return list(best_h)

    # 9. Sparse Sidon-like patterns
    # Try to place mass on indices that form a Sidon set (all pairwise sums distinct)
    # This reduces convolution collisions
    # For simplicity, try arithmetic progressions with large gaps
    for gap in [2, 3, 5, 7, 11]:
        for start in range(0, N - gap*3, 2):
            h = np.zeros(N)
            # Place 3-4 spikes
            num_spikes = min(4, (N - start) // gap)
            for k in range(num_spikes):
                idx = start + k * gap
                if idx < N:
                    h[idx] = 1.0 / num_spikes
            if np.sum(h) > 0:
                sc = compute_ratio(h)
                update_best(h, sc)
    if time.time() > deadline - 15.0:
        return list(best_h)

    # 10. Random sparse vectors
    np.random.seed(42)
    for _ in range(200):
        h = np.zeros(N)
        # Choose 2-5 random indices
        num_nonzero = np.random.randint(2, 6)
        indices = np.random.choice(N, num_nonzero, replace=False)
        weights = np.random.dirichlet(np.ones(num_nonzero))
        h[indices] = weights
        sc = compute_ratio(h)
        update_best(h, sc)
    if time.time() > deadline - 10.0:
        return list(best_h)

    # Phase 2: Sidon Set Optimization (from inspiration)
    sidon_sets = generate_sidon_sets()
    
    for support in sidon_sets:
        if time.time() > deadline - 15.0:
            break
        
        h_opt, sc_opt = optimize_weights_for_support(support, beta=50.0, max_iter=1000, tol=1e-8)
        update_best(h_opt, sc_opt)
    
    if time.time() > deadline - 10.0:
        return list(best_h)

    # Sort candidates by score and keep top K for local refinement
    candidates.sort(key=lambda item: item[0], reverse=True)
    top_k = candidates[:15]
    
    # Phase 3: Local Optimization from Top Candidates
    def objective_neg(h_flat):
        h = np.maximum(h_flat, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        h2 = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        cinf = np.max(conv)
        if cinf < 1e-15:
            return 1e10
        return -h2 / cinf

    for i, (sc, h_start) in enumerate(top_k):
        if time.time() > deadline - 10.0:
            break
        
        h_start = h_start / np.sum(h_start)
        
        # Nelder-Mead
        try:
            res = minimize(objective_neg, h_start, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            h_opt = np.maximum(res.x, 0.0)
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                sc_opt = compute_ratio(h_opt)
                update_best(h_opt, sc_opt)
        except Exception:
            pass

        # Also try a few random perturbations of the best candidate found so far
        if i == 0:
             # The best parametric candidate
             h_best_param = top_k[0][1] / np.sum(top_k[0][1])
             for _ in range(5):
                 if time.time() > deadline - 5.0:
                     break
                 noise = np.random.normal(0, 0.1, N)
                 h_pert = np.maximum(h_best_param + noise, 0.0)
                 s_p = np.sum(h_pert)
                 if s_p > 1e-15:
                     h_pert = h_pert / s_p
                     try:
                         res_p = minimize(objective_neg, h_pert, method='Nelder-Mead', 
                                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                         h_opt_p = np.maximum(res_p.x, 0.0)
                         s_opt_p = np.sum(h_opt_p)
                         if s_opt_p > 1e-15:
                             h_opt_p = h_opt_p / s_opt_p
                             sc_opt_p = compute_ratio(h_opt_p)
                             update_best(h_opt_p, sc_opt_p)
                     except Exception:
                         pass

    # Phase 4: Final polish - try to refine the best solution with tighter tolerance
    if best_score > 0.8 and time.time() < deadline - 3.0:
        try:
            h_final = best_h / np.sum(best_h)
            res_final = minimize(objective_neg, h_final, method='Nelder-Mead', 
                                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            h_opt_final = np.maximum(res_final.x, 0.0)
            s_final = np.sum(h_opt_final)
            if s_final > 1e-15:
                h_opt_final = h_opt_final / s_final
                sc_final = compute_ratio(h_opt_final)
                update_best(h_opt_final, sc_final)
        except Exception:
            pass

    return list(best_h)
