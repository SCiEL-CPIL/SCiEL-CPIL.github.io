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

def optimize_weights_for_support(support, beta=100.0):
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
        
        grad_h = np.zeros(N)
        # conv = fftconvolve(h, h)
        # d conv_k / d h_j = h_{k-j} + h_{j-k}
        # We need sum_k weights_k * (h_{k-j} + h_{j-k})
        # This is equivalent to 2 * (weights * h) convolved? No.
        # Let's just compute it directly for the support indices.
        
        # Actually, grad_h_j = sum_k weights_k * (h_{k-j} + h_{j-k})
        # This is 2 * sum_k weights_k * h_{k-j} (assuming symmetry, which holds for real h)
        # Let w = weights. Then grad_h = 2 * fftconvolve(w, h, mode='full')? No, indices are shifted.
        # conv_k = sum_i h_i h_{k-i}.
        # d conv_k / d h_j = h_{k-j} + h_{k-j} (if we consider h as symmetric in convolution)
        # Actually, d/dh_j sum_i h_i h_{k-i} = h_{k-j} + h_{k-j} = 2 h_{k-j} ?
        # Let's check: conv_k = h_j h_{k-j} + sum_{i != j} h_i h_{k-i}.
        # d/dh_j conv_k = h_{k-j} + h_j * 0 = h_{k-j}.
        # Wait, the term h_{k-j} comes from the i=j term in the sum.
        # The term h_j comes from the i such that k-i = j => i = k-j.
        # So if j != k-j, there are two terms: h_{k-j} and h_{k-j}.
        # So d conv_k / d h_j = h_{k-j} + h_{k-j} = 2 h_{k-j}.
        # If j = k-j (k=2j), then it's h_j * 1 + h_j * 1? No, it's just one term h_j^2, derivative is 2 h_j.
        # So in all cases, d conv_k / d h_j = 2 h_{k-j} (with h_m = 0 if m out of bounds).
        
        # So grad_h_j = 2 * sum_k weights_k * h_{k-j}
        # This is 2 * (weights * h) ? No, it's a convolution-like sum.
        # Let's compute it using FFT for speed if N is large, but N=50 is small.
        
        # grad_h[j] = 2 * sum_k weights[k] * h[k-j]
        # This is 2 * fftconvolve(weights, h, mode='full')[j + (N-1)]?
        # Let's just do it with loops for clarity and correctness, N=50 is small.
        
        # Actually, we can use np.dot or similar.
        # Let's stick to the loop for the support indices only, since h is sparse.
        
        # But weights are dense (conv length is 2N-1).
        # So we need to compute for each j in support:
        # grad_h[j] = 2 * sum_{k} weights[k] * h[k-j]
        # h[k-j] is non-zero only if k-j is in support.
        
        # Let's compute grad_h for all j in 0..N-1 first, then extract for support.
        # grad_h = 2 * fftconvolve(weights, h, mode='full') ?
        # Let's test this identity.
        # conv_w_h[m] = sum_k weights[k] h[m-k].
        # We want sum_k weights[k] h[k-j]. Let l = k-j => k = l+j.
        # sum_l weights[l+j] h[l]. This is cross-correlation.
        # fftconvolve(weights, h) gives sum_k weights[k] h[m-k].
        # If we reverse weights, we get correlation.
        # Let's just use the loop for now, it's fast enough for N=50.
        
        # Optimization:
        # grad_h[j] = 2 * sum_{i in support} h[i] * weights[j+i] ?
        # d conv_k / d h_j = 2 h_{k-j}.
        # sum_k weights_k 2 h_{k-j}. Let m = k-j => k = m+j.
        # 2 sum_m weights_{m+j} h_m.
        # So grad_h[j] = 2 sum_{m in support} h_m weights_{m+j}.
        
        grad_h = np.zeros(N)
        for m in support:
            hm = h[m]
            if hm > 1e-15:
                # weights indices m+j must be in [0, 2N-2]
                # j in [0, N-1], m in [0, N-1] => m+j in [0, 2N-2]
                # weights has length 2N-1
                grad_h[j] # j is the index we are differentiating w.r.t
                # We need to compute this for all j in support.
                # It's easier to compute for all j in 0..N-1.
        
        # Let's compute grad_h for all j in 0..N-1 using the formula:
        # grad_h[j] = 2 * sum_{m in support} h[m] * weights[m+j]
        
        # We can vectorize this using matrix multiplication or just a loop over m.
        # Since support size k is small, loop over m is good.
        for m in support:
            hm = h[m]
            if hm > 1e-15:
                # weights[m+j] for j in 0..N-1
                # This is weights[m : m+N]
                grad_h += 2.0 * hm * weights[m : m+N]
        
        # Now we need the gradient w.r.t x (the weights in support).
        # x[i] corresponds to h[support[i]].
        # So grad_x[i] = grad_h[support[i]].
        # But we also have the normalization constraint sum h = 1.
        # We used h = x / sum(x) in the objective? No, we defined objective with h = x / sum(x).
        # Wait, in the objective function above, I did h = h / s.
        # So the objective is f(x) = smooth_max(conv(x/sum(x))).
        # The gradient is more complex.
        # Let's simplify by fixing sum(x) = 1 in the optimizer constraints.
        # Then h = x.
        
        return grad_h[support]

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
    ]
    bounds = [(0.0, 1.0) for _ in range(k)]
    
    # Use L-BFGS-B with constraints? L-BFGS-B doesn't support general constraints well.
    # SLSQP is better for constraints.
    # Or we can use the fact that we are on the simplex.
    # Let's use SLSQP.
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': 500, 'ftol': 1e-9})
        x_opt = res.x
        x_opt = np.maximum(x_opt, 0.0)
        x_opt = x_opt / np.sum(x_opt)
        
        h_opt = np.zeros(N)
        h_opt[support] = x_opt
        sc = compute_ratio(h_opt)
        return h_opt, sc
    except Exception as e:
        # Fallback to uniform
        h_opt = np.zeros(N)
        h_opt[support] = 1.0 / k
        sc = compute_ratio(h_opt)
        return h_opt, sc

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    sidon_sets = generate_sidon_sets()
    
    # Sort by size descending? Larger sets might have better ratios?
    # Actually, for a fixed support, the ratio is roughly k / (max conv).
    # Uniform weights on Sidon set of size k: max conv is 1/k (if no collisions, which is true for Sidon).
    # So ratio = (1/k) / (1/k) = 1?
    # Wait. If h is uniform on Sidon set S of size k:
    # h_i = 1/k.
    # h2_sq = sum (1/k)^2 = k * (1/k^2) = 1/k.
    # conv_k = sum_i h_i h_{k-i}.
    # For Sidon set, each pair sum is unique.
    # The maximum convolution coefficient is 1/k (for the sums that appear once).
    # Wait, if all sums are distinct, then each conv_k is either 0 or 1/k.
    # So max conv = 1/k.
    # Ratio = (1/k) / (1/k) = 1.
    # So uniform weights on Sidon set give ratio 1.
    # We need to optimize weights to get ratio > 1.
    # This means we want to make the convolution more uniform (lower max) or make h2_sq larger.
    # Making h2_sq larger means concentrating weights.
    # But concentrating weights increases max conv.
    # There is a trade-off.
    
    # Let's try optimizing weights for each Sidon set.
    for s in sidon_sets:
        if time.time() > deadline - 10.0:
            break
        
        h_opt, sc = optimize_weights_for_support(s, beta=50.0)
        if sc > best_score:
            best_score = sc
            best_h = h_opt.copy()
            
        # Try with different beta
        if time.time() > deadline - 10.0:
            break
        h_opt2, sc2 = optimize_weights_for_support(s, beta=100.0)
        if sc2 > best_score:
            best_score = sc2
            best_h = h_opt2.copy()

    return list(best_h)
