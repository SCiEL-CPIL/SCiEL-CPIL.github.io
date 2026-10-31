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

def ratio_and_grad(h):
    """Compute ratio and gradient of ratio w.r.t. h (unnormalized)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0, np.zeros_like(h)
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0, np.zeros_like(h)
    
    ratio = h2_sq / conv_inf
    
    # Gradient of ratio = (h2_sq / conv_inf)
    # d ratio / d h_j = (d h2_sq / d h_j * conv_inf - h2_sq * d conv_inf / d h_j) / conv_inf^2
    # h2_sq = sum_i (h_i / l1)^2 = (1/l1^2) sum_i h_i^2
    # d h2_sq / d h_j = (2 h_j / l1^2) - (2 h_j sum h_i^2 / l1^3) = (2 h_j / l1^2)(1 - h2_sq)
    # Wait, let's be more careful.
    # Let S = sum h_i, Q = sum h_i^2.
    # h2_sq = Q / S^2.
    # d h2_sq / d h_j = (2 h_j S^2 - Q * 2 S) / S^4 = (2 h_j S - 2 Q) / S^3 = 2(h_j S - Q) / S^3.
    
    # conv_inf = max_k conv_k. Let K* be the argmax.
    # conv_k = sum_i h_norm_i h_norm_{k-i}.
    # d conv_k / d h_j = (1/S) (h_norm_{k-j} + h_norm_{j-k}) - (h_j/S^2) * (h_norm_{k-j} + h_norm_{j-k})
    # = (h_norm_{k-j} + h_norm_{j-k}) * (1/S - h_j/S^2)
    # = (h_norm_{k-j} + h_norm_{j-k}) * (S - h_j) / S^2
    
    # We need to handle the max. If the max is unique, we can use the gradient for that specific k.
    # If there are ties, the gradient is a subgradient. We can pick one or average.
    # For optimization, picking the one with the largest value is standard.
    
    max_indices = np.where(conv == conv_inf)[0]
    # If there are multiple max, we can sum gradients? Or just take the first?
    # Taking the first is a valid subgradient. Averaging might be smoother.
    # Let's average over the max indices to be safe.
    
    grad_h2 = np.zeros_like(h)
    for j in range(N):
        grad_h2[j] = 2.0 * (h[j] * l1 - h2_sq * l1) / (l1**3) # This is 2(h_j S - Q) / S^3. Q = h2_sq * S^2. So 2(h_j S - h2_sq S^2)/S^3 = 2(h_j - h2_sq S)/S^2.
        # Let's re-evaluate: d(Q/S^2)/dh_j = (2 h_j S^2 - Q 2 S)/S^4 = 2(h_j S - Q)/S^3.
        # Q = sum h_i^2.
        # So grad_h2[j] = 2 * (h[j] * l1 - np.sum(h**2)) / (l1**3)
    
    Q = np.sum(h**2)
    grad_h2 = 2.0 * (h * l1 - Q) / (l1**3)
    
    grad_conv_inf = np.zeros_like(h)
    for k in max_indices:
        # d conv_k / d h_j = (h_norm_{k-j} + h_norm_{j-k}) * (l1 - h_j) / l1^2
        # Note: h_norm is defined on [0, N-1]. h_norm[m] is 0 if m not in [0, N-1].
        for j in range(N):
            val1 = h_norm[k-j] if 0 <= k-j < N else 0.0
            val2 = h_norm[j-k] if 0 <= j-k < N else 0.0
            grad_conv_inf[j] += (val1 + val2) * (l1 - h[j]) / (l1**2)
    
    # Average gradient if multiple max
    if len(max_indices) > 1:
        grad_conv_inf /= len(max_indices)
    
    grad_ratio = (grad_h2 * conv_inf - h2_sq * grad_conv_inf) / (conv_inf**2)
    
    return ratio, grad_ratio

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
    Optimize weights h_i for i in support to maximize ratio.
    Uses projected gradient ascent with exact gradient.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initial guess: uniform weights
    h0 = np.zeros(N)
    h0[support] = 1.0 / k
    
    h = h0.copy()
    
    # Projected gradient ascent
    # We want to maximize ratio(h).
    # Constraint: h >= 0, sum(h) = 1.
    # Actually, ratio is scale invariant. So we can just optimize on the simplex.
    # Or we can optimize on h >= 0 and normalize at the end?
    # The gradient derived above is for unnormalized h.
    # But ratio is scale invariant, so the gradient in the radial direction is 0.
    # The gradient in the tangential directions is what matters.
    # Projected gradient ascent on the simplex:
    # 1. Compute gradient of ratio w.r.t h.
    # 2. Project gradient onto tangent space of simplex (subtract mean).
    # 3. Update h += step * grad_proj.
    # 4. Project h back onto simplex (clip negatives, normalize).
    
    def project_simplex(v):
        # Project v onto simplex {x | x >= 0, sum x = 1}
        # This is not the standard Euclidean projection, but a simple clip+normalize
        # which works well in practice for this type of problem.
        v = np.maximum(v, 0)
        s = np.sum(v)
        if s < 1e-15:
            return np.ones(N) / N
        return v / s
    
    step_size = 0.1
    n_iter = 1000
    
    for it in range(n_iter):
        ratio, grad = ratio_and_grad(h)
        
        # Project gradient onto tangent space of simplex
        # Tangent space: sum dx = 0.
        # Projection of g onto tangent space: g - mean(g) * ones.
        grad_proj = grad - np.mean(grad)
        
        # Update
        h_new = h + step_size * grad_proj
        
        # Project to simplex
        h_new = project_simplex(h_new)
        
        # Optional: restart if no improvement?
        # For now, just iterate.
        
        h = h_new
        
        # Adaptive step size?
        # If step is too large, ratio might decrease.
        # We can do a line search.
        # But for simplicity, let's fix step size for now.
        # Or use a simple backtracking line search.
        
    sc = compute_ratio(h)
    return h, sc

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
        
        h_opt, sc = optimize_weights_for_support(s)
        if sc > best_score:
            best_score = sc
            best_h = h_opt.copy()
            
        # Try with different beta (not used in new optimizer, but keep for interface compatibility if needed)
        # For now, just run the new optimizer once per set.
        # We can also try multiple random restarts within the support?
        # Let's add a few random perturbations to the initial uniform distribution.
        
        for restart in range(3):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb initial guess
            h_init = np.zeros(N)
            if restart == 0:
                h_init[s] = 1.0 / len(s)
            else:
                # Random positive weights
                weights = np.random.exponential(1.0, size=len(s))
                h_init[s] = weights / np.sum(weights)
            
            # Run optimizer with this initial guess
            # We need to modify optimize_weights_for_support to accept initial guess
            # For now, let's just rely on the single run.
            # Or we can implement a local search on top of it.
            
            # Let's implement a simple local search:
            # 1. Start with h_opt from the previous step (or uniform).
            # 2. Try swapping small amounts of weight between pairs of indices in support.
            # 3. Or try adding a small amount to one index and subtracting from another.
            
            pass

    return list(best_h)
