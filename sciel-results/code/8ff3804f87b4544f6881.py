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

def local_search_polish(h, support, max_iter=200, tol=1e-10):
    h = h.copy()
    k = len(support)
    if k < 2:
        return h
    best_score = compute_ratio(h)
    step_scale = 0.05
    for it in range(max_iter):
        improved = False
        # Try all pairs with multiple step sizes
        for i in range(k):
            for j in range(i+1, k):
                for delta in [step_scale, step_scale*0.3, step_scale*0.1, -step_scale, -step_scale*0.3, -step_scale*0.1]:
                    h_new = h.copy()
                    hi = h[support[i]]
                    hj = h[support[j]]
                    new_hi = hi + delta
                    new_hj = hj - delta
                    if new_hi >= 0 and new_hj >= 0:
                        h_new[support[i]] = new_hi
                        h_new[support[j]] = new_hj
                        s = np.sum(h_new)
                        if s > 1e-15:
                            h_new = h_new / s
                            sc = compute_ratio(h_new)
                            if sc > best_score + tol:
                                h = h_new
                                best_score = sc
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try single coordinate moves
            for i in range(k):
                for delta in [step_scale, step_scale*0.3, step_scale*0.1, -step_scale, -step_scale*0.3, -step_scale*0.1]:
                    h_new = h.copy()
                    hi = h[support[i]]
                    new_hi = hi + delta
                    if new_hi >= 0:
                        h_new[support[i]] = new_hi
                        s = np.sum(h_new)
                        if s > 1e-15:
                            h_new = h_new / s
                            sc = compute_ratio(h_new)
                            if sc > best_score + tol:
                                h = h_new
                                best_score = sc
                                improved = True
                                break
                if improved:
                    break
            if not improved:
                step_scale *= 0.5
                if step_scale < 1e-9:
                    break
    return h

def is_sidon_set(indices):
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

def softmax(z):
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

def optimize_weights_for_support(support, beta=100.0, x_init=None):
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    if x_init is not None:
        x0 = np.asarray(x_init, dtype=np.float64)
        x0 = np.maximum(x0, 0.0)
        s = np.sum(x0)
        if s > 1e-15:
            x0 = x0 / s
        else:
            x0 = np.ones(k) / k
    else:
        x0 = np.ones(k) / k

    def objective_and_grad(x):
        h = np.zeros(N)
        h[support] = x
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(k)
        h = h / s
        conv = fftconvolve(h, h, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        log_sum_exp = max_c + np.log(sum_exp)
        obj = log_sum_exp / beta
        # Gradient of smooth-max w.r.t. h: d/dh_i [sum_j w_j * conv_j] where conv_j = sum_{a+b=j} h_a h_b
        # d(conv_j)/d(h_i) = sum_{a: a+i=j} h_a + sum_{b: i+b=j} h_b = 2 * (h * delta_j)[i] for the full conv
        # For conv index j, d(conv_j)/d(h_i) = h_{j-i} + h_{j-i} if j-i in [0,N-1]... 
        # Actually conv_j = sum_{a=0}^{N-1} sum_{b=0}^{N-1} h_a h_b * [a+b=j]
        # d(conv_j)/d(h_i) = sum_{a: a+i=j} h_a + sum_{b: i+b=j} h_b
        #                    = h_{j-i} (if 0<=j-i<N) + h_{j-i} (if 0<=j-i<N)
        #                    = 2*h_{j-i} if 0<=j-i<N, else 0
        # So d(obj)/d(h_i) = sum_j weights[j] * 2*h_{j-i} (with bounds check)
        grad_h = np.zeros(N)
        for i in range(N):
            if h[i] <= 1e-15:
                continue
            # For conv index j, d(conv_j)/d(h_i) = 2*h_{j-i} if 0 <= j-i < N
            # j ranges from 0 to 2N-2
            # j-i ranges from -i to 2N-2-i
            # Valid when 0 <= j-i < N, i.e., i <= j <= i+N-1
            lo = max(0, i)
            hi = min(2*N-1, i + N - 1)
            if lo <= hi:
                # j-i ranges from lo-i to hi-i
                idx_lo = lo - i
                idx_hi = hi - i
                grad_h[i] = 2.0 * np.sum(weights[lo:hi+1] * h[idx_lo:idx_hi+1])
        grad_x = grad_h[support]
        # Project onto simplex tangent: subtract mean
        grad_x = grad_x - np.mean(grad_x)
        return obj, grad_x

    def objective_z(z):
        x = softmax(z)
        obj, _ = objective_and_grad(x)
        return obj

    def grad_z(z):
        x = softmax(z)
        obj, grad_x = objective_and_grad(x)
        sum_grad_x_x = np.sum(grad_x * x)
        grad_z = x * (grad_x - sum_grad_x_x)
        return grad_z

    z0 = np.log(np.maximum(x0, 1e-15))
    z0 -= np.max(z0)
    try:
        res = minimize(objective_z, z0, jac=grad_z, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        z_opt = res.x
        x_opt = softmax(z_opt)
        x_opt = np.maximum(x_opt, 0.0)
        x_opt = x_opt / np.sum(x_opt)
        h_opt = np.zeros(N)
        h_opt[support] = x_opt
        h_opt = local_search_polish(h_opt, support, max_iter=100)
        sc = compute_ratio(h_opt)
        return h_opt, sc
    except Exception:
        h_opt = np.zeros(N)
        h_opt[support] = 1.0 / k
        h_opt = local_search_polish(h_opt, support, max_iter=50)
        sc = compute_ratio(h_opt)
        return h_opt, sc

def gradient_polish(h, support, max_iter=200, tol=1e-12, beta=500.0):
    k = len(support)
    if k < 2:
        return h
    x = h[support].copy()
    x = np.maximum(x, 0.0)
    s = np.sum(x)
    if s < 1e-15:
        return h
    x = x / s

    def objective_and_grad_z(z):
        x = softmax(z)
        h_full = np.zeros(N)
        h_full[support] = x
        conv = fftconvolve(h_full, h_full, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        log_sum_exp = max_c + np.log(sum_exp)
        obj = log_sum_exp / beta
        # Correct gradient computation
        grad_h = np.zeros(N)
        for i in range(N):
            if h_full[i] <= 1e-15:
                continue
            lo = max(0, i)
            hi = min(2*N-1, i + N - 1)
            if lo <= hi:
                idx_lo = lo - i
                idx_hi = hi - i
                grad_h[i] = 2.0 * np.sum(weights[lo:hi+1] * h_full[idx_lo:idx_hi+1])
        grad_x = grad_h[support]
        grad_x = grad_x - np.mean(grad_x)
        sum_grad_x_x = np.sum(grad_x * x)
        grad_z = x * (grad_x - sum_grad_x_x)
        return obj, grad_z

    z0 = np.log(np.maximum(x, 1e-15))
    z0 -= np.max(z0)
    try:
        res = minimize(objective_and_grad_z, z0, jac=None, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
        z_opt = res.x
        x_opt = softmax(z_opt)
        x_opt = np.maximum(x_opt, 0.0)
        x_opt = x_opt / np.sum(x_opt)
        h_new = h.copy()
        h_new[support] = x_opt
        return h_new
    except Exception:
        return h

def refine_candidate(h, deadline):
    """Full refinement pipeline for a candidate: support selection + weight optimization + polish."""
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return h, 0.0
    h = h / s
    # Select support: keep indices with meaningful mass
    support = [i for i in range(N) if h[i] > 1e-8]
    if len(support) < 2:
        support = list(np.argsort(h)[::-1][:5])
    # Cap support size for optimization tractability
    if len(support) > 12:
        support = sorted(support[:12])
    # Optimize weights on this support
    h_opt, sc = optimize_weights_for_support(support, beta=100.0)
    if sc > 0:
        # High-precision gradient polish
        if time.time() < deadline - 8.0:
            h_opt = gradient_polish(h_opt, support, max_iter=300, tol=1e-14, beta=500.0)
            sc = compute_ratio(h_opt)
        # Local search polish
        if time.time() < deadline - 5.0:
            h_opt = local_search_polish(h_opt, support, max_iter=300, tol=1e-13)
            sc = compute_ratio(h_opt)
    return h_opt, sc

def global_differential_search(deadline, seed=42):
    """
    Global search using Differential Evolution over a dense vector of length N.
    The vector v encodes both support (via sign/threshold) and weights (via magnitude).
    We maximize the ratio by minimizing the smooth-max surrogate of the convolution.
    """
    from scipy.optimize import differential_evolution
    np.random.seed(seed)
    
    # Search space: v in [-1, 1]^N. We map v to h via softplus + thresholding.
    # h_i = max(0, v_i) if we want nonneg, but to allow support selection we use:
    # h_i = exp(v_i) for all i (all positive), then the optimizer will push some to ~0.
    # Actually, to get sparsity, use: h_i = max(0, v_i) ** 2 (squared to emphasize large values)
    # Let's use: h_i = (max(0, v_i))^2, which is nonnegative and sparsifying.
    
    def map_v_to_h(v):
        # v in [-1, 1]^N
        # Use squared positive part to encourage sparsity
        h = np.maximum(v, 0.0) ** 2
        s = np.sum(h)
        if s < 1e-15:
            return np.ones(N) / N
        return h / s
    
    def objective(v):
        h = map_v_to_h(v)
        conv = fftconvolve(h, h, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0
        # Smooth max surrogate (lower is better)
        beta = 50.0
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        log_sum_exp = max_c + np.log(sum_exp)
        # We want to MINIMIZE the convolution peak, but MAXIMIZE h2_sq.
        # Ratio = h2_sq / conv_inf. To maximize ratio, minimize conv_inf / h2_sq.
        # Or: minimize (conv_inf - h2_sq * t) for some t. 
        # Simpler: minimize conv_inf (since h2_sq is also in play, but let's focus on conv_inf)
        # Actually, the ratio is h2_sq / conv_inf. Let's minimize -log(h2_sq / conv_inf) = log(conv_inf) - log(h2_sq)
        h2_sq = np.sum(h**2)
        if h2_sq < 1e-15:
            return 1e10
        return np.log(max_c) - np.log(h2_sq)
    
    bounds = [(-1.0, 1.0)] * N
    try:
        result = differential_evolution(
            objective, bounds,
            seed=seed,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            polish=False,
            update_interval=10,
            workers=1
        )
        if result.success or result.fun < 1e10:
            v_opt = result.x
            h_opt = map_v_to_h(v_opt)
            return h_opt
    except Exception:
        pass
    return None

def global_random_support_search(deadline, seed=99):
    """
    Global search: randomly select supports (not just Sidon), then optimize weights.
    This explores a wider space of support structures.
    """
    np.random.seed(seed)
    best_h = None
    best_score = -1.0
    
    for trial in range(200):
        if time.time() > deadline - 20.0:
            break
        # Random support size
        k = np.random.randint(3, 13)
        support = sorted(np.random.choice(N, k, replace=False).tolist())
        # Random initial weights
        x_init = np.random.dirichlet(np.ones(k) * np.random.uniform(0.5, 3.0))
        h_opt, sc = optimize_weights_for_support(support, beta=100.0, x_init=x_init)
        if sc > best_score:
            best_score = sc
            best_h = h_opt.copy()
    
    return best_h

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    # Phase 1: Sidon set optimization (proven baseline)
    sidon_sets = generate_sidon_sets()
    sidon_sets.sort(key=lambda s: len(s), reverse=True)
    
    for s in sidon_sets[:15]:  # Limit to top 15 for time
        if time.time() > deadline - 40.0:
            break
        for beta in [50.0, 100.0]:
            if time.time() > deadline - 40.0:
                break
            h_opt, sc = optimize_weights_for_support(s, beta=beta)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
    
    # Phase 2: Global Differential Evolution search
    if time.time() < deadline - 30.0:
        h_global = global_differential_search(deadline, seed=42)
        if h_global is not None:
            h_refined, sc = refine_candidate(h_global, deadline)
            if sc > best_score:
                best_score = sc
                best_h = h_refined.copy()
    
    # Phase 3: Global random support search
    if time.time() < deadline - 25.0:
        h_random = global_random_support_search(deadline, seed=99)
        if h_random is not None:
            h_refined, sc = refine_candidate(h_random, deadline)
            if sc > best_score:
                best_score = sc
                best_h = h_refined.copy()
    
    # Phase 4: Multiple random restarts from different seeds
    if time.time() < deadline - 20.0:
        for seed in [7, 13, 21, 33, 47]:
            if time.time() > deadline - 15.0:
                break
            h_global = global_differential_search(deadline, seed=seed)
            if h_global is not None:
                h_refined, sc = refine_candidate(h_global, deadline)
                if sc > best_score:
                    best_score = sc
                    best_h = h_refined.copy()
    
    # Final high-precision polish on the best solution
    if time.time() < deadline - 5.0:
        support = [i for i in range(N) if best_h[i] > 1e-10]
        if len(support) >= 2:
            # Multi-stage polish: start with moderate beta, increase for final stage
            for beta_stage in [200.0, 500.0, 1000.0]:
                if time.time() < deadline - 3.0:
                    best_h = gradient_polish(best_h, support, max_iter=200, tol=1e-14, beta=beta_stage)
                    best_score = compute_ratio(best_h)
            if time.time() < deadline - 2.0:
                best_h = local_search_polish(best_h, support, max_iter=500, tol=1e-14)
                best_score = compute_ratio(best_h)
            # One more gradient pass at very high beta
            if time.time() < deadline - 1.0:
                best_h = gradient_polish(best_h, support, max_iter=100, tol=1e-15, beta=2000.0)
                best_score = compute_ratio(best_h)
    
    return list(best_h)
