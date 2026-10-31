"""Second autocorrelation inequality - optimize step heights."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    """Compute the objective: L2^2 / (L1 * sup|autoconv|).
    
    Scale-invariant, so we normalize L1=1.
    The autoconvolution includes lag 0. For a probability vector,
    ac[0] = sum h[i]^2 = L2^2. The supremum over all lags includes lag 0.
    So ratio = L2^2 / (L1 * max(ac)) where ac = autoconvolution.
    With L1=1: ratio = L2^2 / max(ac).
    Since ac[0] = L2^2, max(ac) >= L2^2, so ratio <= 1.
    
    Wait - let me re-read the task. "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    This is: (||h||_2^2) / (||h||_1 * sup|h*h|)
    
    For nonnegative h, h*h (autoconvolution) has max at lag 0 which equals ||h||_2^2.
    So the ratio would always be 1 if we include lag 0.
    
    But the score is 0.9691 for uniform, which is less than 1. Let me check:
    For uniform h = ones(50): L1 = 50, L2^2 = 50, autoconv max = 50 (at lag 0)
    Ratio = 50 / (50 * 50) = 0.2. That's not 0.9691 either.
    
    Hmm, let me reconsider. Maybe the autoconvolution excludes lag 0?
    For uniform: off-peak max of autoconv = 49 (at lag 1)
    Ratio = 50 / (50 * 49) = 1/49 ≈ 0.0204. Still not matching.
    
    Wait, maybe it's normalized differently. Let me look at the seed code more carefully.
    The seed uses: l2_sq / (l1 * ac_max) where ac is the full autoconvolution including lag 0.
    For uniform ones(50): l2_sq = 50, l1 = 50, ac_max = 50 (lag 0).
    Score = 50/(50*50) = 0.2. But the reported score is 0.9691.
    
    The search score is value/seed. So if seed program gives 0.2, and the score is 0.9691,
    that means the actual value is 0.9691 * 0.2 = 0.1938? That doesn't make sense either.
    
    Actually re-reading: "score is the lower bound on C2" and "search score is value/seed, so 1.0 equals the seed program"
    So the evaluator computes some value, and the search score is that value divided by what the seed program produces.
    
    Let me just use the same formula as the seed: l2_sq / (l1 * ac_max) where ac includes all lags.
    """
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
        return h2_sq / 1e-15
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

def generate_sidon_sets(n, max_iter=10000):
    """Generate several Sidon (B2) sets of size n from {0,...,2n-2}."""
    results = []
    np.random.seed(12345)
    for _ in range(max_iter):
        if len(results) >= 20:
            break
        # Greedy construction
        S = []
        used_sums = set()
        available = list(range(2 * n - 1))
        np.random.shuffle(available)
        for v in available:
            new_sums = set()
            for s in S:
                ns = s + v
                if ns in used_sums or ns in new_sums:
                    break
                new_sums.add(ns)
            else:
                new_sums.add(2 * v)
                if 2 * v in used_sums:
                    break
                S.append(v)
                used_sums |= new_sums
                if len(S) == n:
                    break
        if len(S) == n:
            S.sort()
            if S not in results:
                results.append(S)
    return results

def optimize_weights_for_support(support, h_initial=None, time_budget=5.0):
    """Optimize weights on a given support (list of indices) using L-BFGS-B."""
    k = len(support)
    
    if h_initial is not None:
        x0 = np.array([h_initial[i] for i in support])
        x0 = np.maximum(x0, 1e-8)
    else:
        x0 = np.ones(k) / k
    
    def neg_ratio(log_w):
        w = np.exp(log_w)
        w = w / np.sum(w)  # normalize to sum=1
        h = np.zeros(N)
        for idx, wi in zip(support, w):
            h[idx] = wi
        return -compute_ratio(h)
    
    def neg_ratio_grad(log_w):
        w = np.exp(log_w)
        w = w / np.sum(w)
        h = np.zeros(N)
        for idx, wi in zip(support, w):
            h[idx] = wi
        
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros(k)
        f0 = neg_ratio(log_w)
        for i in range(k):
            lw_p = log_w.copy()
            lw_p[i] += eps
            f_p = neg_ratio(lw_p)
            lw_m = log_w.copy()
            lw_m[i] -= eps
            f_m = neg_ratio(lw_m)
            grad[i] = (f_p - f_m) / (2 * eps)
        return grad
    
    start_time = time.time()
    try:
        res = minimize(neg_ratio, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                               'maxfun': 2000})
        if time.time() - start_time > time_budget:
            return None, -1
        best_log_w = res.x
        best_val = -res.fun
    except Exception:
        return None, -1
    
    # Convert back
    w = np.exp(best_log_w)
    w = w / np.sum(w)
    h = np.zeros(N)
    for idx, wi in zip(support, w):
        h[idx] = wi
    return h, best_val

def local_search_polish(h, time_budget=5.0):
    """Local search: try moving mass between adjacent indices."""
    h = np.maximum(h, 0.0)
    h = h / np.sum(h)
    best = h.copy()
    best_val = compute_ratio(best)
    start_time = time.time()
    
    step_size = 0.05
    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N - 1):
            if h[i] < 1e-8 and h[i+1] < 1e-8:
                continue
            # Try moving mass from i to i+1
            for frac in [step_size, step_size/2, step_size/4]:
                h_trial = h.copy()
                move = min(h[i], frac)
                if move < 1e-10:
                    continue
                h_trial[i] -= move
                h_trial[i+1] += move
                h_trial = h_trial / np.sum(h_trial)
                val = compute_ratio(h_trial)
                if val > best_val + 1e-10:
                    best = h_trial.copy()
                    best_val = val
                    improved = True
            # Try moving mass from i+1 to i
            for frac in [step_size, step_size/2, step_size/4]:
                h_trial = h.copy()
                move = min(h[i+1], frac)
                if move < 1e-10:
                    continue
                h_trial[i+1] -= move
                h_trial[i] += move
                h_trial = h_trial / np.sum(h_trial)
                val = compute_ratio(h_trial)
                if val > best_val + 1e-10:
                    best = h_trial.copy()
                    best_val = val
                    improved = True
        if not improved:
            break
    
    return best, best_val

def gradient_polish(h, time_budget=5.0):
    """Gradient-based polish using L-BFGS-B on the full vector."""
    h = np.maximum(h, 0.0)
    h = h / np.sum(h)
    
    # Work in log-space for positive entries, with small floor
    floor = 1e-10
    h_clipped = np.maximum(h, floor)
    h_clipped = h_clipped / np.sum(h_clipped)
    log_h = np.log(h_clipped)
    
    def neg_ratio(log_w):
        w = np.exp(log_w)
        w = w / np.sum(w)
        return -compute_ratio(w)
    
    start_time = time.time()
    try:
        res = minimize(neg_ratio, log_h, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-11,
                               'maxfun': 5000})
        if time.time() - start_time > time_budget:
            return h, compute_ratio(h)
        best_log_w = res.x
        w = np.exp(best_log_w)
        w = w / np.sum(w)
        val = compute_ratio(w)
        return w, val
    except Exception:
        return h, compute_ratio(h)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin (150s limit, 20s buffer)
    
    best_x = np.ones(N) / N
    best_score = compute_ratio(best_x)
    
    # Phase 1: Try Sidon set supports
    sidon_sets = generate_sidon_sets(N, max_iter=5000)
    
    for support in sidon_sets:
        if time.time() > deadline - 30:
            break
        h, val = optimize_weights_for_support(support, time_budget=3.0)
        if h is not None and val > best_score:
            best_score = val
            best_x = h.copy()
    
    # Phase 2: Differential Evolution for global search
    if time.time() < deadline - 40:
        def neg_ratio_de(x):
            x = np.maximum(x, 0)
            s = np.sum(x)
            if s < 1e-15:
                return 1.0
            x = x / s
            return -compute_ratio(x)
        
        bounds = [(0, 1)] * N
        try:
            de_res = differential_evolution(neg_ratio_de, bounds, 
                                            maxiter=100, popsize=15,
                                            seed=42, tol=1e-8,
                                            polish=False,
                                            updating='deferred')
            de_x = np.maximum(de_res.x, 0)
            de_x = de_x / np.sum(de_x)
            de_val = compute_ratio(de_x)
            if de_val > best_score:
                best_score = de_val
                best_x = de_x.copy()
        except Exception:
            pass
    
    # Phase 3: Gradient polish on the best found
    if time.time() < deadline - 15:
        h, val = gradient_polish(best_x, time_budget=10.0)
        if val > best_score:
            best_score = val
            best_x = h.copy()
    
    # Phase 4: Local search polish
    if time.time() < deadline - 5:
        h, val = local_search_polish(best_x, time_budget=5.0)
        if val > best_score:
            best_score = val
            best_x = h.copy()
    
    # Phase 5: Final gradient polish
    if time.time() < deadline - 2:
        h, val = gradient_polish(best_x, time_budget=3.0)
        if val > best_score:
            best_score = val
            best_x = h.copy()
    
    # Ensure nonnegative and valid
    best_x = np.maximum(best_x, 0.0)
    if np.sum(best_x) < 1e-12:
        best_x = np.ones(N) / N
    
    return list(best_x)

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
