import numpy as np
import time
import sys

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 140.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return -np.inf
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)


def _log_ratio_obj(x, beta):
    """
    Objective for L-BFGS-B: maximize log(L2^2) - log(L1) - logsumexp(conv/beta)*beta
    This is equivalent to maximizing log(ratio) with a smooth approximation of log(max(conv)).
    
    We minimize the negative of this.
    """
    h = np.exp(x)  # Ensure positivity via log-parameterization
    
    l1 = np.sum(h)
    l2_sq = np.dot(h, h)
    
    conv = np.convolve(h, h, mode="full")
    
    # Smooth max approximation: beta * logsumexp(conv / beta)
    # As beta -> 0, this approaches max(conv)
    # We use log-sum-exp for numerical stability
    max_conv = np.max(conv)
    if max_conv > 0:
        lse = max_conv + beta * np.log(np.sum(np.exp((conv - max_conv) / beta)))
    else:
        lse = 0.0
    
    # Objective: log(l2_sq) - log(l1) - log(lse)
    # We want to maximize this, so we return negative for minimization
    if l1 <= 0 or l2_sq <= 0 or lse <= 0:
        return 0.0
    return -(np.log(l2_sq) - np.log(l1) - np.log(lse))


def _log_ratio_grad(x, beta):
    """
    Gradient of the negative log-ratio objective with respect to x (log-parameters).
    
    h = exp(x)
    f(x) = -[log(l2_sq) - log(l1) - log(lse)]
         = -log(l2_sq) + log(l1) + log(lse)
    
    df/dx_j = df/dh_j * dh_j/dx_j = df/dh_j * h_j
    
    df/dh_j = -2h_j/l2_sq + 1/l1 + (1/lse) * d(lse)/dh_j
    
    d(lse)/dh_j = (1/beta) * sum_k [ exp((conv_k - max_conv)/beta) * d(conv_k)/dh_j ]
    
    d(conv_k)/dh_j = 2 * h_{k-j} (with appropriate boundary handling)
    """
    h = np.exp(x)
    l1 = np.sum(h)
    l2_sq = np.dot(h, h)
    
    conv = np.convolve(h, h, mode="full")
    max_conv = np.max(conv)
    
    if l1 <= 0 or l2_sq <= 0 or max_conv <= 0:
        return np.zeros(N)
    
    # Compute log-sum-exp gradient
    # lse = max_conv + beta * log(sum(exp((conv - max_conv)/beta)))
    # d(lse)/dh_j = (1/beta) * sum_k exp((conv_k - max_conv)/beta) * d(conv_k)/dh_j
    
    # Precompute weights for each conv coefficient
    # For numerical stability
    shifted = (conv - max_conv) / beta
    weights = np.exp(shifted)
    sum_weights = np.sum(weights)
    
    # d(lse)/dh_j = (1/beta) * sum_k weights_k * d(conv_k)/dh_j
    # d(conv_k)/dh_j = 2 * h_{k-j} if k-j in [0, N-1], else 0
    # But we need to be careful with the convolution structure.
    # conv_k = sum_i h_i h_{k-i}
    # d(conv_k)/dh_j = h_{k-j} (from i=j) + h_{k-j} (from i=k-j, if k-j != j)
    # = 2 * h_{k-j} (if k-j in range), but if k-j == j, it's 2*h_j
    # Actually, it's always 2 * h_{k-j} where h_out_of_range = 0
    
    # Let's compute d_lse_dh vector
    d_lse_dh = np.zeros(N)
    for j in range(N):
        # Sum over all k where conv_k depends on h_j
        # conv_k depends on h_j if k-j in [0, N-1] or j-k in [0, N-1]
        # Actually, conv_k = sum_i h_i h_{k-i}
        # d(conv_k)/dh_j = h_{k-j} + h_{j-k} (if both in range, but h is defined on [0,N-1])
        # Wait, the convolution is:
        # (h*h)_k = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h_i * h_{k-i}
        # d/dh_j of this is:
        # If j is in the range of i, then term h_j * h_{k-j} contributes h_{k-j}
        # If j is in the range of k-i, i.e., k-j is in range, then term h_{k-j} * h_j contributes h_{k-j}
        # So if both j and k-j are in [0, N-1], derivative is 2*h_{k-j}
        # If only one is in range, it's h_{other}
        # But since h is zero outside [0,N-1], we can just say:
        # d(conv_k)/dh_j = 2 * h_{k-j} if k-j in [0,N-1], else 0
        # Wait, that's not quite right. Let's think again.
        # conv_k = sum_i h_i h_{k-i}
        # d(conv_k)/dh_j:
        # The term where i=j is h_j h_{k-j}, derivative is h_{k-j} (if k-j in range)
        # The term where k-i=j, i.e., i=k-j is h_{k-j} h_j, derivative is h_j (if k-j in range)
        # If j == k-j, i.e., k=2j, then it's the same term, derivative is 2*h_j
        # So in general, if k-j in [0,N-1], derivative is 2*h_{k-j}
        # Otherwise 0.
        
        # But wait, this double counts when j == k-j.
        # Let's just compute it directly.
        # For a given j, which k values have conv_k depending on h_j?
        # k such that j is in the sum index range for conv_k.
        # This is complex. Let's use a different approach.
        
        # Alternative: d_lse_dh_j = (1/beta) * sum_k weights_k * d(conv_k)/dh_j
        # Let's compute d(conv_k)/dh_j for all k, j using matrix form or direct computation.
        
        pass  # We'll compute this more efficiently below
    
    # More efficient computation of d_lse_dh
    # d_lse_dh_j = (1/beta) * sum_k weights_k * (2 * h_{k-j} if k-j in [0,N-1] else 0)
    # Let's reindex. Let l = k-j. Then k = l+j.
    # d_lse_dh_j = (2/beta) * sum_l weights_{l+j} * h_l, where l in [0,N-1] and l+j in [0, 2N-2]
    
    # We can compute this using convolution-like operations
    # Let W = weights (length 2N-1)
    # d_lse_dh_j = (2/beta) * sum_l W_{l+j} * h_l
    
    # This is a cross-correlation of W and h
    # d_lse_dh = (2/beta) * np.convolve(h, W[::-1], mode='valid')? No, let's think.
    # sum_l W_{l+j} h_l
    # Let m = l+j, l = m-j. sum_m W_m h_{m-j}
    # This is (h * W)[j] in some sense.
    
    # Actually, let's just compute it directly for clarity and correctness
    d_lse_dh = np.zeros(N)
    for j in range(N):
        s = 0.0
        for l in range(N):
            k = l + j
            if 0 <= k < len(conv):
                s += weights[k] * h[l]
        d_lse_dh[j] = (2.0 / beta) * s
    
    # Now df/dh_j = -2h_j/l2_sq + 1/l1 + (1/lse) * d_lse_dh_j
    # lse = max_conv + beta * log(sum_weights)
    lse = max_conv + beta * np.log(sum_weights)
    
    df_dh = -2.0 * h / l2_sq + 1.0 / l1 + (1.0 / lse) * d_lse_dh
    
    # df/dx_j = df_dh_j * h_j
    df_dx = df_dh * h
    
    return df_dx


def _lbfgs_optimize(x0, beta, maxiter=2000):
    """Optimize using L-BFGS-B with log-parameterization."""
    from scipy.optimize import minimize
    
    def obj(x):
        return _log_ratio_obj(x, beta)
    
    def grad(x):
        return _log_ratio_grad(x, beta)
    
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False})
        return res.x
    except Exception:
        return x0


def _from_log_param(x):
    """Convert log-parameterization to actual heights, normalized to L1=1."""
    h = np.exp(x)
    h = np.clip(h, 0.0, None)
    s = np.sum(h)
    if s > 0:
        h = h / s
    return h


def _polish_lbfgs(h_init, betas=[1.0, 0.5, 0.2, 0.1, 0.05], maxiter_per_beta=1000):
    """
    Multi-scale beta annealing with L-BFGS-B.
    Start with large beta (smooth), reduce to small beta (sharp).
    """
    h = h_init.copy()
    h = np.clip(h, 1e-10, None)  # Avoid log(0)
    x = np.log(h)
    
    best_h = h.copy()
    best_r = _ratio(h)
    
    for beta in betas:
        if _time_left() < 5:
            break
        
        x = _lbfgs_optimize(x, beta, maxiter=maxiter_per_beta)
        h_new = _from_log_param(x)
        
        r_new = _ratio(h_new)
        if r_new > best_r:
            best_r = r_new
            best_h = h_new.copy()
        
        # Update x for next iteration
        h = np.clip(h_new, 1e-10, None)
        x = np.log(h)
    
    return best_h


def _find_sidon_support(k, seed=0):
    """Find a Sidon-like support set of size k in [0, N)."""
    rng = np.random.RandomState(seed)
    best_set = None
    best_score = -1
    
    for attempt in range(50):
        S = []
        sums = set()
        valid = True
        
        for _ in range(k):
            found = False
            for _trial in range(100):
                cand = rng.randint(0, N)
                if cand in S:
                    continue
                
                new_sums = set()
                conflict = False
                for s in S:
                    pair_sum = cand + s
                    if pair_sum in sums:
                        conflict = True
                        break
                    new_sums.add(pair_sum)
                if not conflict:
                    if cand + cand in sums:
                        conflict = True
                    else:
                        new_sums.add(cand + cand)
                    
                    if not conflict:
                        S.append(cand)
                        sums.update(new_sums)
                        found = True
                        break
            
            if not found:
                valid = False
                break
        
        if valid and len(S) == k:
            S_sorted = sorted(S)
            gaps = np.diff(S_sorted)
            score = np.min(gaps) if len(gaps) > 0 else 0
            if score > best_score:
                best_score = score
                best_set = S_sorted
    
    if best_set is None:
        best_set = sorted(rng.choice(N, k, replace=False).tolist())
    
    return best_set


def construct():
    np.random.seed(12345)
    best_h = np.ones(N) / N
    best_r = _ratio(best_h)

    def upd(h):
        nonlocal best_h, best_r
        h = np.asarray(h, dtype=np.float64)
        if np.sum(h) <= 0 or not np.all(np.isfinite(h)):
            return
        hh = np.clip(h, 0.0, None)
        r = _ratio(hh)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_h = hh.copy()

    # ---------------- PHASE 1: Generate diverse initial points ----------------
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sidon set based inits
    for k in range(5, 15):
        for seed in range(3):
            if _time_left() < 10:
                break
            S = _find_sidon_support(k, seed=seed + k * 100)
            h = np.zeros(N)
            for idx in S:
                h[idx] = 1.0
            inits.append(h / np.sum(h))
    
    # 3. Gaussian bumps
    for _ in range(10):
        if _time_left() < 10:
            break
        a = np.random.uniform(0.5, 5.0)
        c = np.random.uniform(5, N - 5)
        h = np.exp(-a * (np.arange(N) - c) ** 2)
        inits.append(h / np.sum(h))
    
    # 4. Sparse random
    for _ in range(10):
        if _time_left() < 10:
            break
        k = np.random.randint(5, 20)
        h = np.zeros(N)
        h[np.random.choice(N, k, replace=False)] = np.random.uniform(0.1, 1.0, k)
        inits.append(h / np.sum(h))
    
    # 5. Two-point peaks
    for _ in range(5):
        if _time_left() < 10:
            break
        h = np.zeros(N)
        i1, i2 = np.random.choice(N, 2, replace=False)
        h[i1] = 1.0
        h[i2] = np.random.uniform(0.5, 1.0)
        inits.append(h / np.sum(h))

    # ---------------- PHASE 2: L-BFGS-B optimization with beta annealing ----------------
    for h_init in inits:
        if _time_left() < 15:
            break
        
        h_opt = _polish_lbfgs(h_init, betas=[2.0, 1.0, 0.5, 0.2, 0.1], maxiter_per_beta=500)
        upd(h_opt)

    # ---------------- PHASE 3: Final refinement with smaller beta ----------------
    if _time_left() > 20:
        h_final = _polish_lbfgs(best_h, betas=[0.5, 0.2, 0.1, 0.05, 0.02], maxiter_per_beta=1000)
        upd(h_final)

    # ---------------- PHASE 4: Additional random restarts if time permits ----------------
    for _ in range(5):
        if _time_left() < 15:
            break
        h_rand = np.random.uniform(0.01, 1.0, N)
        h_rand /= np.sum(h_rand)
        h_opt = _polish_lbfgs(h_rand, betas=[1.0, 0.5, 0.2, 0.1], maxiter_per_beta=800)
        upd(h_opt)

    # ---------------- PHASE 5: Final polish ----------------
    if _time_left() > 10:
        h_final = _polish_lbfgs(best_h, betas=[0.1, 0.05, 0.02, 0.01], maxiter_per_beta=2000)
        upd(h_final)

    return list(best_h)
