import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _calculate_ratio(h):
    """
    Calculate the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    h is a numpy array of nonnegative heights.
    """
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv == 0:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _sidon_greedy(max_index, target_size, seed=0):
    """
    Greedy construction of a Sidon-like set: indices with all pairwise sums distinct.
    Returns a list of indices (size up to target_size).
    """
    rng = np.random.RandomState(seed)
    used_sums = set()
    selected = []
    # Try multiple orderings, keep the best
    best_selected = []
    for attempt in range(30):
        used_sums = set()
        selected = []
        candidates = list(range(max_index + 1))
        # Bias: random shuffle but allow some structure
        rng.shuffle(candidates)
        for idx in candidates:
            if len(selected) >= target_size:
                break
            new_sums = set()
            ok = True
            for j in selected:
                s = idx + j
                if s in used_sums or s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
            if ok:
                selected.append(idx)
                used_sums.update(new_sums)
        if len(selected) > len(best_selected):
            best_selected = selected[:]
        if len(best_selected) >= target_size:
            break
    return best_selected

def _sidon_local_improve(indices, max_index, iters=200, seed=0):
    """
    Local search: try to grow the Sidon set by swapping/adding indices.
    indices: list of current Sidon indices.
    Returns improved list.
    """
    rng = np.random.RandomState(seed)
    cur = list(indices)
    used_sums = set()
    for a in cur:
        for b in cur:
            used_sums.add(a + b)
    
    def sums_for_set(s):
        st = set()
        for a in s:
            for b in s:
                st.add(a + b)
        return st
    
    best = cur[:]
    best_sums = sums_for_set(best)
    
    for it in range(iters):
        # Try adding a random new index
        improved = False
        for _ in range(50):
            cand = rng.randint(0, max_index + 1)
            if cand in cur:
                continue
            new_sums = set(best_sums)
            ok = True
            for j in cur:
                s = cand + j
                if s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
            if ok:
                # accept the addition
                cur = cur + [cand]
                best = cur[:]
                best_sums = new_sums
                improved = True
                break
        if not improved:
            # try a swap: remove one, add another
            if len(cur) < 2:
                break
            swapped = False
            for _ in range(50):
                remove_idx = rng.randint(0, len(cur))
                cand = rng.randint(0, max_index + 1)
                if cand in cur:
                    continue
                new_set = cur[:remove_idx] + cur[remove_idx+1:] + [cand]
                ns = sums_for_set(new_set)
                if len(ns) == len(new_set) * len(new_set):
                    cur = new_set
                    best = cur[:]
                    best_sums = ns
                    swapped = True
                    break
            if not swapped:
                break
    return best

def _optimize_weights_on_support(support, seed=0, deadline=None):
    """
    Given a support (list of indices), optimize nonnegative weights to maximize the ratio.
    Scale-invariant: fix sum=1.
    Returns (best_h, best_score).
    """
    m = len(support)
    if m == 0:
        return np.zeros(N), 0.0
    if m == 1:
        h = np.zeros(N)
        h[support[0]] = 1.0
        return h, _calculate_ratio(h)
    
    # Parameterize by m-1 free weights (last = 1 - sum of others) to enforce sum=1.
    # But simpler: optimize m weights with sum constraint handled by normalizing.
    # Use log-sum-exp smooth max surrogate on the autoconvolution.
    
    def smooth_objective(w):
        # w: length m, nonnegative
        w = np.maximum(w, 0)
        s = np.sum(w)
        if s < 1e-12:
            return 1e10
        w = w / s  # normalize to sum 1
        h = np.zeros(N)
        h[support] = w
        conv = np.convolve(h, h, mode='full')
        # smooth max
        tau = 0.01
        # shift for numerical stability
        cm = conv - np.max(conv)
        soft_max = np.max(conv) + tau * np.log(np.sum(np.exp(cm / tau)))
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-12:
            return 1e10
        return soft_max - np.log(l2_sq)
    
    rng = np.random.RandomState(seed)
    best_h = None
    best_score = -1e10
    
    for restart in range(8):
        if deadline is not None and time.time() > deadline:
            break
        # init: equal weights with noise
        w0 = np.ones(m) * (1.0 / m)
        w0 += rng.exponential(0.1, m)
        w0 = np.maximum(w0, 0)
        w0 = w0 / np.sum(w0)
        
        bounds = [(0, None)] * m
        try:
            res = minimize(
                smooth_objective,
                w0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            w_opt = np.maximum(res.x, 0)
            s = np.sum(w_opt)
            if s < 1e-12:
                continue
            w_opt = w_opt / s
            h = np.zeros(N)
            h[support] = w_opt
            sc = _calculate_ratio(h)
            if sc > best_score:
                best_score = sc
                best_h = h.copy()
        except Exception:
            continue
    
    if best_h is None:
        # fallback equal weights
        h = np.zeros(N)
        h[support] = 1.0 / m
        best_score = _calculate_ratio(h)
        best_h = h
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 150s kill
    
    best_score = 0.0
    best_h = None
    
    def consider(h):
        nonlocal best_score, best_h
        sc = _calculate_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        return sc
    
    # Candidate 1: spike
    h_spike = np.zeros(N); h_spike[0] = 1.0
    consider(h_spike)
    
    # Candidate 2: uniform
    consider(np.ones(N))
    
    # Candidate 3: two-point
    h2 = np.zeros(N); h2[0] = 0.5; h2[1] = 0.5
    consider(h2)
    
    # Phase 1: build large Sidon sets via greedy + local improve, with multiple seeds
    # Target sizes from large to small; keep the best per size.
    target_sizes = [12, 11, 10, 9, 8, 7, 6, 5]
    sidon_candidates = []
    for ts in target_sizes:
        if time.time() > deadline:
            break
        for seed in range(6):
            if time.time() > deadline:
                break
            s = _sidon_greedy(N - 1, ts, seed=seed)
            if len(s) >= 4:
                # local improve to try to grow
                s2 = _sidon_local_improve(s, N - 1, iters=50, seed=seed + 100)
                sidon_candidates.append(s2)
    
    # Deduplicate by frozenset
    seen = set()
    unique_sidon = []
    for s in sidon_candidates:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_sidon.append(s)
    # Sort by size descending
    unique_sidon.sort(key=lambda x: -len(x))
    
    # Phase 2: optimize weights on each Sidon support
    for s in unique_sidon:
        if time.time() > deadline:
            break
        if len(s) < 2:
            continue
        # quick equal-weight baseline
        h_eq = np.zeros(N)
        h_eq[s] = 1.0 / len(s)
        consider(h_eq)
        # weight optimization
        h_opt, sc_opt = _optimize_weights_on_support(s, seed=len(s), deadline=deadline)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
    
    # Phase 3: a few pure random-support optimizations as a safety net
    rng = np.random.RandomState(1234)
    for trial in range(10):
        if time.time() > deadline:
            break
        m = rng.randint(3, 11)
        support = list(rng.choice(N, size=m, replace=False))
        h_opt, sc_opt = _optimize_weights_on_support(support, seed=trial, deadline=deadline)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
    
    if best_h is None:
        return [1.0] + [0.0] * (N - 1)
    
    # Final polish: a tight local search on the best_h's support using exact ratio
    # (small N, cheap). Try perturbing weights on nonzero indices.
    nz = np.nonzero(best_h)[0]
    if len(nz) >= 2 and time.time() < deadline - 5:
        cur = best_h.copy()
        cur_sc = best_score
        for _ in range(300):
            if time.time() > deadline - 5:
                break
            # random perturbation of two weights
            i = rng.randint(0, len(nz))
            j = rng.randint(0, len(nz))
            if i == j:
                continue
            a, b = nz[i], nz[j]
            da = cur[a] * (rng.uniform(-0.3, 0.3))
            db = cur[b] * (rng.uniform(-0.3, 0.3))
            new = cur.copy()
            new[a] = max(0.0, cur[a] + da)
            new[b] = max(0.0, cur[b] + db)
            sc = _calculate_ratio(new)
            if sc > cur_sc + 1e-12:
                cur = new
                cur_sc = sc
        if cur_sc > best_score:
            best_score = cur_sc
            best_h = cur
    
    return list(best_h)
