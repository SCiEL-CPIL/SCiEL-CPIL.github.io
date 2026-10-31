"""Improved program: direct ratio optimization with fast conv and adaptive local search."""
import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

N = 50
DELTA = 1.0 / (2 * N)

def _compute_ratio(h):
    """Compute ratio ||h||_2^2 / (||h||_1 * sup|h*h|*DELTA)."""
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h * h)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    max_val = max_conv * DELTA
    if max_val == 0:
        return 0.0
    return l2_sq / (l1 * max_val)

def _ratio_from_weights(w, support):
    """Compute ratio from weights on support (faster, avoids full h construction)."""
    K = len(support)
    h = np.zeros(N)
    for i, idx in enumerate(support):
        h[idx] = w[i]
    l1 = np.sum(w)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(w * w)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    max_val = max_conv * DELTA
    if max_val == 0:
        return 0.0
    return l2_sq / (l1 * max_val)

def _greedy_sidon(n, k):
    """Greedy construction of Sidon set of size k in [0, n)."""
    if k <= 0:
        return []
    ruler = [0]
    for i in range(1, n):
        if len(ruler) >= k:
            break
        # Check if adding i preserves Sidon property
        new_ruler = ruler + [i]
        diffs = set()
        valid = True
        for a in range(len(new_ruler)):
            for b in range(a):
                d = abs(new_ruler[a] - new_ruler[b])
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if not valid:
                break
        if valid:
            ruler = new_ruler
    return ruler[:k]

def _random_sidon(n, k, rng):
    """Try to generate a random Sidon set."""
    for _ in range(500):
        candidate = sorted(rng.sample(range(n), k))
        diffs = set()
        valid = True
        for a in range(len(candidate)):
            for b in range(a):
                d = abs(candidate[a] - candidate[b])
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if not valid:
                break
        if valid:
            return candidate
    return None

def _local_search(h, support, max_iters=5000, rng=None):
    """Adaptive local search: coordinate + pair moves."""
    if rng is None:
        rng = random
    K = len(support)
    w = np.zeros(K)
    for i, idx in enumerate(support):
        w[i] = h[idx]
    
    best_ratio = _ratio_from_weights(w, support)
    best_w = w.copy()
    no_improve = 0
    
    for it in range(max_iters):
        move_type = rng.random()
        if move_type < 0.7:
            # Coordinate move: move mass from j to i
            i = rng.randint(0, K - 1)
            j = rng.randint(0, K - 1)
            if i == j:
                continue
            eps = best_w[j] * rng.uniform(0.05, 0.3)
            if eps <= 0 or best_w[i] + eps > 1.5:
                continue
            w_new = best_w.copy()
            w_new[i] += eps
            w_new[j] -= eps
            if np.any(w_new < -1e-12):
                continue
            w_new = np.clip(w_new, 0, None)
            ratio = _ratio_from_weights(w_new, support)
            if ratio > best_ratio:
                best_ratio = ratio
                best_w = w_new.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Pair swap: redistribute between two coordinates
            i, j = rng.sample(range(K), 2)
            total = best_w[i] + best_w[j]
            if total <= 0:
                continue
            alpha = rng.uniform(0, 1)
            w_new = best_w.copy()
            w_new[i] = total * alpha
            w_new[j] = total * (1 - alpha)
            ratio = _ratio_from_weights(w_new, support)
            if ratio > best_ratio:
                best_ratio = ratio
                best_w = w_new.copy()
                no_improve = 0
            else:
                no_improve += 1
        
        if no_improve > 200:
            no_improve = 0
            # Perturb: add small random noise
            w_perturbed = best_w + rng.random(K) * 0.02 * np.max(best_w)
            w_perturbed = np.clip(w_perturbed, 0, None)
            s = np.sum(w_perturbed)
            if s > 0:
                w_perturbed /= s
            ratio = _ratio_from_weights(w_perturbed, support)
            if ratio > best_ratio:
                best_ratio = ratio
                best_w = w_perturbed.copy()
    
    h_out = np.zeros(N)
    for i, idx in enumerate(support):
        h_out[idx] = best_w[i]
    return h_out, best_ratio

def _optimize_on_support(support, rng, time_budget=10.0):
    """Optimize weights on a given support using multiple strategies."""
    K = len(support)
    if K < 2:
        h = np.zeros(N)
        h[support[0]] = 1.0
        return h, _compute_ratio(h)
    
    best_ratio = 0.0
    best_h = None
    
    # Strategy 1: SLSQP from multiple inits
    inits = []
    inits.append(np.ones(K) / K)
    for d in range(min(3, K)):
        w = np.zeros(K)
        w[d] = 0.7
        rest = 0.3 / (K - 1)
        for i in range(K):
            if i != d:
                w[i] = rest
        inits.append(w)
    for _ in range(4):
        inits.append(rng.dirichlet(np.ones(K)))
    
    def objective(w):
        return -_ratio_from_weights(w, support)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0, 1) for _ in range(K)]
    
    for w_init in inits:
        if time.time() > start_time + time_budget:
            break
        try:
            res = minimize(objective, w_init, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 80, 'ftol': 1e-10})
            if res.success or res.x is not None:
                w_opt = np.clip(res.x, 0, None)
                s = np.sum(w_opt)
                if s > 0:
                    w_opt /= s
                h_opt = np.zeros(N)
                for i, idx in enumerate(support):
                    h_opt[idx] = w_opt[i]
                ratio = _compute_ratio(h_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Strategy 2: Local search from best found
    if best_h is not None:
        h_polished, ratio_polished = _local_search(best_h, support, max_iters=3000, rng=rng)
        if ratio_polished > best_ratio:
            best_ratio = ratio_polished
            best_h = h_polished.copy()
    
    if best_h is None:
        h = np.zeros(N)
        h[support[0]] = 1.0
        return h, _compute_ratio(h)
    
    return best_h, best_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    best_h = None
    best_ratio = 0.0
    
    # Generate diverse supports
    supports = set()
    
    # Greedy Sidon sets of various sizes
    for k in [6, 7, 8, 5, 4, 3]:
        s = _greedy_sidon(N, k)
        if len(s) >= 3:
            supports.add(tuple(s))
    
    # Random Sidon sets
    for k in [4, 5, 6, 7]:
        for _ in range(15):
            s = _random_sidon(N, k, rng)
            if s is not None:
                supports.add(tuple(s))
    
    # Random sparse sets (not necessarily Sidon)
    for k in [3, 4, 5]:
        for _ in range(20):
            s = tuple(sorted(rng.sample(range(N), k)))
            supports.add(s)
    
    # Some structured sets: evenly spaced, clustered
    for base in range(0, N - 4, 3):
        supports.add(tuple([base, base+2, base+5, base+9]))
        supports.add(tuple([base, base+1, base+4, base+8]))
    
    # Convert to list and shuffle
    support_list = list(supports)
    rng.shuffle(support_list)
    
    # Process supports in order of time budget
    time_per_support = max(5.0, (deadline - start_time) / max(1, len(support_list)))
    
    for support in support_list:
        if time.time() > deadline:
            break
        remaining = deadline - start_time
        if remaining < 3.0:
            break
        
        t_budget = min(time_per_support, remaining * 0.8)
        try:
            h, ratio = _optimize_on_support(list(support), rng, time_budget=t_budget)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h.copy()
        except Exception:
            continue
    
    # Final local search on best result
    if best_h is not None and time.time() < deadline - 10:
        # Find support of best_h
        nz = np.nonzero(best_h)[0]
        if len(nz) >= 2:
            h_polished, ratio_polished = _local_search(best_h, list(nz), max_iters=5000, rng=rng)
            if ratio_polished > best_ratio:
                best_ratio = ratio_polished
                best_h = h_polished.copy()
    
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return [float(x) for x in best_h]
