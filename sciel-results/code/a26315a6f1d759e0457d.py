import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 50

# ----------------------------------------------------------------------
# Core ratio (exact)
# ----------------------------------------------------------------------
def compute_ratio(h):
    """Exact value of ||h||_2^2 / (||h||_1 * sup|h*h|)."""
    h = np.asarray(h, dtype=float)
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    l2_sq = float(np.sum(h * h))
    conv = np.convolve(h, h, mode='full')
    max_conv = float(np.max(conv))
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

# ----------------------------------------------------------------------
# Smooth surrogate for optimization (log-sum-exp soft max)
# ----------------------------------------------------------------------
def _smooth_max(x, beta):
    m = np.max(x)
    if m < 0:
        m = 0.0
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _neg_smooth_ratio(params, beta):
    """Negative smooth ratio; params are log-weights (unconstrained)."""
    w = np.exp(params)
    s = np.sum(w)
    if s < 1e-300:
        return 0.0
    w = w / s
    l2_sq = np.sum(w * w)
    conv = np.convolve(w, w, mode='full')
    max_conv = _smooth_max(conv, beta)
    if max_conv < 1e-15:
        return 0.0
    return -l2_sq / max_conv

# ----------------------------------------------------------------------
# Sidon-set support generation (index set {0..99}, no repeated pair sums)
# ----------------------------------------------------------------------
def is_sidon_set(indices):
    """True if no two unordered pairs of distinct indices share a sum."""
    idx = list(indices)
    sums = set()
    for a, b in combinations(idx, 2):
        s = a + b
        if s in sums:
            return False
        sums.add(s)
    return True

def _sidon_greedy(max_size, seed, direction=1):
    """Greedy construction of a Sidon set in {0..99}."""
    rng = np.random.default_rng(seed)
    pool = list(range(100))
    if direction < 0:
        pool = pool[::-1]
    # Shuffle to get variety
    rng.shuffle(pool)
    selected = []
    sums = set()
    for i in pool:
        if len(selected) >= max_size:
            break
        ok = True
        for j in selected:
            s = i + j
            if s in sums:
                ok = False
                break
        if ok:
            # add sums with existing elements
            for j in selected:
                sums.add(i + j)
            selected.append(i)
    return selected

def generate_sidon_sets(max_size, n_seeds):
    """Generate many candidate Sidon supports of the given size."""
    sets = []
    seen = set()
    for seed in range(n_seeds):
        for direction in (1, -1):
            s = _sidon_greedy(max_size, seed + direction * 1000, direction)
            if len(s) == max_size:
                key = frozenset(s)
                if key not in seen:
                    seen.add(key)
                    sets.append(sorted(s))
    return sets

# ----------------------------------------------------------------------
# Optimize weights on a fixed support
# ----------------------------------------------------------------------
def optimize_weights_for_support(support, deadline, max_iters=300):
    """Optimize the weight distribution over a fixed support (sum=1)."""
    k = len(support)
    best_score = 0.0
    best_w = None

    # Initialize: uniform over support
    w = np.ones(k) / k
    params = np.log(w)
    best_score = compute_ratio(np.full(N, np.nan) * 0 + 0)  # placeholder
    # evaluate on full vector
    full = np.zeros(N)
    full[support] = w
    best_score = compute_ratio(full)
    best_w = w.copy()

    for beta in [30.0, 80.0, 150.0, 300.0]:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(
                _neg_smooth_ratio,
                params,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': max_iters, 'ftol': 1e-13, 'gtol': 1e-11},
            )
            w_new = np.exp(res.x)
            w_new = w_new / np.sum(w_new)
            full = np.zeros(N)
            full[support] = w_new
            sc = compute_ratio(full)
            if sc > best_score:
                best_score = sc
                best_w = w_new.copy()
                params = res.x
        except Exception:
            pass

    return best_score, best_w

def _support_full(support, w):
    full = np.zeros(N)
    full[support] = w
    return full

# ----------------------------------------------------------------------
# Local search polish (perturb weights, keep if improved)
# ----------------------------------------------------------------------
def local_search_polish(full, support, deadline, steps=400, scale=0.05):
    """Random perturbation local search on the full vector over support."""
    rng = np.random.default_rng(12345)
    cur = full.copy()
    cur_score = compute_ratio(cur)
    k = len(support)
    step = 0
    while time.time() < deadline - 5 and step < steps:
        step += 1
        # pick a random coordinate to perturb
        idx = rng.integers(0, k)
        pos = support[idx]
        cur2 = cur.copy()
        delta = rng.normal(0.0, scale)
        cur2[pos] = max(0.0, cur[pos] + delta)
        # renormalize to sum=1 (scale invariant, but keep fixed L1)
        s = np.sum(cur2)
        if s < 1e-15:
            continue
        cur2 = cur2 / s
        sc = compute_ratio(cur2)
        if sc > cur_score:
            cur = cur2
            cur_score = sc
    return cur_score, cur

# ----------------------------------------------------------------------
# Gradient polish (high precision)
# ----------------------------------------------------------------------
def gradient_polish(full, deadline, max_iters=600):
    """Final high-precision L-BFGS-B polish over all 50 coordinates."""
    h = np.maximum(full, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return 0.0, full
    h = h / s
    params = np.log(h + 1e-12)
    best_score = compute_ratio(h)
    best_h = h.copy()
    for beta in [300.0, 600.0]:
        if time.time() > deadline - 4:
            break
        try:
            res = minimize(
                _neg_smooth_ratio,
                params,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': max_iters, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            h_new = np.exp(res.x)
            h_new = h_new / np.sum(h_new)
            sc = compute_ratio(h_new)
            if sc > best_score:
                best_score = sc
                best_h = h_new.copy()
                params = res.x
        except Exception:
            pass
    return best_score, best_h

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin under 150s kill

    best_score = 0.0
    best_full = None

    def consider(score, full):
        nonlocal best_score, best_full
        if score > best_score:
            best_score = score
            best_full = full.copy()

    # --- Generate candidate Sidon supports of various sizes ---
    supports = []
    for max_size in (10, 11, 12, 13, 14, 15, 16, 17):
        if time.time() > deadline - 25:
            break
        # number of seeds scales down for larger sizes (harder to find)
        n_seeds = max(3, 40 - max_size)
        cand = generate_sidon_sets(max_size, n_seeds)
        supports.extend(cand)

    # Deduplicate supports by frozenset
    uniq = {}
    for s in supports:
        key = frozenset(s)
        if key not in uniq:
            uniq[key] = sorted(s)
    supports = list(uniq.values())

    # Also add a few random sparse supports (not necessarily Sidon) for variety
    rng = np.random.default_rng(7)
    for nz in (8, 10, 12):
        for _ in range(3):
            idx = sorted(rng.choice(N, size=nz, replace=False).tolist())
            if frozenset(idx) not in uniq:
                supports.append(idx)

    # --- Optimize weights on each support (budget-aware) ---
    # Sort by size desc (larger Sidon supports give stronger structural bound)
    supports.sort(key=lambda s: -len(s))

    # Reserve time for final polish
    polish_reserve = 30.0
    per_support_budget = 6.0
    for support in supports:
        if time.time() > deadline - polish_reserve:
            break
        local_deadline = min(deadline - polish_reserve, time.time() + per_support_budget)
        try:
            score, w = optimize_weights_for_support(support, local_deadline)
            full = _support_full(support, w)
            # quick local polish
            score2, full2 = local_search_polish(full, support, local_deadline, steps=120, scale=0.04)
            if score2 > score:
                score, full = score2, full2
            consider(score, full)
        except Exception:
            continue

    # Fallback: uniform
    if best_full is None:
        best_full = np.ones(N) / N
        best_score = compute_ratio(best_full)

    # --- Final high-precision gradient polish from best found ---
    if time.time() < deadline - 4:
        try:
            score, h_polish = gradient_polish(best_full, deadline)
            consider(score, h_polish)
        except Exception:
            pass

    # Ensure nonnegativity
    best_full = np.maximum(best_full, 0.0)
    s = np.sum(best_full)
    if s > 0:
        best_full = best_full / s

    return best_full.tolist()
