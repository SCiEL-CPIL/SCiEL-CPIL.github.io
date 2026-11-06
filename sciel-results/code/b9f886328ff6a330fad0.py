"""Additive bases with few representations - structural redesign.

Instead of the previous ILP / hill-climb hybrid, this program uses:
1. A fast incremental count-update for single-element add/remove.
2. A multi-restart local search with two move types:
     a) swap (remove one, add one)
     b) single flip (add or remove)
   Accepting moves that lower peak, then variance, then size.
3. A 'peak-driven' repair: if a move breaks coverage, we repair by adding
   elements that cover the most uncovered targets with minimal peak increase.
4. Time-budgeted loop with explicit deadline guard.
"""
import time
import random
import numpy as np

_N = 128
_LEN = 129  # indices 0..128
_T0 = 128
_T1 = 256
_WIN = _T1 - _T0 + 1  # 129 targets

_START = time.time()
_DEADLINE = 87.0  # safety margin before the 90s kill


def _elapsed():
    return time.time() - _START


def _compute_counts(x):
    """Compute unordered pair counts for targets [128, 256] via convolution."""
    y = np.convolve(x.astype(np.int64), x.astype(np.int64), mode="full")
    w = y[_T0:_T1 + 1].copy()
    ks = np.arange(_T0, _T1 + 1)
    is_even = (ks % 2 == 0)
    mids = ks // 2
    diag_corr = np.where(is_even, x[mids], 0)
    u_counts = (w + diag_corr) // 2
    return u_counts


def _valid(u_counts):
    return bool(np.all(u_counts > 0))


def _peak(u_counts):
    return int(u_counts.max())


def _variance(u_counts):
    return float(np.var(u_counts))


def _objective(u_counts, x):
    """Composite objective: (peak, variance, size). Lower is better."""
    if not _valid(u_counts):
        return (10**9, 10**9, 10**9)
    return (_peak(u_counts), _variance(u_counts), int(np.sum(x)))


def _delta_add(x, u, c):
    """
    Return the delta in unordered counts when adding element c (x[c] was 0).
    Only returns the delta array for targets [128, 256].
    """
    delta = np.zeros(_WIN, dtype=np.int64)
    # For each target k in [128, 256]:
    #   if k - c is in [0, 128] and x[k-c] == 1, then unordered count increases by 1
    #   if 2*c == k, unordered count increases by 1 (the pair (c,c))
    # Note: if k - c == c, i.e., k == 2c, we must not double count.
    # The pair (c, k-c) with k-c == c is the same as (c, c).
    # In unordered counting, (c, c) is one pair.
    # So: if k - c in [0,128] and x[k-c]==1 and k-c != c: +1
    #     if 2*c == k: +1
    # But if k-c == c and x[c] was 0 (before adding), then x[k-c] is 0, so no +1 from first condition.
    # After adding c, x[c] becomes 1, but we're computing delta for the transition.
    # The new pairs formed are (c, a) for a in A_old, and (c, c).
    # For a in A_old, a != c (since c not in A_old).
    # So unordered pairs added: for each a in A_old with a + c in [128,256], one pair (c,a) (unordered, a!=c).
    # Plus (c,c) if 2c in [128,256].
    
    # Vectorized:
    # k = c + a, a in [0, 128], x[a] == 1
    # k in [c, c+128]
    # Intersection with [128, 256]:
    k_min = max(_T0, c)
    k_max = min(_T1, c + _N)
    if k_min <= k_max:
        ks = np.arange(k_min, k_max + 1)
        partners = ks - c
        # partners in [0, 128]
        mask = x[partners] == 1
        idxs = ks - _T0
        delta[idxs[mask]] += 1
    
    # Diagonal: (c, c)
    if _T0 <= 2 * c <= _T1:
        delta[2 * c - _T0] += 1
    
    return delta


def _delta_remove(x, u, c):
    """
    Return the delta in unordered counts when removing element c (x[c] was 1).
    This is the negative of _delta_add for the state BEFORE removal,
    but careful: when removing c, the pairs (c, a) for a in A_old \ {c} are lost,
    and (c, c) is lost if present.
    """
    delta = np.zeros(_WIN, dtype=np.int64)
    k_min = max(_T0, c)
    k_max = min(_T1, c + _N)
    if k_min <= k_max:
        ks = np.arange(k_min, k_max + 1)
        partners = ks - c
        mask = x[partners] == 1
        idxs = ks - _T0
        delta[idxs[mask]] -= 1
    
    if _T0 <= 2 * c <= _T1:
        delta[2 * c - _T0] -= 1
    
    return delta


def _make_greedy_seed(rng):
    """Build a valid seed using greedy coverage with load balancing."""
    x = np.zeros(_LEN, dtype=np.int64)
    u = _compute_counts(x)
    
    candidates = np.arange(_LEN)
    rng.shuffle(candidates)
    
    while not _valid(u):
        uncovered = np.where(u == 0)[0] + _T0
        if len(uncovered) == 0:
            break
        
        best_c = None
        best_score = None
        
        for c in candidates:
            if x[c] == 1:
                continue
            
            delta = _delta_add(x, u, c)
            new_u = u + delta
            
            if not _valid(new_u):
                continue
            
            peak_new = _peak(new_u)
            var_new = _variance(new_u)
            score = (peak_new, var_new)
            
            if best_score is None or score < best_score:
                best_score = score
                best_c = c
        
        if best_c is None:
            # Fallback: pick any candidate that covers at least one uncovered target
            for c in candidates:
                if x[c] == 1:
                    continue
                delta = _delta_add(x, u, c)
                if np.any(delta > 0):
                    best_c = c
                    break
        
        if best_c is None:
            best_c = candidates[0]
        
        x[best_c] = 1
        u = _compute_counts(x)
    
    return x


def _local_search(x, rng, max_time, restart_seed=None):
    """
    Local search with swap and flip moves.
    Objective: minimize (peak, variance, size).
    """
    u = _compute_counts(x)
    if not _valid(u):
        return x, u
    
    best_x = x.copy()
    best_u = u.copy()
    best_obj = _objective(u, x)
    
    in_A = np.where(x == 1)[0]
    not_in_A = np.where(x == 0)[0]
    
    while _elapsed() < max_time:
        improved = False
        n_swaps = 200
        n_flips = 100
        
        # Try swaps
        for _ in range(n_swaps):
            if _elapsed() > max_time:
                break
            if len(in_A) == 0 or len(not_in_A) == 0:
                break
            
            i = rng.choice(in_A)
            j = rng.choice(not_in_A)
            
            # Remove i, add j
            delta_remove = _delta_remove(x, u, i)
            x[j] = 1
            delta_add = _delta_add(x, u, j)
            x[j] = 0
            
            new_u = u + delta_remove + delta_add
            
            if not _valid(new_u):
                continue
            
            new_obj = _objective(new_u, x - (x[i] - 1) + (x[j] + 1))  # approximate size
            # Actually, size change: remove i, add j, so size stays same.
            new_obj = (_peak(new_u), _variance(new_u), int(np.sum(x)))
            
            if new_obj < best_obj:
                x[i] = 0
                x[j] = 1
                u = new_u
                best_x = x.copy()
                best_u = u.copy()
                best_obj = new_obj
                in_A = np.where(x == 1)[0]
                not_in_A = np.where(x == 0)[0]
                improved = True
        
        # Try flips (add only, since removing might break coverage)
        for _ in range(n_flips):
            if _elapsed() > max_time:
                break
            if len(not_in_A) == 0:
                break
            
            j = rng.choice(not_in_A)
            delta = _delta_add(x, u, j)
            new_u = u + delta
            
            if not _valid(new_u):
                continue
            
            new_obj = (_peak(new_u), _variance(new_u), int(np.sum(x) + 1))
            
            if new_obj < best_obj:
                x[j] = 1
                u = new_u
                best_x = x.copy()
                best_u = u.copy()
                best_obj = new_obj
                in_A = np.where(x == 1)[0]
                not_in_A = np.where(x == 0)[0]
                improved = True
        
        # Try removals (only if coverage remains)
        for _ in range(n_flips // 2):
            if _elapsed() > max_time:
                break
            if len(in_A) == 0:
                break
            
            i = rng.choice(in_A)
            delta = _delta_remove(x, u, i)
            new_u = u + delta
            
            if not _valid(new_u):
                continue
            
            new_obj = (_peak(new_u), _variance(new_u), int(np.sum(x) - 1))
            
            if new_obj < best_obj:
                x[i] = 0
                u = new_u
                best_x = x.copy()
                best_u = u.copy()
                best_obj = new_obj
                in_A = np.where(x == 1)[0]
                not_in_A = np.where(x == 0)[0]
                improved = True
        
        if not improved:
            break
    
    return best_x, best_u


def construct():
    global _START
    _START = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Get a valid seed
    x = _make_greedy_seed(rng)
    u = _compute_counts(x)
    if not _valid(u):
        # Fallback: full set
        x = np.ones(_LEN, dtype=np.int64)
        u = _compute_counts(x)
    
    best_x = x.copy()
    best_u = u.copy()
    best_obj = _objective(u, x)
    
    # Phase 2: Multi-restart local search
    n_restarts = 5
    time_per_restart = (_DEADLINE - _elapsed() - 5) / n_restarts
    
    for r in range(n_restarts):
        if _elapsed() > _DEADLINE - 5:
            break
        
        if r == 0:
            x_cur = best_x.copy()
        else:
            # Perturb: random swaps
            x_cur = best_x.copy()
            u_cur = best_u.copy()
            n_perturb = rng.randint(3, 10)
            in_A = np.where(x_cur == 1)[0]
            not_in_A = np.where(x_cur == 0)[0]
            for _ in range(n_perturb):
                if len(in_A) == 0 or len(not_in_A) == 0:
                    break
                i = rng.choice(in_A)
                j = rng.choice(not_in_A)
                x_cur[i] = 0
                x_cur[j] = 1
                in_A = np.where(x_cur == 1)[0]
                not_in_A = np.where(x_cur == 0)[0]
            u_cur = _compute_counts(x_cur)
            if not _valid(u_cur):
                # Repair: add elements to cover uncovered targets
                while not _valid(u_cur) and _elapsed() < _DEADLINE - 5:
                    uncovered = np.where(u_cur == 0)[0] + _T0
                    best_c = None
                    best_gain = -1
                    for c in not_in_A:
                        delta = _delta_add(x_cur, u_cur, c)
                        gain = np.sum((delta > 0) & (u_cur == 0))
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
                    if best_c is None or best_gain <= 0:
                        break
                    x_cur[best_c] = 1
                    u_cur = _compute_counts(x_cur)
                    in_A = np.where(x_cur == 1)[0]
                    not_in_A = np.where(x_cur == 0)[0]
                if not _valid(u_cur):
                    continue
        
        max_time = min(_DEADLINE - 5 - _elapsed(), time_per_restart)
        if max_time <= 0:
            break
        
        x_ls, u_ls = _local_search(x_cur, rng, max_time)
        
        if _valid(u_ls):
            obj_ls = _objective(u_ls, x_ls)
            if obj_ls < best_obj:
                best_x = x_ls.copy()
                best_u = u_ls.copy()
                best_obj = obj_ls
    
    # Phase 3: Final polish
    if _elapsed() < _DEADLINE - 2:
        x_ls, u_ls = _local_search(best_x, rng, _DEADLINE - 2 - _elapsed())
        if _valid(u_ls):
            obj_ls = _objective(u_ls, x_ls)
            if obj_ls < best_obj:
                best_x = x_ls.copy()
                best_u = u_ls.copy()
    
    # Convert to sorted list
    result = [int(i) for i in np.where(best_x == 1)[0]]
    return result
