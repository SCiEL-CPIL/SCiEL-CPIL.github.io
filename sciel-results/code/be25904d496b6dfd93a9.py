import time
import random
import sys

# Global cache for priority sets
_PRIORITY_CACHE = {}

# Deadline guard
_START_TIME = time.time()
DEADLINE = _START_TIME + 310.0  # Leave 20s safety margin before 330s kill


def _int_to_tuple(idx, n):
    """Convert an integer index to a tuple of digits in base 3."""
    v = []
    x = idx
    for _ in range(n):
        v.append(x % 3)
        x //= 3
    return tuple(v)


def _tuple_to_int(v):
    """Convert a tuple of digits in base 3 to an integer index."""
    idx = 0
    for i, d in enumerate(reversed(v)):
        idx += d * (3 ** i)
    return idx


def _reflection_index(v_idx, a_idx, n):
    """
    Compute the index of w where w_i = (2*v_i - a_i) mod 3,
    given integer indices v_idx and a_idx in {0,1,2}^n.
    """
    w_idx = 0
    power = 1
    vi = v_idx
    ai = a_idx
    for _ in range(n):
        vd = vi % 3
        ad = ai % 3
        wd = (2 * vd - ad) % 3
        w_idx += wd * power
        vi //= 3
        ai //= 3
        power *= 3
    return w_idx


def _greedy_cap_set_fast(n, seed, time_limit):
    """
    Fast greedy cap set using integer indices and direct reflection computation.
    """
    start = time.time()
    total = 3 ** n
    
    # Shuffle indices
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    
    S = set()
    forbidden = set()
    
    for idx in indices:
        if time.time() - start > time_limit:
            break
        if idx in forbidden:
            continue
        
        S.add(idx)
        
        # For each existing vector a in S, compute the forbidden reflection
        for a_idx in S:
            if a_idx == idx:
                continue
            w_idx = _reflection_index(idx, a_idx, n)
            forbidden.add(w_idx)
    
    return S


def _greedy_with_priority(n, seed, time_limit, prio_func):
    """
    Greedy cap set with a custom priority function.
    prio_func(idx, n) -> float (higher priority = processed first)
    """
    start = time.time()
    total = 3 ** n
    
    # Create list of (priority, index) and sort
    items = [(prio_func(i, n), i) for i in range(total)]
    items.sort(key=lambda x: -x[0])
    order = [item[1] for item in items]
    
    S = set()
    forbidden = set()
    
    for idx in order:
        if time.time() - start > time_limit:
            break
        if idx in forbidden:
            continue
        
        S.add(idx)
        
        for a_idx in S:
            if a_idx == idx:
                continue
            w_idx = _reflection_index(idx, a_idx, n)
            forbidden.add(w_idx)
    
    return S


def _prio_zero_count(idx, n):
    """Priority: more zeros first."""
    count = 0
    x = idx
    for _ in range(n):
        if x % 3 == 0:
            count += 1
        x //= 3
    return count


def _prio_nonzero_count(idx, n):
    """Priority: fewer non-zeros first."""
    count = 0
    x = idx
    for _ in range(n):
        if x % 3 != 0:
            count += 1
        x //= 3
    return -count  # Negate so fewer non-zeros have higher priority


def _prio_coord_sum(idx, n):
    """Priority: lower coordinate sum first."""
    s = 0
    x = idx
    for _ in range(n):
        s += x % 3
        x //= 3
    return -s


def _prio_hybrid(idx, n):
    """Hybrid: zeros * 2 - nonzeros."""
    z = 0
    nz = 0
    x = idx
    for _ in range(n):
        d = x % 3
        if d == 0:
            z += 1
        else:
            nz += 1
        x //= 3
    return z * 2 - nz


def priority(vector, n):
    if n not in _PRIORITY_CACHE:
        # Perform search for this n
        time_limit = 250.0  # Leave time for overhead
        
        best_set = set()
        best_size = 0
        
        start_search = time.time()
        
        # Phase 1: Fast random greedy with many seeds
        random_seeds = list(range(50))  # 50 random seeds
        for seed in random_seeds:
            if time.time() - start_search > time_limit:
                break
            remaining = time_limit - (time.time() - start_search)
            if remaining < 3.0:
                break
            
            # Adaptive time per run
            runs_done = random_seeds.index(seed) + 1
            per_run_time = min(5.0, remaining * 0.8 / max(1, len(random_seeds) - runs_done + 1))
            per_run_time = max(1.0, per_run_time)
            
            S = _greedy_cap_set_fast(n, seed, per_run_time)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
        
        # Phase 2: Try structured priorities with the best remaining time
        remaining = time_limit - (time.time() - start_search)
        if remaining > 10.0:
            prio_funcs = [
                ("zero_count", _prio_zero_count),
                ("nonzero_count", _prio_nonzero_count),
                ("coord_sum", _prio_coord_sum),
                ("hybrid", _prio_hybrid),
            ]
            
            for name, pf in prio_funcs:
                if time.time() - start_search > time_limit:
                    break
                remaining = time_limit - (time.time() - start_search)
                if remaining < 5.0:
                    break
                
                # Try a few seeds for each priority mode
                for seed in [42, 123, 456]:
                    if time.time() - start_search > time_limit:
                        break
                    remaining = time_limit - (time.time() - start_search)
                    if remaining < 3.0:
                        break
                    
                    per_run_time = min(8.0, remaining * 0.7)
                    S = _greedy_with_priority(n, seed, per_run_time, pf)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S.copy()
        
        # Phase 3: Local improvement - try to add more vectors to the best set
        remaining = time_limit - (time.time() - start_search)
        if remaining > 5.0 and best_size > 0:
            # Try random additions to the current best set
            total = 3 ** n
            rng = random.Random(999)
            candidates = list(range(total))
            rng.shuffle(candidates)
            
            S = best_set.copy()
            forbidden = set()
            # Rebuild forbidden set
            for idx in S:
                for a_idx in S:
                    if a_idx == idx:
                        continue
                    w_idx = _reflection_index(idx, a_idx, n)
                    forbidden.add(w_idx)
            
            added = 0
            for idx in candidates:
                if time.time() - start_search > time_limit:
                    break
                if idx in S or idx in forbidden:
                    continue
                
                # Check if adding idx keeps cap set property
                # idx is not forbidden, so it should be safe
                S.add(idx)
                for a_idx in S:
                    if a_idx == idx:
                        continue
                    w_idx = _reflection_index(idx, a_idx, n)
                    forbidden.add(w_idx)
                added += 1
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
        
        # Convert best_set (indices) to set of tuples
        best_tuples = set()
        for idx in best_set:
            best_tuples.add(_int_to_tuple(idx, n))
        
        _PRIORITY_CACHE[n] = best_tuples
    
    return 1.0 if vector in _PRIORITY_CACHE[n] else 0.0
