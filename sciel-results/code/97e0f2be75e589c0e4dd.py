"""Find maximum cardinality dissociated reciprocal subset of [1,22].

Structural approach:
- Represent reciprocal sums as exact integers via LCM(1..22).
- Use fast collision check with sorted sums.
- Search via diverse greedy construction + aggressive local search.
"""
import time
import random

# LCM of 1..22
_LCM = 232792560

# Large prime for modular filtering
_MOD_PRIME = 1000000007

# Precompute modular weights
_MOD_WEIGHTS = {x: (_LCM // x) % _MOD_PRIME for x in range(1, 23)}

def _check_add_fast(sums_set, v):
    """Check if adding weight v creates a collision. Returns True if valid."""
    # For each existing sum s, check if s+v is also in sums_set
    # This is equivalent to: does the set intersect with itself shifted by v?
    for s in sums_set:
        if (s + v) in sums_set:
            return False
    return True

def _check_add_mod(mod_sums_set, v_mod):
    """Fast modular pre-check: if collision mod P, definitely collision in integers."""
    for s in mod_sums_set:
        if (s + v_mod) % _MOD_PRIME in mod_sums_set:
            return False
    return True

def _build_sums_set(subset):
    """Build the set of all subset sums for the given subset."""
    sums = {0}
    for x in subset:
        v = _LCM // x
        sums |= {s + v for s in sums}
    return sums

def _build_mod_sums_set(subset):
    """Build the set of all modular subset sums for the given subset."""
    sums = {0}
    for x in subset:
        v = _MOD_WEIGHTS[x]
        sums |= {(s + v) % _MOD_PRIME for s in sums}
    return sums

def _is_dissociated(subset):
    """Check if all reciprocal subset sums are distinct."""
    if not subset:
        return True
    sums = {0}
    for x in subset:
        v = _LCM // x
        new_sums = {s + v for s in sums}
        if new_sums & sums:
            return False
        sums |= new_sums
    return True

def _is_dissociated_fast(subset):
    """Fast check using modular pre-filter then exact check."""
    if not subset:
        return True
    # Modular pre-check
    mod_sums = {0}
    for x in subset:
        v = _MOD_WEIGHTS[x]
        new_mod = {(s + v) % _MOD_PRIME for s in mod_sums}
        if new_mod & mod_sums:
            return False
        mod_sums |= new_mod
    # Exact check
    return _is_dissociated(subset)

def _greedy_build(order, deadline):
    """Build a dissociated set greedily following the given order."""
    current = []
    sums = {0}
    mod_sums = {0}
    for x in order:
        if time.time() > deadline:
            break
        v = _LCM // x
        v_mod = _MOD_WEIGHTS[x]
        # Fast modular pre-check
        if not _check_add_mod(mod_sums, v_mod):
            continue
        # Exact check
        if _check_add_fast(sums, v):
            current.append(x)
            sums |= {s + v for s in sums}
            mod_sums |= {(s + v_mod) % _MOD_PRIME for s in mod_sums}
    return current, sums

def _local_search_add(subset, sums, candidates, deadline):
    """Try to add elements from candidates to subset."""
    mod_sums = _build_mod_sums_set(subset)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates:
            if x in subset:
                continue
            v = _LCM // x
            v_mod = _MOD_WEIGHTS[x]
            # Fast modular pre-check
            if not _check_add_mod(mod_sums, v_mod):
                continue
            # Exact check
            if _check_add_fast(sums, v):
                subset.append(x)
                sums |= {s + v for s in sums}
                mod_sums |= {(s + v_mod) % _MOD_PRIME for s in mod_sums}
                improved = True
                break
    return subset, sums

def _local_search_swap(subset, sums, universe, deadline):
    """Try swap moves to improve the set."""
    best_subset = subset[:]
    best_sums = sums
    best_size = len(subset)
    
    subset_set = set(subset)
    
    # Try all possible single swaps
    for i, out_elem in enumerate(subset):
        if time.time() > deadline:
            break
        for in_elem in universe:
            if in_elem in subset_set:
                continue
            
            # Build new set
            new_subset = subset[:i] + subset[i+1:] + [in_elem]
            
            if _is_dissociated_fast(new_subset):
                new_sums = _build_sums_set(new_subset)
                if len(new_subset) > best_size:
                    best_subset = new_subset
                    best_sums = new_sums
                    best_size = len(new_subset)
                    subset_set = set(best_subset)
    
    return best_subset, best_sums, best_size

def _multi_swap_search(subset, sums, universe, deadline):
    """Try multiple consecutive swaps to escape local optima."""
    best_subset = subset[:]
    best_sums = sums
    best_size = len(subset)
    current_subset = subset[:]
    current_sums = sums
    current_size = len(subset)
    current_set = set(current_subset)
    
    max_iter = 500
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try a random swap
        if not current_subset:
            break
        
        out_idx = random.randint(0, len(current_subset) - 1)
        available = [x for x in universe if x not in current_set]
        if not available:
            break
        
        in_elem = random.choice(available)
        new_subset = current_subset[:out_idx] + current_subset[out_idx+1:] + [in_elem]
        
        if _is_dissociated_fast(new_subset):
            new_size = len(new_subset)
            # Accept if better or with some probability for diversification
            if new_size >= current_size or random.random() < 0.15:
                current_subset = new_subset
                current_set = set(current_subset)
                current_sums = _build_sums_set(current_subset)
                current_size = new_size
                
                if current_size > best_size:
                    best_subset = current_subset[:]
                    best_sums = current_sums
                    best_size = current_size
    
    return best_subset, best_sums, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    universe = list(range(1, 23))
    random.seed(42)
    
    # Start with a known good seed
    best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_sums = _build_sums_set(best_subset)
    best_size = len(best_subset)
    
    # Verify seed
    if not _is_dissociated_fast(best_subset):
        import itertools
        for size in range(9, 0, -1):
            found = False
            for combo in itertools.combinations(range(1, 23), size):
                if _is_dissociated_fast(list(combo)):
                    best_subset = list(combo)
                    best_sums = _build_sums_set(best_subset)
                    best_size = size
                    found = True
                    break
            if found:
                break
    
    # Strategy 1: Greedy with diverse orderings
    for attempt in range(800):
        if time.time() > deadline:
            break
        
        shuffled = universe[:]
        random.shuffle(shuffled)
        
        # Various biases
        mode = attempt % 6
        if mode == 0:
            shuffled.sort(key=lambda x: -x)  # Descending
        elif mode == 1:
            shuffled.sort(key=lambda x: x)   # Ascending
        elif mode == 2:
            shuffled.sort(key=lambda x: _LCM // x)  # By weight ascending
        elif mode == 3:
            shuffled.sort(key=lambda x: -(_LCM // x))  # By weight descending
        elif mode == 4:
            shuffled.sort(key=lambda x: _MOD_WEIGHTS[x])  # By mod weight ascending
        # else: already shuffled
        
        current, cur_sums = _greedy_build(shuffled, deadline)
        
        # Local search: add
        current, cur_sums = _local_search_add(current, cur_sums, universe, deadline)
        
        if len(current) > best_size:
            best_subset = current[:]
            best_sums = cur_sums
            best_size = len(current)
    
    # Strategy 2: Swap-based local search on best
    if time.time() < deadline:
        best_subset, best_sums, best_size = _local_search_swap(best_subset, best_sums, universe, deadline)
    
    # Strategy 3: Multi-swap search for diversification
    if time.time() < deadline:
        best_subset, best_sums, best_size = _multi_swap_search(best_subset, best_sums, universe, deadline)
    
    # Final: try to add more
    if time.time() < deadline:
        best_subset, best_sums = _local_search_add(best_subset, best_sums, universe, deadline)
        best_size = len(best_subset)
    
    # Second pass: more greedy attempts with different random seeds
    if time.time() < deadline:
        for attempt in range(300):
            if time.time() > deadline:
                break
            random.seed(42 + attempt)
            shuffled = universe[:]
            random.shuffle(shuffled)
            
            current, cur_sums = _greedy_build(shuffled, deadline)
            current, cur_sums = _local_search_add(current, cur_sums, universe, deadline)
            
            if len(current) > best_size:
                best_subset = current[:]
                best_sums = cur_sums
                best_size = len(current)
    
    return best_subset
