import time
import random
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _conflict_count(vec, vec_set, n):
    """Count how many pairs in vec_set would form a cap set violation with vec."""
    count = 0
    vec_list = list(vec_set)
    for i, a in enumerate(vec_list):
        # Check a + a + vec = 0 mod 3 (i.e., 2a + vec = 0)
        ok = True
        for k in range(n):
            if (2 * a[k] + vec[k]) % 3 != 0:
                ok = False
                break
        if ok:
            count += 1
    for i in range(len(vec_list)):
        for j in range(i + 1, len(vec_list)):
            a, b = vec_list[i], vec_list[j]
            ok = True
            for k in range(n):
                if (a[k] + b[k] + vec[k]) % 3 != 0:
                    ok = False
                    break
            if ok:
                count += 1
    return count

def _can_add_fast(vec, vec_set, n):
    """Check if vec can be added to vec_set while maintaining cap set property."""
    vec_list = list(vec_set)
    for a in vec_list:
        ok = True
        for k in range(n):
            if (2 * a[k] + vec[k]) % 3 != 0:
                ok = False
                break
        if ok:
            return False
    for i in range(len(vec_list)):
        for j in range(i + 1, len(vec_list)):
            a, b = vec_list[i], vec_list[j]
            ok = True
            for k in range(n):
                if (a[k] + b[k] + vec[k]) % 3 != 0:
                    ok = False
                    break
            if ok:
                return False
    return True

def _greedy_dynamic(n, seed=42):
    """Build cap set using dynamic conflict-based greedy selection."""
    rng = random.Random(seed)
    cap_set = []
    vec_set = set()
    
    # Pre-generate all vectors
    all_vecs = list(itertools.product(range(3), repeat=n))
    rng.shuffle(all_vecs)
    
    # Greedily add vectors, preferring those with fewer conflicts
    remaining = list(all_vecs)
    
    while remaining and time.time() < _DEADLINE - 10.0:
        if len(cap_set) >= 3**n // 2:  # sanity bound
            break
        
        # Find the vector with minimum conflict count (among first batch for speed)
        batch_size = min(500, len(remaining))
        batch = remaining[:batch_size]
        
        best_vec = None
        best_conflicts = float('inf')
        
        for vec in batch:
            if vec in vec_set:
                continue
            conflicts = _conflict_count(vec, vec_set, n)
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_vec = vec
                if conflicts == 0:
                    break
        
        if best_vec is None:
            # No vector in batch has 0 conflicts; try adding any with min conflicts
            # But only if it doesn't violate cap set property
            for vec in batch:
                if vec in vec_set:
                    continue
                if _can_add_fast(vec, vec_set, n):
                    best_vec = vec
                    break
        
        if best_vec is not None and _can_add_fast(best_vec, vec_set, n):
            cap_set.append(best_vec)
            vec_set.add(best_vec)
            remaining.remove(best_vec)
        else:
            # Remove the worst vectors from remaining to avoid re-checking
            for vec in batch[:10]:
                if vec in remaining:
                    remaining.remove(vec)
    
    return cap_set

def _local_search(cap_set, n, rng, time_budget=30.0):
    """Improve cap set via local search with swaps."""
    if not cap_set:
        return cap_set
    
    vec_set = set(cap_set)
    best_cap_set = list(cap_set)
    best_size = len(cap_set)
    start_time = time.time()
    
    while time.time() - start_time < time_budget and time.time() < _DEADLINE - 15.0:
        if len(cap_set) < 1:
            break
        
        idx = rng.randint(0, len(cap_set) - 1)
        removed = cap_set[idx]
        new_vec_set = vec_set - {removed}
        
        # Try random vectors to replace
        found = False
        for _ in range(300):
            vec = tuple(rng.randint(0, 2) for _ in range(n))
            if vec == removed or vec in new_vec_set:
                continue
            if _can_add_fast(vec, new_vec_set, n):
                cap_set[idx] = vec
                new_vec_set.add(vec)
                found = True
                if len(new_vec_set) > best_size:
                    best_size = len(new_vec_set)
                    best_cap_set = list(new_vec_set)
                break
        
        if not found:
            vec_set.add(removed)
    
    return best_cap_set

def _multi_restart_greedy(n):
    """Try multiple greedy restarts with different seeds and local search."""
    best_cap_set = []
    best_size = 0
    
    num_restarts = max(3, int(120 // max(1, n)))
    
    for seed in range(num_restarts):
        if time.time() > _DEADLINE - 20.0:
            break
        
        cap_set = _greedy_dynamic(n, seed=seed * 1000 + 42)
        
        if len(cap_set) > best_size:
            best_size = len(cap_set)
            best_cap_set = list(cap_set)
        
        # Local search to improve
        rng = random.Random(seed * 777 + 13)
        time_left = _DEADLINE - time.time() - 15.0
        if time_left > 10.0:
            improved = _local_search(cap_set, n, rng, time_budget=min(20.0, time_left))
            if len(improved) > best_size:
                best_size = len(improved)
                best_cap_set = list(improved)
    
    return best_cap_set

def _precompute_best(n):
    """Precompute the best cap set found for dimension n."""
    return _multi_restart_greedy(n)

# Cache for precomputed results
_best_cache = {}

def priority(vector, n):
    # Use a hybrid approach: 
    # 1. For small n, precompute optimal/near-optimal cap set and rank by membership
    # 2. For larger n, use dynamic structural scoring
    
    if n in _best_cache:
        best_set = _best_cache[n]
        # Vectors in the precomputed best set get highest priority
        if tuple(vector) in best_set:
            # Rank within the set by a hash for uniqueness
            h = 0
            for i, v in enumerate(vector):
                h = (h * 31 + v) & 0x7FFFFFFF
            return 10000.0 + h / 1000.0
        else:
            # Vectors not in best set get lower priority based on distance
            # Count how many vectors in best_set would conflict with this vector
            conflicts = 0
            best_list = list(best_set)
            for a in best_list:
                ok = True
                for k in range(n):
                    if (2 * a[k] + vector[k]) % 3 != 0:
                        ok = False
                        break
                if ok:
                    conflicts += 1
            h = 0
            for i, v in enumerate(vector):
                h = (h * 31 + v) & 0x7FFFFFFF
            return 1000.0 - conflicts * 10.0 + h / 10000.0
    
    # For n where we haven't precomputed, use structural heuristic
    # This path is used during the first call when we're still computing
    
    # Trigger precomputation in background (but we need to return something now)
    # Use a quick heuristic as fallback
    c0 = sum(1 for v in vector if v == 0)
    c1 = sum(1 for v in vector if v == 1)
    c2 = sum(1 for v in vector if v == 2)
    imbalance = abs(c1 - c2)
    
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v) & 0x7FFFFFFF
    
    structural = -3.0 * c0 - 1.5 * imbalance + 0.5 * min(c1, c2)
    result = (h / 2147483647.0) * 100.0 + structural
    return result
