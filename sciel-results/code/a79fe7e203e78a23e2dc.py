import random
import time

N = 256

def _is_sidon(S):
    S = sorted(S)
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(S):
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    return sums

def _is_maximal_fast(S, sums):
    """Check maximality: every missing element must be blocked."""
    S_set = set(S)
    S_list = list(S)
    n = len(S_list)
    for x in range(1, N+1):
        if x in S_set:
            continue
        if 2*x in sums:
            continue
        blocked = False
        for s in S_list:
            if x + s in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_maximal():
    selected = []
    sums = set()
    for x in range(1, N+1):
        added = [x + a for a in selected] + [2*x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _reduce(S, deadline=None):
    """Try to remove elements while maintaining maximality."""
    S = sorted(S)
    changed = True
    while changed:
        changed = False
        if deadline and time.time() > deadline:
            break
        for i in range(len(S)):
            if deadline and time.time() > deadline:
                break
            candidate = S[:i] + S[i+1:]
            if not candidate:
                continue
            if not _is_sidon(candidate):
                continue
            sums = _get_sums(candidate)
            if _is_maximal_fast(candidate, sums):
                S = candidate
                changed = True
                break
    return S

def _local_search_swap(S, sums, deadline):
    """Try swap moves: remove one element, add another, check if still maximal and smaller or same size."""
    S = sorted(S)
    S_set = set(S)
    for i in range(len(S)):
        if time.time() > deadline:
            break
        S_without = S[:i] + S[i+1:]
        if not S_without:
            continue
        if not _is_sidon(S_without):
            continue
        sums_without = _get_sums(S_without)
        S_without_set = set(S_without)
        
        for x in range(1, N+1):
            if x in S_without_set:
                continue
            added = [x + a for a in S_without] + [2*x]
            if len(set(added)) != len(added):
                continue
            if sums_without.intersection(added):
                continue
            new_S = S_without + [x]
            new_S.sort()
            new_sums = sums_without | set(added)
            if _is_maximal_fast(new_S, new_sums):
                return new_S
    return None

def _random_sidon_set(seed_offset=0):
    """Build a random Sidon set."""
    S = set()
    sums = set()
    candidates = list(range(1, N+1))
    random.shuffle(candidates)
    for x in candidates:
        added = [x + a for a in S] + [2*x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            S.add(x)
            sums.update(added)
    return sorted(S), sums

def _conflict_graph():
    """Build conflict graph: edge (a,b) means a+b collision with some pair.
    Two elements conflict if they can't coexist in a Sidon set.
    Actually, for Sidon sets, the conflict is more complex. Let's use a different approach:
    For each pair (a,b), check if adding both to a partial set creates conflicts.
    
    Better approach: Build a graph where vertex x is "dominated" by set S if 
    x cannot be added to S (i.e., S is maximal w.r.t. x).
    
    For the dominating set approach:
    - A set S is a valid Sidon set
    - S is inclusion-maximal iff for every x not in S, x conflicts with S
    - We want minimum size S that is both Sidon and maximal
    
    Conflict: x conflicts with S if 2x is in sums(S) OR exists s in S such that x+s is in sums(S)
    """
    pass

def _dominating_set_greedy(deadline=None):
    """Find a small Sidon set that dominates all other vertices.
    
    Strategy: 
    1. Start with empty set
    2. Repeatedly add the element that "covers" (blocks) the most uncovered elements
    3. Ensure the result is a valid Sidon set
    4. The set is maximal when all elements are covered
    """
    S = set()
    sums = set()
    covered = set()  # Elements that are blocked by S
    
    # Precompute: for each pair (a,b), what sums they create
    # For each element x, determine which elements it blocks when added to S
    
    remaining = list(range(1, N+1))
    
    while time.time() < deadline:
        if not remaining:
            break
        
        # Find the element that blocks the most remaining elements
        best_x = None
        best_blocked_count = -1
        best_blocked = set()
        
        for x in remaining:
            # Check if x can be added to S
            added = [x + a for a in S] + [2*x]
            if len(set(added)) != len(added):
                continue
            if sums.intersection(added):
                continue
            
            # x can be added. How many remaining elements does it block?
            new_sums = sums | set(added)
            blocked = set()
            for y in remaining:
                if y == x:
                    continue
                # Check if y is blocked by S ∪ {x}
                if 2*y in new_sums:
                    blocked.add(y)
                    continue
                for s in S:
                    if y + s in new_sums:
                        blocked.add(y)
                        break
                if y not in blocked and y != x:
                    if x + y in new_sums:
                        blocked.add(y)
            
            if len(blocked) > best_blocked_count:
                best_blocked_count = len(blocked)
                best_x = x
                best_blocked = blocked
        
        if best_x is None:
            break
        
        # Add best_x to S
        added = [best_x + a for a in S] + [2*best_x]
        S.add(best_x)
        sums.update(added)
        covered.update(best_blocked)
        remaining.remove(best_x)
        
        # Check if all elements are covered
        if len(covered) + len(S) >= N:
            break
        
        # Check if we're making progress
        if len(remaining) == 0:
            break
    
    # Verify it's maximal
    S_list = sorted(S)
    if not _is_sidon(S_list):
        return None
    final_sums = _get_sums(S_list)
    if not _is_maximal_fast(S_list, final_sums):
        return None
    
    return S_list

def _greedy_dominating_with_random(deadline=None, seed=None):
    """Greedy dominating set with random tie-breaking."""
    if seed is not None:
        random.seed(seed)
    
    S = set()
    sums = set()
    covered = set()
    remaining = list(range(1, N+1))
    
    while time.time() < deadline:
        if not remaining:
            break
        
        best_x = None
        best_score = -1
        best_blocked = set()
        
        for x in remaining:
            added = [x + a for a in S] + [2*x]
            if len(set(added)) != len(added):
                continue
            if sums.intersection(added):
                continue
            
            new_sums = sums | set(added)
            blocked = set()
            for y in remaining:
                if y == x:
                    continue
                if 2*y in new_sums:
                    blocked.add(y)
                    continue
                is_blocked = False
                for s in S:
                    if y + s in new_sums:
                        is_blocked = True
                        break
                if not is_blocked and x + y in new_sums:
                    is_blocked = True
                if is_blocked:
                    blocked.add(y)
            
            # Add small random perturbation for diversity
            score = len(blocked) + random.random() * 0.1
            if score > best_score:
                best_score = score
                best_x = x
                best_blocked = blocked
        
        if best_x is None:
            break
        
        added = [best_x + a for a in S] + [2*best_x]
        S.add(best_x)
        sums.update(added)
        covered.update(best_blocked)
        remaining.remove(best_x)
        
        if len(covered) + len(S) >= N:
            break
        if not remaining:
            break
    
    S_list = sorted(S)
    if not S_list:
        return None
    if not _is_sidon(S_list):
        return None
    final_sums = _get_sums(S_list)
    if not _is_maximal_fast(S_list, final_sums):
        return None
    
    return S_list

def construct():
    deadline = time.time() + 25
    
    best = _greedy_maximal()
    best_size = len(best)
    
    # Try to reduce the greedy result
    reduced = _reduce(best, deadline)
    if len(reduced) < best_size:
        best = reduced
        best_size = len(best)
    
    # Try dominating set approach
    dom_set = _greedy_dominating_with_random(deadline, seed=42)
    if dom_set and len(dom_set) < best_size:
        best = dom_set
        best_size = len(best)
        reduced = _reduce(best, deadline)
        if len(reduced) < best_size:
            best = reduced
            best_size = len(best)
    
    # Try random approaches with multiple strategies
    random.seed(42)
    strategies = [
        'uniform',
        'low_first',
        'high_first',
        'mid_first',
        'prime_bias',
    ]
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        strategy = strategies[attempt % len(strategies)]
        
        if strategy == 'uniform':
            S, sums = _random_sidon_set()
        elif strategy == 'low_first':
            S = set()
            sums = set()
            candidates = list(range(1, N+1))
            for i in range(len(candidates) - 1, 0, -1):
                j = random.randint(0, min(i, i * 2 // 3))
                candidates[i], candidates[j] = candidates[j], candidates[i]
            for x in candidates:
                added = [x + a for a in S] + [2*x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    S.add(x)
                    sums.update(added)
            S = sorted(S)
        elif strategy == 'high_first':
            S = set()
            sums = set()
            candidates = list(range(1, N+1))
            for i in range(len(candidates) - 1, 0, -1):
                j = random.randint(max(0, i - i * 2 // 3), i)
                candidates[i], candidates[j] = candidates[j], candidates[i]
            for x in candidates:
                added = [x + a for a in S] + [2*x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    S.add(x)
                    sums.update(added)
            S = sorted(S)
        elif strategy == 'mid_first':
            S = set()
            sums = set()
            mid = N // 2
            candidates = [mid]
            for d in range(1, N):
                candidates.append(mid + d)
                if mid - d >= 1:
                    candidates.append(mid - d)
            random.shuffle(candidates[1:])
            for x in candidates:
                added = [x + a for a in S] + [2*x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    S.add(x)
                    sums.update(added)
            S = sorted(S)
        else:
            S = set()
            sums = set()
            is_prime = [True] * (N+1)
            is_prime[0] = is_prime[1] = False
            for p in range(2, int(N**0.5)+1):
                if is_prime[p]:
                    for m in range(p*p, N+1, p):
                        is_prime[m] = False
            primes = [x for x in range(1, N+1) if is_prime[x]]
            non_primes = [x for x in range(1, N+1) if not is_prime[x]]
            random.shuffle(primes)
            random.shuffle(non_primes)
            candidates = primes + non_primes
            for x in candidates:
                added = [x + a for a in S] + [2*x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    S.add(x)
                    sums.update(added)
            S = sorted(S)
        
        if len(S) >= best_size:
            continue
        
        if _is_maximal_fast(S, sums):
            best = S
            best_size = len(best)
            reduced = _reduce(best, deadline)
            if len(reduced) < best_size:
                best = reduced
                best_size = len(best)
            if time.time() < deadline:
                swapped = _local_search_swap(best, _get_sums(best), deadline)
                if swapped and len(swapped) < best_size:
                    best = swapped
                    best_size = len(best)
                    reduced = _reduce(best, deadline)
                    if len(reduced) < best_size:
                        best = reduced
                        best_size = len(best)
    
    return best
