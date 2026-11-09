import random
import time

N = 256

def _build_sums(S):
    """Build set of all unordered pair sums a+b with a<=b from S."""
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    return sums

def _is_valid_sidon(S):
    """Check if S is a Sidon set."""
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            s = a + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(S, N):
    """Check if S is inclusion-maximal in [1,N]."""
    S_set = set(S)
    sums = _build_sums(S)
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if x can be added
        valid = True
        for a in S:
            if (x + a) in sums:
                valid = False
                break
        if valid:
            if (2 * x) in sums:
                valid = False
        if valid:
            return False
    return True

def _greedy_randomize(order, N):
    """Greedy Sidon set construction given an order of elements."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _try_remove(S, x, N):
    """Try removing element x from S and check if result is still maximal."""
    S_new = [s for s in S if s != x]
    if not S_new:
        return None
    if _is_maximal(S_new, N):
        return S_new
    return None

def _local_search(S, N, time_limit, start_time):
    """Local search to minimize cardinality while maintaining maximality."""
    S = list(S)
    if not S:
        return S
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Precompute sums
        sums = _build_sums(S)
        
        # Try single element removals
        candidates = list(S)
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            
            S_new = _try_remove(S, x, N)
            if S_new is not None:
                S = S_new
                improved = True
                break
        
        # Try pair removals if single removals stalled
        if not improved and time.time() - start_time < time_limit - 1:
            S_set = set(S)
            sums = _build_sums(S)
            
            # Find elements that appear as blockers for the same missing elements
            # If two elements always block together, they might be removable as a pair
            pair_candidates = []
            for i, x in enumerate(S):
                for j in range(i + 1, len(S)):
                    y = S[j]
                    pair_candidates.append((x, y))
            
            random.shuffle(pair_candidates)
            
            for x, y in pair_candidates[:50]:  # Limit pair trials
                if time.time() - start_time > time_limit:
                    break
                
                S_new = [s for s in S if s != x and s != y]
                if not S_new:
                    continue
                if _is_maximal(S_new, N):
                    S = S_new
                    improved = True
                    break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best = None
    best_size = N + 1
    
    # Baseline: Ascending greedy
    order = list(range(1, N + 1))
    S = _greedy_randomize(order, N)
    S = _local_search(S, N, deadline - start_time - 5, start_time)
    if len(S) < best_size:
        best = S
        best_size = len(S)
    
    random.seed(42)
    
    # Try multiple random orderings with different strategies
    for trial in range(2000):
        if time.time() > deadline:
            break
        
        # Different randomization strategies
        strategy = trial % 4
        
        if strategy == 0:
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif strategy == 1:
            # Weighted random: favor middle elements
            weights = [1.0 / (1 + abs(x - N/2)/N) for x in range(1, N + 1)]
            order = []
            remaining = list(range(1, N + 1))
            while remaining:
                total_w = sum(weights[i-1] for i in remaining)
                r = random.random() * total_w
                cum = 0
                for idx, x in enumerate(remaining):
                    cum += weights[x-1]
                    if cum >= r:
                        order.append(x)
                        remaining.remove(x)
                        break
        elif strategy == 2:
            # Reverse-ish: start from middle, expand outward
            mid = N // 2
            order = [mid]
            for d in range(1, mid + 1):
                if mid + d <= N:
                    order.append(mid + d)
                if mid - d >= 1:
                    order.append(mid - d)
            random.shuffle(order[1:])  # Keep first element but shuffle rest
        else:
            # Block-based: shuffle blocks then interleave
            block_size = 16
            blocks = [list(range(i+1, min(i+block_size+1, N+1))) for i in range(0, N, block_size)]
            random.shuffle(blocks)
            order = []
            idx = 0
            while any(blocks):
                for b in blocks:
                    if b:
                        order.append(b.pop(0))
        
        S = _greedy_randomize(order, N)
        
        # Reduce with remaining time budget
        remaining = deadline - time.time()
        if remaining > 5:
            S = _local_search(S, N, remaining - 5, start_time)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
    
    if best is None:
        best = _greedy_randomize(list(range(1, N + 1)), N)
        
    return best
