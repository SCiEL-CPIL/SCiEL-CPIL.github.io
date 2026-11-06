"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a nonempty subset of [1, 512] with lcm(a,b) <= 512 for every pair.
    Maximize cardinality.
    """
    import time
    from math import gcd
    
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    N = 512
    
    # 1. Build compatibility matrix
    compat = [bytearray(N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            g = gcd(a, b)
            if a * b // g <= N:
                compat[a][b] = 1
                compat[b][a] = 1
    
    def get_divisors(M):
        divs = []
        for i in range(1, int(M**0.5) + 1):
            if M % i == 0:
                divs.append(i)
                if i != M // i:
                    divs.append(M // i)
        return divs
    
    def try_add_to_set(in_set):
        """Try to greedily add all compatible elements. Returns updated set and count added."""
        added = 0
        for v in range(1, N + 1):
            if v not in in_set:
                if all(compat[v][s] for s in in_set):
                    in_set.add(v)
                    added += 1
        return in_set, added
    
    def local_search(in_set, max_iters=50):
        """Local search: add + conflict-based swap."""
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            in_set, added = try_add_to_set(in_set)
            if added > 0:
                continue
            # No additions possible. Try conflict-based swap.
            # For each y in in_set, count how many outside elements become compatible if y is removed.
            best_y = None
            best_gain = 0
            outside = [x for x in range(1, N + 1) if x not in in_set]
            for y in in_set:
                gain = 0
                for x in outside:
                    # Check if x is compatible with in_set - {y}
                    ok = True
                    for s in in_set:
                        if s != y and not compat[x][s]:
                            ok = False
                            break
                    if ok and not compat[x][y]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_y = y
            if best_y is not None and best_gain > 0:
                in_set.remove(best_y)
                # Now try to add back the newly compatible elements
                in_set, _ = try_add_to_set(in_set)
            else:
                break
        return in_set
    
    # 2. Find candidate initial sets: top divisor sets
    candidates = []
    for M in range(1, N + 1):
        divs = get_divisors(M)
        candidates.append((len(divs), M, set(divs)))
    
    # Sort by size descending, take top 10 distinct sets
    candidates.sort(key=lambda x: -x[0])
    seen = set()
    top_candidates = []
    for size, M, divs in candidates:
        key = frozenset(divs)
        if key not in seen:
            seen.add(key)
            top_candidates.append(divs)
            if len(top_candidates) >= 10:
                break
    
    # 3. Run local search on each candidate, keep the best
    best_result = set()
    best_size = 0
    
    for i, init_set in enumerate(top_candidates):
        if time.time() > deadline - 10:
            break
        cur = local_search(set(init_set))
        if len(cur) > best_size:
            best_size = len(cur)
            best_result = cur
    
    # 4. Also try a pure greedy from scratch as fallback
    if time.time() < deadline - 10:
        greedy_set = set()
        greedy_set, _ = try_add_to_set(greedy_set)
        greedy_set = local_search(greedy_set)
        if len(greedy_set) > best_size:
            best_size = len(greedy_set)
            best_result = greedy_set
    
    return list(best_result) if best_result else [1]
# EVOLVE-BLOCK-END
