def construct():
    import time
    import random
    from math import gcd
    from itertools import combinations

    deadline = time.time() + 85

    n = 128
    # Precompute GCD table
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, n+1):
            gcd_table[i][j] = gcd(i, j)

    # Build conflict hyperedges: for each triple (a,b,c) that is "bad",
    # we record that these three cannot all be in the set.
    # Instead of storing all triples, we build a pairwise conflict structure:
    # For each pair (a,b), store the set of c such that (a,b,c) is bad.
    # Then when checking if x can join S, we check if any pair in S has x in its bad set.
    
    # More efficient: for each x, precompute which pairs (a,b) would form a bad triple with x.
    # bad_pairs[x] = set of (min(a,b), max(a,b)) such that gcd(a,b)==gcd(a,x)==gcd(b,x)
    
    bad_pairs = [set() for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, n+1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    bad_pairs[c].add((a, b))
                    bad_pairs[b].add((a, c))
                    bad_pairs[a].add((b, c))

    def can_add(x, S_set):
        for a, b in bad_pairs[x]:
            if a in S_set and b in S_set:
                return False
        return True

    def greedy_extend(S, S_set, order):
        """Extend S greedily using the given order."""
        for x in order:
            if x not in S_set and can_add(x, S_set):
                S.append(x)
                S_set.add(x)
        return S, S_set

    def local_search(S, S_set, rng, max_iters=500):
        """Improve S using add/remove/swap operations."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Try adding elements
            for x in range(1, n+1):
                if x not in S_set and can_add(x, S_set):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, try to add two
            if len(S) >= 3:
                removed = None
                added = []
                for x in S:
                    S_set.remove(x)
                    removed = x
                    # Try to add elements
                    for y in range(1, n+1):
                        if y not in S_set and y != x and can_add(y, S_set):
                            S_set.add(y)
                            added.append(y)
                            if len(added) >= 1:
                                break
                    if added:
                        # Keep the swap if we gained at least one element
                        S = [z for z in S if z != x] + added
                        improved = True
                        break
                    else:
                        # Revert
                        S_set.add(x)
                        removed = None
                if removed is not None and not added:
                    # Revert the last removal
                    S_set.add(removed)
        
        return S, S_set

    # Strategy: Use a "core" structure based on gcd patterns.
    # Key insight: Numbers that are powers of 2 are mutually safe.
    # More generally, numbers with "unique" gcd profiles are safe.
    
    # Let's try a different approach: build the set by layers of "gcd-depth".
    # Layer 0: {1}
    # Layer 1: primes
    # Layer 2: products of two distinct primes
    # etc.
    
    # Actually, let's use a smarter initialization:
    # Start with numbers that have few conflicts, then greedily add.
    
    # Compute conflict degree for each x: number of bad pairs involving x
    conflict_degree = [0] * (n+1)
    for x in range(1, n+1):
        conflict_degree[x] = len(bad_pairs[x])
    
    # Sort by conflict degree (ascending) - low conflict first
    order_by_conflict = sorted(range(1, n+1), key=lambda x: conflict_degree[x])
    
    best_S = []
    best_S_set = set()
    
    rng = random.Random(42)
    
    # Multiple strategies for initialization
    for trial in range(100):
        if time.time() >= deadline - 10:
            break
        
        if trial == 0:
            # Greedy by conflict degree
            order = order_by_conflict
        elif trial < 5:
            # Random order
            order = list(range(1, n+1))
            rng.shuffle(order)
        elif trial < 10:
            # Reverse conflict degree
            order = list(reversed(order_by_conflict))
        else:
            # Perturbed conflict order
            order = order_by_conflict[:]
            rng.shuffle(order)
            # Put some high-conflict elements early to test
            for _ in range(5):
                idx = rng.randint(0, len(order)-1)
                order.insert(0, order.pop(idx))
        
        S = []
        S_set = set()
        S, S_set = greedy_extend(S, S_set, order)
        S, S_set = local_search(S, S_set, rng, max_iters=200)
        
        if len(S) > len(best_S):
            best_S = S
            best_S_set = S_set
    
    # Further local search on the best solution
    if time.time() < deadline - 10:
        best_S, best_S_set = local_search(best_S, best_S_set, rng, max_iters=2000)
    
    # Try kick-and-recover: remove a few elements, re-greedy
    for _ in range(50):
        if time.time() >= deadline - 5:
            break
        if not best_S:
            continue
        
        k = rng.randint(3, 8)
        if k > len(best_S):
            k = max(1, len(best_S) // 3)
        to_remove = set(rng.sample(best_S, k))
        S = [x for x in best_S if x not in to_remove]
        S_set = set(S)
        
        # Re-greedy with multiple orders
        for order in [order_by_conflict, list(range(1, n+1))]:
            S2 = S[:]
            S2_set = S_set.copy()
            S2, S2_set = greedy_extend(S2, S2_set, order)
            S2, S2_set = local_search(S2, S2_set, rng, max_iters=100)
            if len(S2) > len(best_S):
                best_S = S2
                best_S_set = S2_set
    
    return best_S
