import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: Andrásfai graph And(9) on 26 vertices (from seed)
    # Actually And(9) is on 27 vertices. Seed used 26.
    # Let's use the seed's valid graph as fallback.
    fallback_n = 26
    fallback_edges = [[i, j] for i in range(fallback_n) for j in range(i + 1, fallback_n) if (j - i) % fallback_n % 3 == 1]
    
    def is_sum_free(S, n):
        # S is a set of integers mod n
        # Check if x+y in S for any x,y in S
        # Also check x-x=0? No, 0 not in S usually.
        # Check x-y in S? In undirected circulant, S is symmetric.
        # Triangle condition: No a,b,c in S such that a+b+c=0?
        # In C_n(S), triangle exists if there are d1, d2 in S such that d1+d2 in S or d1-d2 in S.
        # Since S is symmetric, d1-d2 in S is equivalent to d1+d2' in S where -d2=d2'.
        # So we just need S to be sum-free: (S+S) cap S = empty.
        S_list = list(S)
        for a in S_list:
            for b in S_list:
                if (a + b) % n in S:
                    return False
        return True

    def greedy_alpha(n, S):
        # Estimate independence number of C_n(S)
        # Greedy: pick vertex, remove it and neighbors, repeat.
        # To get a good upper bound, we can run multiple random orderings.
        # Or use a simple heuristic: 
        # The independence number is at most n / (avg_degree + 1) roughly? No.
        # Let's use a greedy set packing.
        remaining = set(range(n))
        alpha = 0
        # Randomized greedy
        for _ in range(5):
            rem = set(range(n))
            count = 0
            while rem:
                # Pick vertex with fewest neighbors in rem? (Min-degree heuristic for independent set)
                # Or just pick random.
                # Min-degree in induced subgraph is better for packing independent sets.
                min_deg = float('inf')
                best_v = None
                for v in rem:
                    deg = 0
                    for d in S:
                        if (v + d) % n in rem: deg += 1
                        if (v - d) % n in rem: deg += 1
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                if best_v is not None:
                    count += 1
                    rem.remove(best_v)
                    for d in S:
                        rem.discard((best_v + d) % n)
                        rem.discard((best_v - d) % n)
            alpha = max(alpha, count)
        return alpha

    def try_find_circulant(n, max_time):
        # Search for a symmetric sum-free set S in Z_n
        # Start with a random set and refine
        best_S = None
        best_alpha = float('inf')
        
        # Initialize with some random sets
        for _ in range(100):
            if time.time() > max_time: break
            S = set()
            # Try to build S
            # Randomly add elements, checking sum-free constraint
            candidates = list(range(1, n//2 + 1))
            random.shuffle(candidates)
            for d in candidates:
                if time.time() > max_time: break
                new_S = S | {d, n-d}
                if is_sum_free(new_S, n):
                    # Check if this improves alpha
                    a = greedy_alpha(n, new_S)
                    if a < best_alpha:
                        best_alpha = a
                        best_S = new_S
                    if a <= 9:
                        return new_S
            # Also try to remove elements to fix sum-free issues if we started with a dense set
            # But we are building up.
            
        return best_S

    # Try n=40
    S40 = try_find_circulant(40, time.time() + 30)
    if S40 and greedy_alpha(40, S40) <= 9:
        # Verify with exact check? 
        # For n=40, exact check is feasible if we use a good B&B or just trust greedy if it's very low.
        # Let's do a quick exact check with a simple B&B with timeout.
        def exact_alpha_check(n, S):
            # Simple B&B for Max Independent Set
            # Use bitsets
            adj = [0] * n
            for i in range(n):
                mask = 0
                for d in S:
                    j1 = (i + d) % n
                    j2 = (i - d) % n
                    mask |= (1 << j1) | (1 << j2)
                adj[i] = mask
            mask_all = (1 << n) - 1
            best = 0
            def bb(rem, count):
                nonlocal best
                if not rem:
                    if count > best: best = count
                    return
                # Bound
                if count + popcount(rem) <= best:
                    return
                # Pick vertex
                # Heuristic: pick vertex with max degree in rem
                v = -1
                max_deg = -1
                r = rem
                while r:
                    lsb = r & -r
                    i = lsb.bit_length() - 1
                    deg = popcount(adj[i] & rem)
                    if deg > max_deg:
                        max_deg = deg
                        v = i
                    r ^= lsb
                if v == -1: return
                # Branch 1: Include v
                bb(rem & ~(1 << v) & ~adj[v], count + 1)
                # Branch 2: Exclude v
                bb(rem & ~(1 << v), count)
            
            def popcount(x):
                return bin(x).count("1")
                
            bb(mask_all, 0)
            return best

        # Run exact check only if greedy is promising
        if greedy_alpha(40, S40) <= 10:
            t0 = time.time()
            try:
                alpha_exact = exact_alpha_check(40, S40)
                if alpha_exact <= 9:
                    edges = []
                    for i in range(40):
                        for d in S40:
                            j = (i + d) % 40
                            if i < j:
                                edges.append([i, j])
                    return {'n': 40, 'edges': edges}
            except Exception:
                pass
            if time.time() - t0 > 20:
                pass # Timeout, skip

    # Try n=39
    S39 = try_find_circulant(39, time.time() + 30)
    if S39 and greedy_alpha(39, S39) <= 9:
        if greedy_alpha(39, S39) <= 10:
            t0 = time.time()
            try:
                alpha_exact = exact_alpha_check(39, S39)
                if alpha_exact <= 9:
                    edges = []
                    for i in range(39):
                        for d in S39:
                            j = (i + d) % 39
                            if i < j:
                                edges.append([i, j])
                    return {'n': 39, 'edges': edges}
            except Exception:
                pass

    # Fallback to seed
    return {'n': fallback_n, 'edges': fallback_edges}
