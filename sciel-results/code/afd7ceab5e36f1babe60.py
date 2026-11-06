def construct():
    import time
    from math import gcd
    from itertools import combinations
    
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s
    
    n = 128
    numbers = list(range(1, n+1))
    
    # Precompute gcd matrix
    gcd_mat = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # Precompute bad triples: (a,b,c) where gcd(a,b)=gcd(a,c)=gcd(b,c)
    # Represent as set of frozensets or as a lookup
    # For efficiency, for each pair (a,b), store the set of c that form a bad triple
    # bad_pair[a][b] = set of c != a,b such that gcd(a,b)=gcd(a,c)=gcd(b,c)
    
    # Actually, let's think differently. For a candidate set S, adding x is valid
    # iff for all pairs (a,b) in S, NOT (gcd(a,b) == gcd(a,x) == gcd(b,x)).
    
    # For local search, we need to check if removing one and adding another works.
    
    # Greedy initialization with multiple strategies
    def greedy(order):
        selected = set()
        for x in order:
            ok = True
            for a in selected:
                for b in selected:
                    if a < b and gcd_mat[a][b] == gcd_mat[a][x] == gcd_mat[b][x]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.add(x)
        return selected
    
    # Try multiple orderings
    import random
    rng = random.Random(42)
    
    best_set = set()
    
    # Strategy 1: natural order
    s = greedy(list(range(1, n+1)))
    if len(s) > len(best_set):
        best_set = s
    
    # Strategy 2: reverse order
    s = greedy(list(range(n, 0, -1)))
    if len(s) > len(best_set):
        best_set = s
    
    # Strategy 3: random orderings
    for trial in range(20):
        order = list(range(1, n+1))
        rng.shuffle(order)
        s = greedy(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search: try swaps (remove one, add one)
    def is_valid_set(S):
        """Check if set S has no bad triple."""
        elems = list(S)
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                a, b = elems[i], elems[j]
                g = gcd_mat[a][b]
                for k in range(j+1, len(elems)):
                    c = elems[k]
                    if gcd_mat[a][c] == g and gcd_mat[b][c] == g:
                        return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to S without creating a bad triple."""
        for a in S:
            for b in S:
                if a < b and gcd_mat[a][b] == gcd_mat[a][x] == gcd_mat[b][x]:
                    return False
        return True
    
    # Local search with simulated annealing-like approach
    current = set(best_set)
    best = set(best_set)
    best_size = len(best)
    
    # Precompute for each x, which pairs (a,b) would be "conflicting" with x
    # conflict[x] = set of (a,b) pairs such that gcd(a,b)=gcd(a,x)=gcd(b,x)
    # This is expensive to precompute fully, so let's do it on the fly but efficiently
    
    # Better: for local search, try all single swaps
    elapsed = time.time() - start_time
    if elapsed < DEADLINE:
        # Try all possible swaps: remove i from current, add j not in current
        for _ in range(50):  # iterations of local search
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for i in list(current):
                for j in range(1, n+1):
                    if j in current:
                        continue
                    # Try: current - {i} + {j}
                    trial = (current - {i}) | {j}
                    if len(trial) <= best_size:
                        continue
                    # Check validity
                    valid = True
                    elems = list(trial)
                    for a_idx in range(len(elems)):
                        a = elems[a_idx]
                        for b_idx in range(a_idx+1, len(elems)):
                            b = elems[b_idx]
                            g = gcd_mat[a][b]
                            for c_idx in range(b_idx+1, len(elems)):
                                c = elems[c_idx]
                                if gcd_mat[a][c] == g and gcd_mat[b][c] == g:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        current = trial
                        best = trial
                        best_size = len(best)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    return list(best)
