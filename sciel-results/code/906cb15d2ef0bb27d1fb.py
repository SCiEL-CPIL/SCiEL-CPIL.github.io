"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    N = 128
    elements = list(range(1, N + 1))

    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # Find all bad triples: (a,b,c) with a<b<c where gcd(a,b)=gcd(a,c)=gcd(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    bad_triples.append((a, b, c))

    # For each element, precompute which other elements form bad triples with it
    # conflict[x] = set of elements y such that there exists z with (x,y,z) a bad triple
    # But we need hypergraph: for branch and bound, we need to know for a set S,
    # whether adding x would complete a bad triple with two elements in S.

    # Precompute: for each pair (x, d) where d is a gcd value,
    # the set of elements y where gcd(x,y) = d
    # This helps check: if we add x to set S, is there a pair (a,b) in S
    # such that gcd(x,a)=gcd(x,b)=gcd(a,b)?

    # For each element x, for each possible gcd value d,
    # precompute the bitmask of elements y where gcd(x,y) = d
    gcd_mask = [[0] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            d = gcd_table[x][y]
            gcd_mask[x][d] |= (1 << y)

    # For each d, precompute all pairs (a,b) with gcd(a,b) = d
    # and for fast lookup, for each d and element x, the bitmask of elements y
    # where gcd(x,y) = d AND there exists some pair... 
    # Actually, let's think differently.

    # A triple (a,b,c) is bad iff gcd(a,b)=gcd(a,c)=gcd(b,c)=d for some d.
    # This means: a = d*a', b = d*b', c = d*c' where gcd(a',b')=gcd(a',c')=gcd(b',c')=1
    # i.e., a', b', c' are pairwise coprime.

    # For branch and bound on the hypergraph:
    # We want max independent set in 3-uniform hypergraph.
    # Standard approach: branch on vertices, use upper bound = |remaining| + |selected|
    # with pruning when upper bound <= best.

    # But 128 vertices is too many for pure branch and bound.
    # Instead, use a greedy + local search approach with better conflict checking.

    # Let me use a different structural approach:
    # Group numbers by their "GCD profile" and use a smarter greedy.

    # Actually, let's implement a proper maximum independent set solver
    # using the fact that the hypergraph is 3-uniform.

    # Key: for a candidate set S, x can be added iff there is NO pair (a,b) in S
    # such that gcd(x,a) = gcd(x,b) = gcd(a,b).

    # Precompute for each x and each d: bitmask of y where gcd(x,y) = d
    # (already done above as gcd_mask)

    # For each d, precompute: for each pair of elements in gcd_mask[?][d]...
    # This is getting complex. Let me use a direct approach with memoization.

    # For the branch and bound, I'll use a recursive approach with:
    # - current set (as bitmask)
    # - available elements (as bitmask)
    # - best solution found so far
    # Upper bound: |current| + |available|
    # Prune if upper bound <= best_size

    # But 2^128 is too large. Need smarter pruning.
    # Use: if two elements in available form a bad triple with any element in current,
    # then at most one of them can be added. This gives a tighter bound.

    # For now, let's do a strong greedy with multiple restarts and local search,
    # but with a better conflict check that's faster.

    def can_add_fast(x, selected_mask):
        """Check if x can be added to set represented by selected_mask."""
        for d in range(1, N + 1):
            mask_x_d = gcd_mask[x][d]
            # Elements in selected set that have gcd(x, y) = d
            candidates = selected_mask & mask_x_d
            if candidates == 0:
                continue
            # Need to check if any pair (a,b) in candidates has gcd(a,b) = d
            # Precomputed: for each d, list of pairs with gcd = d
            # Check if any such pair is fully in candidates
            for a, b in pairs_by_gcd[d]:
                if (candidates & (1 << a)) and (candidates & (1 << b)):
                    return False
        return True

    # Precompute pairs_by_gcd
    from collections import defaultdict
    pairs_by_gcd = defaultdict(list)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = gcd_table[a][b]
            pairs_by_gcd[d].append((a, b))

    def greedy(order):
        selected_mask = 0
        for x in order:
            if can_add_fast(x, selected_mask):
                selected_mask |= (1 << x)
        return selected_mask

    def mask_to_list(mask):
        return [i for i in range(1, N + 1) if mask & (1 << i)]

    def mask_size(mask):
        return bin(mask).count('1')

    # Local search: try to improve by removing one element and adding more
    def local_search(mask, time_limit):
        best_mask = mask
        best_size = mask_size(best_mask)
        
        while time.time() < time_limit:
            improved = False
            
            # Try adding
            for x in range(1, N + 1):
                if best_mask & (1 << x):
                    continue
                if can_add_fast(x, best_mask):
                    best_mask |= (1 << x)
                    best_size += 1
                    improved = True
                    break
            if improved:
                continue
            
            # Try swap: remove one, add multiple
            for i in range(1, N + 1):
                if not (best_mask & (1 << i)):
                    continue
                temp = best_mask & ~(1 << i)
                added = 0
                for x in range(1, N + 1):
                    if x == i or (temp & (1 << x)):
                        continue
                    if can_add_fast(x, temp):
                        temp |= (1 << x)
                        added += 1
                        if added >= 2:
                            break
                if added > 1:
                    best_mask = temp
                    best_size += added - 1
                    improved = True
                    break
            
            if not improved:
                break
        
        return best_mask

    # Multiple greedy with different orderings
    best_mask = 0

    # Ordering 1: ascending
    m = greedy(range(1, N + 1))
    if mask_size(m) > mask_size(best_mask):
        best_mask = m

    if time.time() > deadline - 20:
        return mask_to_list(best_mask) if best_mask else [1]

    # Ordering 2: descending
    m = greedy(range(N, 0, -1))
    if mask_size(m) > mask_size(best_mask):
        best_mask = m

    if time.time() > deadline - 15:
        return mask_to_list(best_mask) if best_mask else [1]

    # Ordering 3: by number of prime factors
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    order3 = sorted(range(1, N + 1), key=lambda x: (num_prime_factors(x), x))
    m = greedy(order3)
    if mask_size(m) > mask_size(best_mask):
        best_mask = m

    if time.time() > deadline - 10:
        return mask_to_list(best_mask) if best_mask else [1]

    # Random orderings
    import random
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        m = greedy(order)
        if mask_size(m) > mask_size(best_mask):
            best_mask = m

    if time.time() > deadline - 3:
        return mask_to_list(best_mask) if best_mask else [1]

    # Local search
    best_mask = local_search(best_mask, deadline - 1)

    return mask_to_list(best_mask) if best_mask else [1]
# EVOLVE-BLOCK-END
