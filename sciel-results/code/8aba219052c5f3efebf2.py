import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute for each x, the set of (a,b) pairs where gcd(a,x)==gcd(b,x)==gcd(a,b)
    # conflict_mask[x] is a bitmask over elements: for each pair (a,b) with a<b,
    # if adding x to {a,b} creates a bad triple, mark both a and b in conflict_mask[x]
    # Actually, let's think differently:
    # For a given set S and candidate x, x conflicts with S if there exist a,b in S (a≠b, a≠x, b≠x)
    # such that gcd(a,x) == gcd(b,x) == gcd(a,b).
    #
    # Key observation: gcd(a,x) == gcd(b,x) == gcd(a,b) = g means:
    #   g | a, g | b, g | x, and a/g, b/g, x/g are pairwise coprime... no that's not right.
    #   Actually: gcd(a,x)=g means g|a and g|x and gcd(a/g, x/g)=1
    #   Similarly gcd(b,x)=g means g|b and g|x and gcd(b/g, x/g)=1
    #   And gcd(a,b)=g means g|a and g|b and gcd(a/g, b/g)=1
    #
    # So for a fixed x and fixed g = gcd(a,x) = gcd(b,x) = gcd(a,b):
    #   g must divide x, and a/g and b/g must be coprime to x/g and to each other.
    #
    # Alternative approach: For each x, precompute for each possible gcd value g that divides x,
    # which elements a satisfy gcd(a,x) = g. Then two elements a,b in S conflict with x if
    # gcd(a,x) == gcd(b,x) == gcd(a,b).
    #
    # Simpler bitmask approach: For each x, define conflict[x] as a bitmask where bit i is set
    # if there exists some y (y≠x) such that... no, this doesn't work directly because the
    # conflict depends on the pair (a,b) both being in S.
    #
    # Let me use a different approach: For each x, precompute a list of "witness pairs" - 
    # pairs (a,b) such that gcd(a,x)==gcd(b,x)==gcd(a,b). Then x can be added to S iff
    # no witness pair has both elements in S.
    #
    # Even better: For each x, for each a ≠ x, compute g_ax = gcd(a,x). Then x conflicts with
    # the pair (a,b) in S iff gcd(b,x) == g_ax and gcd(a,b) == g_ax.
    #
    # For efficient checking: For each x, group elements by their gcd with x.
    # gcd_with_x[a] = gcd(a,x) for all a.
    # Then x can be added to S iff for every g, the set {a in S : gcd(a,x)=g} has the property
    # that no two elements a,b in it have gcd(a,b)=g.
    #
    # Let me precompute: for each x, for each a, gcd(a,x). Store as gcd_with[x][a].
    # Then for a candidate x and set S (as bitmask), check:
    #   For each a in S: g = gcd_with[x][a]
    #     For each b in S with b > a: if gcd_with[x][b] == g and gcd_table[a][b] == g: conflict
    #
    # This is still O(n²) per check. Let me think of a better way.
    #
    # Alternative: Precompute for each x a "conflict bitmask" approach:
    # For each x, define a graph on {1..128}\{x} where edge (a,b) exists if 
    # gcd(a,x)==gcd(b,x)==gcd(a,b). Then x can be added to S iff S∩({1..128}\{x}) is an
    # independent set in this graph.
    #
    # For n=128, checking if a set is an independent set in a graph can be done by:
    # For each vertex v in S, check if any neighbor of v is also in S.
    # Using bitmasks: neighbor_mask[v] has bits set for all neighbors of v.
    # Then S is independent iff for all v in S: (S & neighbor_mask[v]) == 0 (excluding v itself).
    #
    # But building this graph for each x is expensive. Let me just use the direct approach
    # but with bitmasks to speed up the inner loop.
    
    # Precompute gcd_with[x][a] = gcd(x, a) for all x, a in 1..128
    gcd_with = [[0]*(N+1) for _ in range(N+1)]
    for x in range(1, N+1):
        for a in range(1, N+1):
            gcd_with[x][a] = gcd_table[x][a]

    # For each x, precompute conflict graph as bitmask:
    # conflict_graph[x] is a list of N+1 bitmasks, where conflict_graph[x][a] has bit b set
    # if gcd(a,x) == gcd(b,x) == gcd(a,b) and a != b
    # Then x can be added to S (bitmask) iff for all a in S: (S & conflict_graph[x][a]) == 0
    
    # Actually, let's precompute: for each x, a list where for each a, we store the bitmask
    # of all b such that gcd(a,x)==gcd(b,x)==gcd(a,b).
    # Then to check if x can be added to S: for each a in S, check if (S & mask_x[a]) has any bit set (other than a itself, but a won't be in mask_x[a] since a≠b).
    
    # Precompute masks for each x
    # masks[x][a] = bitmask of b (b≠a) such that gcd(a,x)==gcd(b,x)==gcd(a,b)
    masks = [None]*(N+1)
    for x in range(1, N+1):
        m = [0]*(N+1)
        for a in range(1, N+1):
            if a == x:
                continue
            g_ax = gcd_table[a][x]
            for b in range(1, N+1):
                if b == a or b == x:
                    continue
                if gcd_table[b][x] == g_ax and gcd_table[a][b] == g_ax:
                    m[a] |= (1 << b)
        masks[x] = m

    def can_add_mask(S_mask, x):
        """Check if x can be added to set represented by S_mask."""
        m = masks[x]
        # For each a in S_mask, check if (S_mask & m[a]) != 0
        temp = S_mask
        while temp:
            lowbit = temp & (-temp)
            a = lowbit.bit_length() - 1
            if (S_mask & m[a]) != 0:
                return False
            temp ^= lowbit
        return True

    def greedy_construct(order):
        """Greedy: add elements in given order if valid."""
        S_mask = 0
        for x in order:
            if can_add_mask(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def local_search_add(S_mask):
        """Try to add any remaining elements to S."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = [x for x in range(1, N+1) if not (S_mask & (1 << x))]
            random.shuffle(candidates)
            for x in candidates:
                if can_add_mask(S_mask, x):
                    S_mask |= (1 << x)
                    improved = True
                    break
        return S_mask

    def local_search_swap(S_mask):
        """Try 1-1 swaps."""
        for _ in range(20):
            if time.time() >= deadline:
                break
            candidates = [x for x in range(1, N+1) if not (S_mask & (1 << x))]
            if not candidates:
                break
            # Pick a random element in S to remove
            temp = S_mask
            bits_in_S = []
            while temp:
                lowbit = temp & (-temp)
                bits_in_S.append(lowbit.bit_length() - 1)
                temp ^= lowbit
            if not bits_in_S:
                break
            u = random.choice(bits_in_S)
            v = random.choice(candidates)
            
            S_temp = S_mask & ~(1 << u)
            if can_add_mask(S_temp, v):
                S_mask = S_temp | (1 << v)
        return S_mask

    def mask_to_list(S_mask):
        result = []
        temp = S_mask
        while temp:
            lowbit = temp & (-temp)
            result.append(lowbit.bit_length() - 1)
            temp ^= lowbit
        return result

    def get_order(strategy):
        items = list(range(1, N+1))
        if strategy == 0:
            random.shuffle(items)
        elif strategy == 1:
            primes = [x for x in items if x > 1 and all(x % i != 0 for i in range(2, int(x**0.5)+1))]
            rest = [x for x in items if x not in primes]
            random.shuffle(rest)
            items = primes + rest
        elif strategy == 2:
            items.sort(reverse=True)
            random.shuffle(items)
        elif strategy == 3:
            def num_divisors(n):
                count = 0
                for i in range(1, int(n**0.5)+1):
                    if n % i == 0:
                        count += 2
                        if i != n//i:
                            count -= 1
                return count
            items.sort(key=lambda x: num_divisors(x))
            random.shuffle(items)
        elif strategy == 4:
            random.shuffle(items)
        return items

    best_size = 0
    best_mask = 0

    # Phase 1: Greedy construction with multiple strategies
    while time.time() < deadline - 5:
        strategy = random.randint(0, 4)
        order = get_order(strategy)
        S_mask = greedy_construct(order)
        S_mask = local_search_add(S_mask)
        
        sz = bin(S_mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = S_mask
            if best_size >= N - 5:
                break

    # Phase 2: Start from full set, remove conflicts
    full_mask = (1 << (N+1)) - 2  # bits 1..128 set
    while time.time() < deadline - 2:
        strategy = random.randint(0, 4)
        order = get_order(strategy)
        S_mask = full_mask
        # Remove elements in order if they're in a bad triple
        for x in order:
            if not (S_mask & (1 << x)):
                continue
            # Check if x is in any bad triple with other elements in S
            should_remove = False
            m = masks[x]
            temp = S_mask & ~(1 << x)
            while temp and not should_remove:
                lowbit = temp & (-temp)
                a = lowbit.bit_length() - 1
                if (S_mask & m[a]) != 0:
                    should_remove = True
                temp ^= lowbit
            if should_remove:
                S_mask &= ~(1 << x)
        
        S_mask = local_search_add(S_mask)
        S_mask = local_search_swap(S_mask)
        S_mask = local_search_add(S_mask)
        
        sz = bin(S_mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = S_mask

    # Phase 3: More local search on best
    if time.time() < deadline:
        S_mask = local_search_add(best_mask)
        S_mask = local_search_swap(S_mask)
        S_mask = local_search_add(S_mask)
        sz = bin(S_mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = S_mask

    return mask_to_list(best_mask)
