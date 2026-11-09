"""
Maximize |S| for S ⊆ [1,512] with lcm(a,b) <= 512 for all a,b in S.

This is a maximum clique problem on the compatibility graph where edge (i,j)
exists iff lcm(i,j) <= 512.

Strategy:
1. Precompute adjacency bitmasks for O(1) compatibility checks.
2. Use Tomita's maximum clique algorithm (with coloring bound) for exact-ish search.
3. Use greedy initialization + local search to find large cliques.
4. Use the structure: numbers with large prime factors > 256 are very restrictive.
   Key insight: if p is prime and p > 256, then any number divisible by p
   can only pair with numbers whose lcm with it stays <= 512.
   Since lcm(a,b) >= max(a,b), and if p|a with p>256, then b must divide a
   (because lcm(a,b) = a*b/gcd(a,b) >= a, and for it to be <= 512, b/gcd(a,b) <= 512/a < 2).
   So b/gcd(a,b) = 1, meaning b|a.
   
   More generally: the set must be "lcm-closed downward" in a sense.
   
   Key structural observation: Let S be a valid set. For any a in S, all divisors
   of a that are in [1,512] are compatible with a (since lcm(a,d) = a when d|a).
   But not all divisors of different elements are mutually compatible.
   
   Better approach: Think in terms of prime factorization.
   A number n <= 512 has prime factors from {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,...}
   But if p > 256, then p can only appear in numbers that are exactly p (since 2p > 512).
   And such a number p can only pair with divisors of p, i.e., {1, p}.
   
   If p is in (256, 512] and prime, then including p forces us to only include 1 from
   numbers divisible by p (just p itself and 1). But 1 is compatible with everything.
   
   Actually, let's think differently. The constraint lcm(a,b) <= 512 means:
   For each prime p, max(v_p(a), v_p(b)) * p^{max(v_p(a),v_p(b))} ... no.
   lcm(a,b) = prod_p p^{max(v_p(a), v_p(b))} <= 512 = 2^9.
   
   So for the prime 2: max(v_2(a), v_2(b)) <= 9 (always true for numbers <= 512).
   For prime 3: max(v_3(a), v_3(b)) <= 5 (since 3^5=243 <= 512, 3^6=729 > 512).
   For prime 5: max(v_5(a), v_5(b)) <= 3 (5^3=125, 5^4=625>512).
   For prime 7: max(v_7(a), v_7(b)) <= 2 (7^2=49, 7^3=343, 7^4=2401>512). Wait 7^3=343<=512.
   For prime 11: max <= 2 (11^2=121, 11^3=1331>512).
   For prime 13: max <= 2 (13^2=169, 13^3=2197>512).
   For prime 17: max <= 2 (17^2=289, 17^3=4913>512).
   For prime 19: max <= 2 (19^2=361, 19^3=6859>512).
   For prime 23: max <= 2 (23^2=529>512). So max <= 1.
   For primes p >= 23: max(v_p(a), v_p(b)) <= 1.
   
   This means: for any prime p >= 23, at most one of a,b can be divisible by p.
   Equivalently, if a and b are both in S, they cannot share a prime factor >= 23,
   AND no number in S can have p^2 for p >= 23 (since p^2 > 512 for p>=23... wait 23^2=529>512, so indeed no number <= 512 has p^2 for p>=23).
   
   So for primes >= 23: each number in S can be divisible by at most one such prime,
   and two different numbers in S cannot be divisible by the same such prime.
   Actually they CAN'T be divisible by the same such prime p, because then
   max(v_p(a),v_p(b)) >= 1 for both, but that's fine as long as it's <=1.
   Wait: if a = 23*2 = 46 and b = 23*3 = 69, then lcm(46,69) = 23*2*3 = 138 <= 512. That's fine!
   The constraint is max(v_23(a), v_23(b)) <= 1, which is satisfied.
   
   The real constraint is the PRODUCT: prod_p p^{max(v_p(a),v_p(b))} <= 512.
   
   Let me think about what structure gives large sets.
   
   Consider the set of all divisors of 512 = 2^9: {1,2,4,8,16,32,64,128,256,512}. Size 10.
   lcm of any two is the larger one, <= 512. Good.
   
   Consider divisors of 360 = 2^3*3^2*5: lcm of any two divisors divides 360, so <= 360 <= 512. 
   Number of divisors of 360: 4*3*2 = 24.
   
   Consider divisors of 480 = 2^5*3*5: lcm divides 480 <= 512. Divisors: 6*2*2=24.
   
   Consider divisors of 432 = 2^4*3^3: lcm divides 432. Divisors: 5*4=20.
   
   What about union of divisor sets? If S1 = divisors of n1 and S2 = divisors of n2,
   we need lcm(a,b) <= 512 for all a in S1, b in S2.
   This requires lcm(n1,n2) <= 512? No, that's sufficient but not necessary.
   Actually if a|n1 and b|n2, then lcm(a,b) | lcm(n1,n2), so lcm(a,b) <= lcm(n1,n2).
   So if lcm(n1,n2) <= 512, then divisors(n1) ∪ divisors(n2) is valid.
   
   So we want to find n1,n2,...,nk such that lcm(n1,n2,...,nk) <= 512, and maximize
   |divisors(n1) ∪ ... ∪ divisors(nk)|.
   
   But this is a subset of valid sets. The optimal set might not be a union of divisor sets.
   
   However, for a first pass, let's find good combinations.
   
   lcm(n1,n2) <= 512 means n1 and n2 must be "compatible" in the lcm sense.
   
   Let's just use the graph approach with a good max-clique algorithm.
   The graph has 512 nodes. Tomita's algorithm with coloring should handle this.
"""

def construct():
    import math
    import time
    import random
    from math import gcd, isqrt

    t0 = time.time()
    DEADLINE = t0 + 85  # 85s safety margin (limit is 90s)
    LIMIT = 512
    N = LIMIT

    def time_left():
        return DEADLINE - time.time()

    # Precompute adjacency bitmasks
    # adj[i] = bitmask where bit j is set iff lcm(i,j+1) <= LIMIT
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if i * j // g <= LIMIT:
                mask |= (1 << (j - 1))
        adj[i] = mask

    def popcount(m):
        return m.bit_count() if hasattr(m, 'bit_count') else bin(m).count('1')

    def mask_to_list(mask):
        """Convert bitmask to list of 1-indexed values."""
        result = []
        m = mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            result.append(idx + 1)
            m ^= lb
        return result

    def list_to_mask(lst):
        mask = 0
        for x in lst:
            mask |= (1 << (x - 1))
        return mask

    # Tomita's maximum clique algorithm with coloring
    # Uses bitmasks for efficiency
    def max_clique_tomita(adj, N):
        """
        Tomita's algorithm for maximum clique.
        adj[i] is bitmask of neighbors of node i (1-indexed, bits 0..N-1 for nodes 1..N).
        Returns (size, list_of_nodes).
        """
        best = [0]
        best_clique = [[]]
        
        # Color sort: assign colors greedily
        def color_sort(candidates_mask):
            """
            Color the candidates. Returns list of (color, node) sorted by color descending.
            candidates_mask: bitmask of candidate nodes.
            """
            colors = {}
            uncolored = candidates_mask
            color = 0
            
            while uncolored:
                color += 1
                # Greedily color: take a maximal independent set from uncolored
                # A node can get this color if it's not adjacent to any already-colored-with-this-color node
                colored_this_color = 0
                remaining = uncolored
                while remaining:
                    # Pick a node from remaining
                    lb = remaining & (-remaining)
                    node_bit = lb
                    node_idx = node_bit.bit_length() - 1  # 0-indexed
                    node = node_idx + 1  # 1-indexed
                    
                    # Check if node is non-adjacent to all nodes in colored_this_color
                    # i.e., adj[node] & colored_this_color == 0
                    if (adj[node] & colored_this_color) == 0:
                        colored_this_color |= node_bit
                        colors[node_idx] = color
                        remaining ^= node_bit
                
                uncolored &= ~colored_this_color
            
            # Sort by color descending, then by node index
            # Build list of (color, node_idx)
            colored_list = []
            for node_idx, c in colors.items():
                colored_list.append((c, node_idx))
            colored_list.sort(key=lambda x: (-x[0], x[1]))
            return colored_list, color
        
        def expand(clique, candidates_mask):
            """Recursively expand clique with candidates."""
            if time_left() < 2:
                return
            
            if not candidates_mask:
                if len(clique) > best[0]:
                    best[0] = len(clique)
                    best_clique[0] = clique[:]
                return
            
            # Color sort
            colored_list, max_color = color_sort(candidates_mask)
            
            # Process in reverse order (highest color first)
            i = len(colored_list) - 1
            while i >= 0:
                if time_left() < 2:
                    return
                
                color, node_idx = colored_list[i]
                node = node_idx + 1
                
                # Pruning: if len(clique) + color <= best[0], no point continuing
                if len(clique) + color <= best[0]:
                    return
                
                # Add node to clique
                new_clique = clique + [node]
                new_candidates = candidates_mask & adj[node] & ~(1 << node_idx)
                
                expand(new_clique, new_candidates)
                
                # Remove node from candidates
                candidates_mask &= ~(1 << node_idx)
                i -= 1
        
        # Initial call
        full_mask = (1 << N) - 1
        expand([], full_mask)
        
        return best[0], best_clique[0]

    # Greedy clique construction
    def greedy_clique(order):
        """Greedy: process nodes in given order, add if compatible with all in clique."""
        clique = []
        clique_mask = 0
        for node in order:
            if time_left() < 5:
                break
            if (adj[node] & clique_mask) == clique_mask:
                clique.append(node)
                clique_mask |= (1 << (node - 1))
        return clique

    # Local improvement: try to add nodes, or swap
    def local_improve(clique):
        S = list(clique)
        S_set = set(S)
        S_mask = list_to_mask(S)
        improved = True
        
        while improved and time_left() > 15:
            improved = False
            
            # Try adding any compatible node
            best_add = None
            best_add_deg = -1
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if (adj[x] & S_mask) == S_mask:
                    # Count compatibility with other potential additions
                    deg = 0
                    for y in range(1, N + 1):
                        if y not in S_set and y != x:
                            if (adj[x] >> (y - 1)) & 1:
                                deg += 1
                    if deg > best_add_deg:
                        best_add_deg = deg
                        best_add = x
            
            if best_add is not None:
                S.append(best_add)
                S_set.add(best_add)
                S_mask |= (1 << (best_add - 1))
                improved = True
            
            # Try 1-for-2 swap
            if time_left() > 15:
                for remove_node in list(S):
                    if time_left() < 15:
                        break
                    S_trial_mask = S_mask & ~(1 << (remove_node - 1))
                    
                    added = []
                    for x in range(1, N + 1):
                        if len(added) >= 2:
                            break
                        if x == remove_node:
                            continue
                        if (S_trial_mask >> (x - 1)) & 1:
                            continue
                        if (adj[x] & S_trial_mask) == S_trial_mask:
                            if added:
                                compat = True
                                for y in added:
                                    if not ((adj[x] >> (y - 1)) & 1):
                                        compat = False
                                        break
                                if not compat:
                                    continue
                            added.append(x)
                            S_trial_mask |= (1 << (x - 1))
                    
                    if len(added) >= 2:
                        S = [x for x in S if x != remove_node] + added
                        S_set = set(S)
                        S_mask = list_to_mask(S)
                        improved = True
        
        return S

    # Phase 1: Greedy from various orderings
    best_clique = []
    best_size = 0

    # Order by degree descending
    degrees = [(popcount(adj[i]), i) for i in range(1, N + 1)]
    degrees.sort(reverse=True)
    order_high = [i for _, i in degrees]
    
    S = greedy_clique(order_high)
    S = local_improve(S)
    if len(S) > best_size:
        best_size = len(S)
        best_clique = S[:]

    # Order by value ascending (small numbers tend to be more compatible)
    order_asc = list(range(1, N + 1))
    S = greedy_clique(order_asc)
    S = local_improve(S)
    if len(S) > best_size:
        best_size = len(S)
        best_clique = S[:]

    # Phase 2: Random restarts
    random.seed(12345)
    for restart in range(200):
        if time_left() < 10:
            break
        
        order = list(range(1, N + 1))
        
        if restart % 4 == 0:
            random.shuffle(order)
        elif restart % 4 == 1:
            # High degree first, shuffle rest
            order = order_high[:]
            split = N // 3
            random.shuffle(order[split:])
        elif restart % 4 == 2:
            # Low degree first
            order_low = [i for _, i in sorted(degrees)]
            order = order_low[:]
            split = N // 3
            random.shuffle(order[:split])
        else:
            # Random with bias toward small numbers
            order = list(range(1, N + 1))
            for i in range(len(order) - 1, 0, -1):
                j = random.randint(0, i)
                # Bias: prefer smaller numbers earlier
                if order[j] < order[i] and random.random() < 0.3:
                    order[j], order[i] = order[i], order[j]
                else:
                    order[j], order[i] = order[i], order[j]
        
        S = greedy_clique(order)
        S = local_improve(S)
        if len(S) > best_size:
            best_size = len(S)
            best_clique = S[:]

    # Phase 3: Tomita's max clique (exact, but may be slow)
    if time_left() > 30:
        try:
            size, clique = max_clique_tomita(adj, N)
            if size > best_size:
                best_size = size
                best_clique = clique[:]
        except Exception:
            pass

    # Phase 4: More aggressive local search from best
    if time_left() > 15 and best_clique:
        S = list(best_clique)
        S_set = set(S)
        S_mask = list_to_mask(S)
        
        # Try removing each node and re-optimizing
        for remove_node in list(S):
            if time_left() < 10:
                break
            
            S_trial = [x for x in S if x != remove_node]
            S_trial_set = set(S_trial)
            S_trial_mask = list_to_mask(S_trial)
            
            # Greedily add back
            changed = True
            while changed and time_left() > 10:
                changed = False
                best_add = None
                best_deg = -1
                for x in range(1, N + 1):
                    if x in S_trial_set:
                        continue
                    if (adj[x] & S_trial_mask) == S_trial_mask:
                        deg = popcount(adj[x] & ~S_trial_mask)
                        if deg > best_deg:
                            best_deg = deg
                            best_add = x
                if best_add is not None:
                    S_trial.append(best_add)
                    S_trial_set.add(best_add)
                    S_trial_mask |= (1 << (best_add - 1))
                    changed = True
            
            if len(S_trial) > len(S):
                S = S_trial
                S_set = S_trial_set
                S_mask = S_trial_mask
                best_clique = S[:]
                best_size = len(S)

    # Final validation
    result = sorted(set(best_clique)) if best_clique else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    valid = []
    valid_mask = 0
    for a in result:
        if (adj[a] & valid_mask) == valid_mask:
            valid.append(a)
            valid_mask |= (1 << (a - 1))
    
    if len(valid) < 2:
        valid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    return valid
