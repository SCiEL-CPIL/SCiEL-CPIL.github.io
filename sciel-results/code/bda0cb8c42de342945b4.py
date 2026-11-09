"""
Maximize |S| for S ⊆ [1,512] with lcm(a,b) <= 512 for all a,b in S.

Efficient approach using precomputed adjacency bitmasks:
- Precompute for each number 1..512 a bitmask of all numbers it's compatible with
- Use bit operations for fast compatibility checks
- Multiple greedy strategies + local search to find large cliques
"""

def construct():
    import math
    import time
    import random
    from math import gcd

    t0 = time.time()
    deadline = t0 + 85  # 85s safety margin (limit is 90s)
    LIMIT = 512
    N = LIMIT

    def time_left():
        return deadline - time.time()

    # Precompute lcm compatibility using GCD for speed
    # adj[i] = bitmask of all j (1..512) such that lcm(i,j) <= 512
    adj = [0] * (N + 1)  # 1-indexed, adj[i] is bitmask
    
    # Build adjacency using GCD-based lcm check: lcm(i,j) = i*j // gcd(i,j)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            # lcm = i * j // g; check if <= LIMIT without overflow issues
            # Since i,j <= 512, i*j <= 262144, well within Python int range
            if i * j // g <= LIMIT:
                mask |= (1 << (j - 1))
        adj[i] = mask

    # Helper: check if x is compatible with all elements in set S (represented as bitmask)
    def compatible_with_all(x, S_mask):
        # x is compatible with all in S iff adj[x] & S_mask == S_mask
        # i.e., S_mask is a subset of adj[x]
        return (adj[x] & S_mask) == S_mask

    def set_to_mask(S):
        mask = 0
        for x in S:
            mask |= (1 << (x - 1))
        return mask

    def mask_to_set(mask):
        S = []
        m = mask
        while m:
            # Find lowest set bit
            lb = m & (-m)
            idx = lb.bit_length() - 1
            S.append(idx + 1)
            m ^= lb
        return S

    def popcount(mask):
        return mask.bit_count() if hasattr(mask, 'bit_count') else bin(mask).count('1')

    # Greedy max clique using bitmasks
    def greedy_clique(adj_list, order=None):
        """Greedy: repeatedly pick the node with highest degree among remaining."""
        if order is None:
            order = list(range(1, N + 1))
        
        remaining = set_to_mask(order)
        clique_mask = 0
        clique = []
        
        while remaining and time_left() > 5:
            # Find node in remaining with max degree within remaining
            best_node = None
            best_deg = -1
            m = remaining
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                node = idx + 1
                deg = popcount(adj[node] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_node = node
                m ^= lb
            
            if best_node is None:
                break
            
            clique.append(best_node)
            clique_mask |= (1 << (best_node - 1))
            remaining &= adj[best_node]
        
        return clique

    # Greedy with ordering by global degree
    def greedy_by_degree(order):
        """Pick nodes in given order, add if compatible with current clique."""
        clique = []
        clique_mask = 0
        for node in order:
            if time_left() < 5:
                break
            if compatible_with_all(node, clique_mask):
                clique.append(node)
                clique_mask |= (1 << (node - 1))
        return clique

    # Local search: try to swap one node out and two in, or add one
    def local_improve(clique):
        """Try to improve clique by adding compatible nodes or swapping."""
        S = list(clique)
        S_set = set(S)
        S_mask = set_to_mask(S)
        improved = True
        
        while improved and time_left() > 10:
            improved = False
            
            # Try adding any compatible node
            candidates = []
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if compatible_with_all(x, S_mask):
                    # Count how many other candidates it's compatible with
                    deg = 0
                    for y in range(1, N + 1):
                        if y not in S_set and y != x:
                            if (adj[x] >> (y - 1)) & 1:
                                deg += 1
                    candidates.append((deg, x))
            
            if candidates:
                candidates.sort(reverse=True)
                # Add the best candidate
                _, best_x = candidates[0]
                S.append(best_x)
                S_set.add(best_x)
                S_mask |= (1 << (best_x - 1))
                improved = True
            
            # Try 1-for-2 swap: remove one node, add two
            if time_left() > 10:
                for remove_node in list(S):
                    if time_left() < 10:
                        break
                    S_trial_mask = S_mask & ~(1 << (remove_node - 1))
                    
                    # Find two nodes that can be added (must also be compatible with each other)
                    added = []
                    for x in range(1, N + 1):
                        if len(added) >= 2:
                            break
                        if x == remove_node:
                            continue
                        if (S_trial_mask >> (x - 1)) & 1:
                            continue  # already in S
                        if compatible_with_all(x, S_trial_mask):
                            # Check if x is compatible with all previously added nodes
                            if added:
                                all_compat = True
                                for y in added:
                                    if not ((adj[x] >> (y - 1)) & 1):
                                        all_compat = False
                                        break
                                if not all_compat:
                                    continue
                            added.append(x)
                            S_trial_mask |= (1 << (x - 1))
                    
                    if len(added) >= 2:
                        S = [x for x in S if x != remove_node] + added
                        S_set = set(S)
                        S_mask = set_to_mask(S)
                        improved = True
        
        return S

    # Phase 1: Start with divisors of highly composite numbers
    def divisors_of(n):
        divs = []
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)

    # Find numbers <= 512 with many divisors
    div_counts = []
    for n in range(1, N + 1):
        d = len(divisors_of(n))
        div_counts.append((d, n))
    div_counts.sort(reverse=True)

    best_clique = []
    best_size = 0

    # Try greedy from top-divisor-count bases
    for d, n in div_counts[:15]:
        if time_left() < 20:
            break
        base = divisors_of(n)
        # Greedy expand from base
        S = list(base)
        S_set = set(S)
        S_mask = set_to_mask(S)
        
        # Iteratively add compatible nodes
        changed = True
        while changed and time_left() > 15:
            changed = False
            for x in range(1, N + 1):
                if time_left() < 15:
                    break
                if x in S_set:
                    continue
                if compatible_with_all(x, S_mask):
                    S.append(x)
                    S_set.add(x)
                    S_mask |= (1 << (x - 1))
                    changed = True
        
        S = local_improve(S)
        if len(S) > best_size:
            best_size = len(S)
            best_clique = S[:]

    # Phase 2: Greedy by degree ordering
    if time_left() > 15:
        # Order by global degree descending
        degrees = [(popcount(adj[i]), i) for i in range(1, N + 1)]
        degrees.sort(reverse=True)
        order = [i for _, i in degrees]
        
        S = greedy_by_degree(order)
        S = local_improve(S)
        if len(S) > best_size:
            best_size = len(S)
            best_clique = S[:]

    # Phase 3: Multiple random restarts with greedy
    if time_left() > 20:
        random.seed(42)
        for restart in range(100):
            if time_left() < 10:
                break
            
            # Random order with bias toward high-degree nodes
            order = list(range(1, N + 1))
            
            # Strategy 1: fully random
            if restart % 3 == 0:
                random.shuffle(order)
            # Strategy 2: random with high-degree nodes first
            elif restart % 3 == 1:
                degrees = [(popcount(adj[i]), i) for i in range(1, N + 1)]
                degrees.sort(reverse=True)
                order = [i for _, i in degrees]
                # Shuffle bottom 60%
                split = int(N * 0.4)
                random.shuffle(order[split:])
            # Strategy 3: random with low-degree nodes first (counterintuitive)
            else:
                degrees = [(popcount(adj[i]), i) for i in range(1, N + 1)]
                degrees.sort()  # ascending
                order = [i for _, i in degrees]
                # Shuffle top 60%
                split = int(N * 0.4)
                random.shuffle(order[:split])
            
            S = greedy_by_degree(order)
            S = local_improve(S)
            if len(S) > best_size:
                best_size = len(S)
                best_clique = S[:]

    # Phase 4: More aggressive local search - try removing one node to add more
    if time_left() > 15 and best_clique:
        S = list(best_clique)
        S_set = set(S)
        S_mask = set_to_mask(S)
        
        # For each node in S, try removing it and see if we can add more
        for remove_node in list(S):
            if time_left() < 10:
                break
            
            S_trial = [x for x in S if x != remove_node]
            S_trial_set = S_set - {remove_node}
            S_trial_mask = S_mask & ~(1 << (remove_node - 1))
            
            # Try to add as many nodes as possible
            changed = True
            while changed and time_left() > 10:
                changed = False
                for x in range(1, N + 1):
                    if time_left() < 10:
                        break
                    if x in S_trial_set:
                        continue
                    if compatible_with_all(x, S_trial_mask):
                        S_trial.append(x)
                        S_trial_set.add(x)
                        S_trial_mask |= (1 << (x - 1))
                        changed = True
            
            if len(S_trial) > len(S):
                S = S_trial
                S_set = S_trial_set
                S_mask = S_trial_mask
                best_clique = S[:]
                best_size = len(S)
        
        # Phase 5: Final polish - try all single-node swaps and 1-for-2 swaps
        if time_left() > 10:
            S = list(best_clique)
            S_set = set(S)
            S_mask = set_to_mask(S)
            improved = True
            while improved and time_left() > 5:
                improved = False
                for remove_node in list(S):
                    if time_left() < 5:
                        break
                    S_trial_mask = S_mask & ~(1 << (remove_node - 1))
                    
                    # Try adding one node
                    for add_node in range(1, N + 1):
                        if time_left() < 5:
                            break
                        if add_node == remove_node:
                            continue
                        if (S_trial_mask >> (add_node - 1)) & 1:
                            continue
                        if compatible_with_all(add_node, S_trial_mask):
                            S = [x for x in S if x != remove_node] + [add_node]
                            S_set = set(S)
                            S_mask = set_to_mask(S)
                            best_clique = S[:]
                            best_size = len(S)
                            improved = True
                            break
                    
                    # Try 1-for-2 swap
                    if not improved and time_left() > 5:
                        added = []
                        for x in range(1, N + 1):
                            if len(added) >= 2:
                                break
                            if x == remove_node:
                                continue
                            if (S_trial_mask >> (x - 1)) & 1:
                                continue
                            if compatible_with_all(x, S_trial_mask):
                                if added:
                                    all_compat = True
                                    for y in added:
                                        if not ((adj[x] >> (y - 1)) & 1):
                                            all_compat = False
                                            break
                                    if not all_compat:
                                        continue
                                added.append(x)
                                S_trial_mask |= (1 << (x - 1))
                        
                        if len(added) >= 2:
                            S = [x for x in S if x != remove_node] + added
                            S_set = set(S)
                            S_mask = set_to_mask(S)
                            best_clique = S[:]
                            best_size = len(S)
                            improved = True

    # Final validation
    result = sorted(set(best_clique)) if best_clique else divisors_of(480)
    
    # Safety: remove incompatible pairs
    valid = []
    valid_mask = 0
    for a in result:
        if compatible_with_all(a, valid_mask):
            valid.append(a)
            valid_mask |= (1 << (a - 1))
    
    if len(valid) < 2:
        valid = divisors_of(480)
    
    return valid
