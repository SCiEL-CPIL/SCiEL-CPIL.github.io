def construct():
    import time
    from math import gcd
    from fractions import Fraction
    import sys
    
    N = 512
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s kill
    
    # Precompute prime factors for each number 1..511
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        if all(p % i != 0 for i in range(2, int(p**0.5) + 1)):
            for m in range(p, 512, p):
                prime_factors[m].add(p)
    
    # Precompute conflict masks: mask[a] has bit b set if gcd(a,b) != 1
    mask = [0] * 512
    for a in range(1, 512):
        if not prime_factors[a]:
            continue
        m = 0
        for pf in prime_factors[a]:
            for b in range(pf, 512, pf):
                if b != a:
                    m |= (1 << b)
        mask[a] = m
    
    # Weight of each vertex: 1/(512 - a)
    weight = [0.0] * 512
    weight_frac = [Fraction(0, 1)] * 512
    for a in range(1, 512):
        weight[a] = 1.0 / (N - a)
        weight_frac[a] = Fraction(1, N - a)
    
    # Sort candidates by descending weight (high a first)
    candidates = list(range(511, 0, -1))
    
    # Initial greedy solution
    def greedy(order, exclude_mask=0):
        S = []
        cm = exclude_mask
        for a in order:
            if (cm & mask[a]) == 0:
                S.append(a)
                cm |= (1 << a)
        return S, cm
    
    best_S, best_mask = greedy(candidates)
    best_score = sum(weight[a] for a in best_S)
    
    # Track best solution
    best_solution = list(best_S)
    
    # Branch and bound
    # State: (available_mask, current_score, current_set)
    # available_mask: bits set for vertices that can still be added
    # We branch on whether to include or exclude the next candidate
    
    # Precompute for each vertex, which other vertices it conflicts with (among candidates)
    # conflict_with[a] = set of candidates that conflict with a
    conflict_with = [set() for _ in range(512)]
    for a in range(1, 512):
        if not prime_factors[a]:
            continue
        m = mask[a]
        # Extract candidates from mask
        for b in range(1, 512):
            if (m >> b) & 1:
                conflict_with[a].add(b)
    
    # Upper bound: fractional knapsack relaxation
    # Given a set of available vertices, what's the max weight independent set?
    # Relaxation: ignore conflicts, just take vertices by weight until "budget" filled
    # But we don't have a budget - we can take all non-conflicting ones
    # Better bound: for each prime, at most one number with that prime can be selected
    # So upper bound = sum over primes of max weight among available numbers with that prime
    
    # Precompute for each prime, the list of numbers that have it
    prime_to_numbers = {}
    for a in range(1, 512):
        for pf in prime_factors[a]:
            if pf not in prime_to_numbers:
                prime_to_numbers[pf] = []
            prime_to_numbers[pf].append(a)
    
    # For upper bound: for each prime, we can pick at most one number
    # But a number has multiple primes, so this is an overestimate
    # Upper bound = sum over primes of max_{a with pf} weight[a]
    # This is valid because any independent set picks at most one number per prime
    
    def upper_bound(available_mask):
        """Compute upper bound on additional weight from available vertices."""
        bound = 0.0
        for pf, numbers in prime_to_numbers.items():
            max_w = 0.0
            for a in numbers:
                if (available_mask >> a) & 1:
                    if weight[a] > max_w:
                        max_w = weight[a]
            bound += max_w
        return bound
    
    # More refined upper bound: for each prime, pick the best available number with that prime
    # But ensure we don't double-count numbers with multiple primes
    # Actually, the simple sum over primes of max weight is a valid upper bound
    
    # Branch and bound search
    # We'll use a stack-based DFS to avoid recursion depth issues
    
    # Sort candidates by weight descending for branching order
    branch_order = sorted(range(1, 512), key=lambda a: -weight[a])
    
    # State: (index in branch_order, available_mask, current_score, current_set)
    # We process candidates in branch_order, deciding include/exclude
    
    # For efficiency, precompute for each index in branch_order, the vertex
    # and update available_mask when we include a vertex
    
    # Actually, let's use a different approach:
    # Maintain a set of "active" candidates (those not yet decided)
    # At each step, pick the highest-weight active candidate, branch on include/exclude
    
    # This is complex. Let's use a simpler approach:
    # Greedy + local search with multiple restarts and different strategies
    
    # Given time constraints, let's enhance the local search with:
    # 1. Multiple diverse initial solutions
    # 2. Stronger local moves (2-for-2, 3-for-3 swaps)
    # 3. Simulated annealing-like acceptance of worse solutions
    
    import random
    random.seed(42)
    
    def score(S):
        return sum(weight[a] for a in S)
    
    def local_search(S, smask, time_limit):
        S = list(S)
        smask = smask
        s_score = score(S)
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < time_limit:
            improved = False
            no_improve_count += 1
            
            # Move 1: Drop one and re-greedy
            for x in list(S):
                if time.time() >= time_limit:
                    break
                order = [a for a in candidates if a != x]
                S2, _ = greedy(order)
                s2 = score(S2)
                if s2 > s_score + 1e-12:
                    S = S2
                    smask = 0
                    for a in S:
                        smask |= (1 << a)
                    s_score = s2
                    improved = True
                    no_improve_count = 0
                    break
            
            # Move 2: 1-for-1 swap
            if time.time() >= time_limit:
                break
            S_set = set(S)
            for x in list(S):
                if time.time() >= time_limit:
                    break
                rest_mask = smask & ~(1 << x)
                S_without_x = [a for a in S if a != x]
                base_score = s_score - weight[x]
                best_y = None
                best_y_gain = None
                for y in candidates:
                    if y in S_set:
                        continue
                    if (rest_mask & mask[y]) != 0:
                        continue
                    gain = weight[y] - weight[x]
                    if best_y_gain is None or gain > best_y_gain:
                        best_y_gain = gain
                        best_y = y
                if best_y is not None and best_y_gain > 1e-12:
                    new_S = S_without_x + [best_y]
                    new_score = base_score + weight[best_y]
                    if new_score > s_score + 1e-12:
                        S = new_S
                        smask = 0
                        for a in S:
                            smask |= (1 << a)
                        s_score = new_score
                        S_set = set(S)
                        improved = True
                        no_improve_count = 0
                        break
            
            # Move 3: 2-for-1 swap
            if time.time() >= time_limit:
                break
            S_set = set(S)
            n_S = len(S)
            for i in range(n_S):
                if time.time() >= time_limit:
                    break
                for j in range(i + 1, n_S):
                    if time.time() >= time_limit:
                        break
                    x, y = S[i], S[j]
                    rest_mask = smask & ~(1 << x) & ~(1 << y)
                    S_without_xy = [a for a in S if a != x and a != y]
                    base_score = s_score - weight[x] - weight[y]
                    best_z = None
                    best_z_gain = None
                    for z in candidates:
                        if z in S_set:
                            continue
                        if (rest_mask & mask[z]) != 0:
                            continue
                        gain = weight[z] - weight[x] - weight[y]
                        if best_z_gain is None or gain > best_z_gain:
                            best_z_gain = gain
                            best_z = z
                    if best_z is not None and best_z_gain > 1e-12:
                        new_S = S_without_xy + [best_z]
                        new_score = base_score + weight[best_z]
                        if new_score > s_score + 1e-12:
                            S = new_S
                            smask = 0
                            for a in S:
                                smask |= (1 << a)
                            s_score = new_score
                            S_set = set(S)
                            n_S = len(S)
                            improved = True
                            no_improve_count = 0
                            break
                else:
                    continue
                break
            
            # Move 4: 2-for-2 swap
            if time.time() >= time_limit:
                break
            S_set = set(S)
            n_S = len(S)
            for i in range(n_S):
                if time.time() >= time_limit:
                    break
                for j in range(i + 1, n_S):
                    if time.time() >= time_limit:
                        break
                    x, y = S[i], S[j]
                    rest_mask = smask & ~(1 << x) & ~(1 << y)
                    S_without_xy = [a for a in S if a != x and a != y]
                    base_score = s_score - weight[x] - weight[y]
                    
                    # Find best pair (z, w) to add
                    candidates_list = [z for z in candidates if z not in S_set and (rest_mask & mask[z]) == 0]
                    best_pair = None
                    best_pair_gain = None
                    for idx1 in range(len(candidates_list)):
                        if time.time() >= time_limit:
                            break
                        z = candidates_list[idx1]
                        temp_mask = rest_mask | (1 << z)
                        for idx2 in range(idx1 + 1, len(candidates_list)):
                            if time.time() >= time_limit:
                                break
                            w = candidates_list[idx2]
                            if (temp_mask & mask[w]) != 0:
                                continue
                            gain = weight[z] + weight[w] - weight[x] - weight[y]
                            if best_pair_gain is None or gain > best_pair_gain:
                                best_pair_gain = gain
                                best_pair = (z, w)
                    
                    if best_pair is not None and best_pair_gain > 1e-12:
                        z, w = best_pair
                        new_S = S_without_xy + [z, w]
                        new_score = base_score + weight[z] + weight[w]
                        if new_score > s_score + 1e-12:
                            S = new_S
                            smask = 0
                            for a in S:
                                smask |= (1 << a)
                            s_score = new_score
                            S_set = set(S)
                            n_S = len(S)
                            improved = True
                            no_improve_count = 0
                            break
                else:
                    continue
                break
            
            # Move 5: Try adding any single element that fits
            if time.time() >= time_limit:
                break
            S_set = set(S)
            for z in candidates:
                if z in S_set:
                    continue
                if (smask & mask[z]) != 0:
                    continue
                new_S = S + [z]
                new_score = s_score + weight[z]
                if new_score > s_score + 1e-12:
                    S = new_S
                    smask |= (1 << z)
                    s_score = new_score
                    S_set = set(S)
                    improved = True
                    no_improve_count = 0
        
        return S, smask, s_score
    
    # Run local search on initial solution
    best_S, best_mask, best_score = local_search(best_S, best_mask, deadline)
    best_solution = list(best_S)
    
    # Try multiple diverse initial solutions
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        max_restarts = int(remaining_time / 12.0)
        for restart in range(max_restarts):
            if time.time() >= deadline:
                break
            
            # Create diverse initial solution
            r = random.random()
            if r < 0.25:
                # Descending order (standard)
                order = list(candidates)
            elif r < 0.5:
                # Random shuffle
                order = list(range(1, 512))
                random.shuffle(order)
            elif r < 0.75:
                # Weighted random: higher weight more likely to come first
                weights = [weight[a] for a in range(1, 512)]
                total_w = sum(weights)
                probs = [w / total_w for w in weights]
                order = random.sample(range(1, 512), 511)
                # Sort by random key weighted by actual weight
                keys = [random.random() + 0.5 * weight[a] for a in order]
                order = [a for _, a in sorted(zip(keys, order), reverse=True)]
            else:
                # Perturbed descending
                order = list(candidates)
                for _ in range(random.randint(5, 20)):
                    i = random.randint(0, len(order) - 2)
                    j = random.randint(i + 1, min(i + 100, len(order) - 1))
                    order[i], order[j] = order[j], order[i]
            
            S2, cm2 = greedy(order)
            time_left = deadline - time.time()
            if time_left > 10:
                S2, cm2, s2 = local_search(S2, cm2, min(deadline, time.time() + min(10.0, time_left)))
                if s2 > best_score + 1e-12:
                    best_S = S2
                    best_mask = cm2
                    best_score = s2
                    best_solution = list(S2)
    
    return best_solution
