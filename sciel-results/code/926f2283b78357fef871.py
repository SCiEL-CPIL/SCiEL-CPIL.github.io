def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 80  # safety margin before 90s kill
    n = 128
    
    # Precompute gcd table
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute all bad triples
    # bad_triples[x] = list of (a,b) such that {a,b,x} is a bad triple
    # For efficiency, use a different approach:
    # For each x, and each a < x, check if there exists b < a such that the triple is bad
    
    # Actually, let's precompute for each x, which pairs (a,b) with a<b, a!=x, b!=x form a bad triple with x
    # conflict_pairs[x] = set of frozensets {a,b} that conflict with x
    
    # Better: for each x, precompute which a's conflict with which b's
    # conflict_mask[x][a] = bitmask of b's that form bad triple with x and a
    
    conflict_mask = [[0]*(n+1) for _ in range(n+1)]
    for x in range(1, n+1):
        for a in range(1, n+1):
            if a == x:
                continue
            g_ax = gcd_table[a][x]
            mask = 0
            for b in range(1, n+1):
                if b == a or b == x:
                    continue
                if gcd_table[a][b] == g_ax and gcd_table[b][x] == g_ax:
                    mask |= (1 << b)
            conflict_mask[x][a] = mask
    
    def can_add(x, S):
        """Check if x can be added to set S (bitmask)."""
        temp = S
        while temp:
            lb = temp & (-temp)
            a = lb.bit_length() - 1
            if conflict_mask[x][a] & S:
                return False
            temp ^= lb
        return True
    
    def count_conflicts(x, S):
        """Count how many pairs in S would form bad triples with x."""
        count = 0
        temp = S
        while temp:
            lb = temp & (-temp)
            a = lb.bit_length() - 1
            count += (conflict_mask[x][a] & S).bit_count()
            temp ^= lb
        return count
    
    def greedy_build(order):
        S = 0
        for x in order:
            if can_add(x, S):
                S |= (1 << x)
        return S
    
    # Structural insight: numbers with few distinct prime factors are "safer"
    # because they create fewer gcd patterns
    def distinct_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    # Try multiple greedy strategies
    strategies = []
    
    # Strategy 1: Sort by ascending distinct prime factor count
    order1 = sorted(range(1, n+1), key=lambda x: distinct_prime_factors(x))
    strategies.append(greedy_build(order1))
    
    # Strategy 2: Sort by ascending total prime factors (with multiplicity)
    def total_pf(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    order2 = sorted(range(1, n+1), key=lambda x: total_pf(x))
    strategies.append(greedy_build(order2))
    
    # Strategy 3: Greedy by conflict degree
    conflict_degree = [0]*(n+1)
    for x in range(1, n+1):
        total = 0
        for a in range(1, n+1):
            if a != x:
                total += conflict_mask[x][a].bit_count()
        conflict_degree[x] = total
    order3 = sorted(range(1, n+1), key=lambda x: conflict_degree[x])
    strategies.append(greedy_build(order3))
    
    # Strategy 4: Random orders
    random.seed(42)
    for _ in range(50):
        order = list(range(1, n+1))
        random.shuffle(order)
        strategies.append(greedy_build(order))
    
    best_S = max(strategies, key=lambda s: s.bit_count())
    best_size = best_S.bit_count()
    
    # Simulated annealing / local search
    def local_search(S, time_limit):
        S = S
        current_size = S.bit_count()
        T = 5.0  # temperature
        cooling = 0.995
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < time_limit:
            if no_improve > max_no_improve:
                # Restart with higher temperature occasionally
                if random.random() < 0.1:
                    T = 3.0
                no_improve = 0
                max_no_improve += 200
            
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a random element
                candidates = [x for x in range(1, n+1) if not ((S >> x) & 1)]
                if not candidates:
                    no_improve += 1
                    T *= cooling
                    continue
                x = random.choice(candidates)
                new_S = S | (1 << x)
                # Check validity
                if can_add(x, S):
                    S = new_S
                    current_size += 1
                    no_improve = 0
                else:
                    no_improve += 1
                    T *= cooling
                    
            elif move_type < 0.6:
                # Try to remove a random element
                elements = [x for x in range(1, n+1) if (S >> x) & 1]
                if not elements:
                    no_improve += 1
                    T *= cooling
                    continue
                a = random.choice(elements)
                new_S = S & ~(1 << a)
                S = new_S
                current_size -= 1
                no_improve += 1
                T *= cooling
                
            elif move_type < 0.85:
                # 1-for-1 swap
                elements = [x for x in range(1, n+1) if (S >> x) & 1]
                non_elements = [x for x in range(1, n+1) if not ((S >> x) & 1)]
                if not elements or not non_elements:
                    no_improve += 1
                    T *= cooling
                    continue
                a = random.choice(elements)
                x = random.choice(non_elements)
                new_S = (S & ~(1 << a)) | (1 << x)
                # Check if x can be added after removing a
                S_without_a = S & ~(1 << a)
                if can_add(x, S_without_a):
                    S = new_S
                    # size stays same
                    no_improve += 1
                else:
                    no_improve += 1
                T *= cooling
                
            else:
                # 1-for-2 swap: remove one, try to add two
                elements = [x for x in range(1, n+1) if (S >> x) & 1]
                non_elements = [x for x in range(1, n+1) if not ((S >> x) & 1)]
                if not elements or len(non_elements) < 2:
                    no_improve += 1
                    T *= cooling
                    continue
                a = random.choice(elements)
                S2 = S & ~(1 << a)
                added = 0
                for x in non_elements:
                    if random.random() < 0.3:  # don't check all
                        if can_add(x, S2):
                            S2 |= (1 << x)
                            added += 1
                            if added == 2:
                                break
                if added >= 2:
                    S = S2
                    current_size += added - 1
                    no_improve = 0
                else:
                    no_improve += 1
                T *= cooling
            
            if S.bit_count() > best_size:
                best_size = S.bit_count()
            
            T *= cooling
        
        return S
    
    # Run local search on best initial solution
    if best_S and time.time() < deadline - 10:
        best_S = local_search(best_S, deadline - 5)
    
    # Final cleanup: try to add any remaining elements
    if time.time() < deadline - 2:
        for x in range(1, n+1):
            if (best_S >> x) & 1:
                continue
            if can_add(x, best_S):
                best_S |= (1 << x)
    
    result = [i for i in range(1, n+1) if (best_S >> i) & 1]
    
    if not result:
        result = [1]
    
    return result
