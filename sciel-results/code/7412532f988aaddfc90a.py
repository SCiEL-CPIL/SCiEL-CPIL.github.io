def construct():
    import random
    from math import gcd
    from itertools import combinations
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute GCD matrix
    gcd_mat = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    def can_add(selected, x):
        """Check if x can be added to selected without creating a bad triple."""
        sel = selected
        n = len(sel)
        # Precompute gcd with x for all elements in selected
        gcds_with_x = [gcd_mat[sel[i]][x] for i in range(n)]
        
        for i in range(n):
            a = sel[i]
            ga = gcds_with_x[i]
            for j in range(i + 1, n):
                b = sel[j]
                if gcd_mat[a][b] == ga == gcds_with_x[j]:
                    return False
        return True
    
    def greedy(order):
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        return selected
    
    # Try multiple strategies
    best = []
    
    # Standard order
    best = greedy(nums)
    
    # Reverse order
    r = greedy(nums[::-1])
    if len(r) > len(best):
        best = r
    
    # Order by number of distinct prime factors (ascending - primes first)
    def num_prime_factors(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                count += 1
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    
    by_pf = sorted(nums, key=num_prime_factors)
    r = greedy(by_pf)
    if len(r) > len(best):
        best = r
    
    by_pf_rev = sorted(nums, key=num_prime_factors, reverse=True)
    r = greedy(by_pf_rev)
    if len(r) > len(best):
        best = r
    
    # Random restarts
    random.seed(42)
    for _ in range(1000):
        shuffled = nums[:]
        random.shuffle(shuffled)
        r = greedy(shuffled)
        if len(r) > len(best):
            best = r
            
    # Simulated Annealing / Local Search
    current = best[:]
    current_set = set(current)
    all_nums_set = set(nums)
    
    # Try to improve via swaps and additions
    for temp in [1.0, 0.5, 0.1, 0.01]:
        for step in range(2000):
            # Try to add a random element from complement
            comp = list(all_nums_set - current_set)
            if comp:
                y = random.choice(comp)
                if can_add(current, y):
                    current.append(y)
                    current_set.add(y)
                    continue
            
            # Try swap
            if current:
                x = random.choice(current)
                comp = list(all_nums_set - current_set)
                if comp:
                    y = random.choice(comp)
                    # Try swapping x with y
                    new_state = [z for z in current if z != x]
                    if can_add(new_state, y):
                        new_state.append(y)
                        # Accept swap (explore)
                        current = new_state
                        current_set.remove(x)
                        current_set.add(y)
                        
        if len(current) > len(best):
            best = current

    return best
