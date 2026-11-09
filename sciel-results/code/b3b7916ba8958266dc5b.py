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
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                a, b = sel[i], sel[j]
                if gcd_mat[a][b] == gcd_mat[a][x] == gcd_mat[b][x]:
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
    for _ in range(5000):
        shuffled = nums[:]
        random.shuffle(shuffled)
        r = greedy(shuffled)
        if len(r) > len(best):
            best = r
    
    # Local improvement: try swapping elements out for better ones
    improved = best[:]
    for _ in range(2000):
        improved = best[:]
        sel_set = set(improved)
        for x in nums:
            if x in sel_set:
                continue
            # Try adding x and removing one element
            for y in improved:
                new_set = [z for z in improved if z != y]
                new_set.append(x)
                if all(can_add(new_set[:i], new_set[i]) for i in range(len(new_set))):
                    # Verify full validity
                    valid = True
                    for i in range(len(new_set)):
                        for j in range(i+1, len(new_set)):
                            a, b = new_set[i], new_set[j]
                            for k in range(len(new_set)):
                                if k == i or k == j:
                                    continue
                                c = new_set[k]
                                if gcd_mat[a][b] == gcd_mat[a][c] == gcd_mat[b][c]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid and len(new_set) > len(improved):
                        improved = new_set
                        break
                # Also try just adding x without removing
                if can_add(improved, x):
                    improved = improved + [x]
                    break
        if len(improved) > len(best):
            best = improved
        else:
            break
    
    return best
