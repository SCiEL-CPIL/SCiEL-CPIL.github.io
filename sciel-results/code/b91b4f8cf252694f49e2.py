def construct():
    import time
    import random
    from math import gcd

    deadline = time.time() + 82  # safety margin under 90s kill

    def time_left():
        return deadline - time.time()

    # ---------- Precompute gcd table ----------
    N = 128
    gcd_t = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            g = gcd(a, b)
            gcd_t[a][b] = g
            gcd_t[b][a] = g

    # ---------- Build conflict structure ----------
    # For each vertex x, store the set of unordered pairs (u,v) with u<v such that {x,u,v} is bad
    # Also for faster checking: for each x, and each u, the set of v such that {x,u,v} is bad
    # conflicts[x][u] = set of v (v != x, v != u) such that {x,u,v} is a bad triple
    conflicts = [dict() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_t[a][b]
            for c in range(b + 1, N + 1):
                if gcd_t[a][c] == gab and gcd_t[b][c] == gab:
                    # (a,b,c) is bad
                    conflicts[a][b] = conflicts[a].get(b, set()) | {c}
                    conflicts[a][c] = conflicts[a].get(c, set()) | {b}
                    conflicts[b][a] = conflicts[b].get(a, set()) | {c}
                    conflicts[b][c] = conflicts[b].get(c, set()) | {a}
                    conflicts[c][a] = conflicts[c].get(a, set()) | {b}
                    conflicts[c][b] = conflicts[c].get(b, set()) | {a}

    # Degree: number of bad triples containing x
    degree = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in conflicts[a]:
            degree[a] += len(conflicts[a][b])
    # Each triple counted 3 times (once per vertex), so divide by 3 for actual degree
    degree = [d // 3 for d in degree]

    all_numbers = list(range(1, N + 1))

    # ---------- Fast feasibility test using conflicts ----------
    def can_add(x, sel_set):
        """Check if x can be added to sel_set without creating a bad triple."""
        for u in sel_set:
            if u in conflicts[x]:
                # Check if any v in conflicts[x][u] is also in sel_set
                for v in conflicts[x][u]:
                    if v in sel_set:
                        return False
        return True

    # ---------- Constructive seed: prime power based ----------
    def is_prime_power(n):
        """Check if n is a prime power p^k for some prime p and k>=1."""
        if n < 2:
            return False
        # Try each prime factor
        orig = n
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                while n % p == 0:
                    n //= p
                if n == 1:
                    return True
                else:
                    return False
        return True  # n itself is prime (p^1)

    def prime_power_base(n):
        """Return the base prime if n is a prime power, else None."""
        if n < 2:
            return None
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                while n % p == 0:
                    n //= p
                if n == 1:
                    return p
                else:
                    return None
        return n  # n is prime

    # Build constructive seed
    # Strategy: group by prime power base. Within each group, all pairwise GCDs are >= p (the base prime).
    # Between groups with different bases, GCD is 1.
    # Danger: three numbers from three different groups all have GCD 1 -> bad triple.
    # So we can include numbers from at most 2 different prime-power groups if we want to avoid all-GCD-1 triples.
    # But we can also include 1 if careful.
    
    # Better strategy: include all numbers from ONE prime power chain (e.g., all powers of 2),
    # then add numbers from other chains carefully.
    
    # Let's try: all powers of 2 (2,4,8,16,32,64,128) = 7 numbers
    # Then try to add more.
    
    seed_sets = []
    
    # Seed 1: powers of 2
    pp2 = [2**k for k in range(1, 9) if 2**k <= 128]
    seed_sets.append(pp2)
    
    # Seed 2: powers of 3
    pp3 = [3**k for k in range(1, 7) if 3**k <= 128]
    seed_sets.append(pp3)
    
    # Seed 3: mix - powers of 2 and 3
    seed_sets.append(pp2 + pp3)
    
    # Seed 4: all numbers with odd part being a prime power
    # Actually let's just use greedy with different orderings as seeds too
    
    # ---------- Greedy with multiple orderings ----------
    def num_distinct_pf(n):
        cnt = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                cnt += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            cnt += 1
        return cnt

    def total_pf(n):
        cnt = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                cnt += 1
                n //= d
            d += 1
        if n > 1:
            cnt += 1
        return cnt

    def num_divisors(n):
        cnt = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                cnt += 1
                if i != n // i:
                    cnt += 1
            i += 1
        return cnt

    def greedy_from_order(order, initial_set=None):
        """Greedy: start with initial_set if given, then add from order."""
        if initial_set is not None:
            sel = list(initial_set)
            sel_set = set(initial_set)
        else:
            sel = []
            sel_set = set()
        for x in order:
            if x in sel_set:
                continue
            if can_add(x, sel_set):
                sel.append(x)
                sel_set.add(x)
        return sel, sel_set

    best = []
    best_set = set()
    
    # Try constructive seeds
    for seed in seed_sets:
        seed_set = set(seed)
        # Verify seed is valid
        valid = True
        for x in seed:
            for y in seed:
                if x == y:
                    continue
                if y in conflicts[x]:
                    for z in conflicts[x][y]:
                        if z in seed_set:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            # Extend with greedy
            remaining = [x for x in all_numbers if x not in seed_set]
            order = sorted(remaining, key=lambda x: (degree[x], x))
            r, rs = greedy_from_order(order, seed)
            if len(r) > len(best):
                best = r
                best_set = rs

    # Try various greedy orderings
    orderings = [
        sorted(all_numbers, key=lambda x: (degree[x], x)),
        sorted(all_numbers, key=lambda x: (-num_distinct_pf(x), -total_pf(x), x)),
        sorted(all_numbers, key=lambda x: (-total_pf(x), -num_distinct_pf(x), x)),
        sorted(all_numbers, key=lambda x: (-num_divisors(x), x)),
        sorted(all_numbers, key=lambda x: (num_divisors(x), x)),
        sorted(all_numbers, key=lambda x: (-degree[x], x)),
        sorted(all_numbers, key=lambda x: (x,)),
        sorted(all_numbers, key=lambda x: (-x,)),
    ]
    for ord_ in orderings:
        r, rs = greedy_from_order(ord_)
        if len(r) > len(best):
            best = r
            best_set = rs

    # ---------- Local search with kicks ----------
    def local_search(sel, sel_set, max_time=None):
        """Iterative improvement: try additions, then swaps, then 1-for-2."""
        if max_time is None:
            max_time = 2.0
        start = time.time()
        unselected = [x for x in all_numbers if x not in sel_set]
        
        improved = True
        while improved and (time.time() - start) < max_time and time_left() > 3:
            improved = False
            
            # Try single additions
            for x in unselected:
                if can_add(x, sel_set):
                    sel.append(x)
                    sel_set.add(x)
                    unselected = [u for u in unselected if u != x]
                    improved = True
            
            # Try 1-for-1 swaps that enable addition
            for x in unselected:
                if time.time() - start > max_time or time_left() < 3:
                    break
                for y in list(sel):
                    cand_set = sel_set - {y}
                    if can_add(x, cand_set):
                        # Try to add another element z
                        for z in unselected:
                            if z == x:
                                continue
                            if can_add(z, cand_set | {x}):
                                # Net gain: remove y, add x and z -> size +1
                                sel.remove(y)
                                sel.append(x)
                                sel.append(z)
                                sel_set.discard(y)
                                sel_set.add(x)
                                sel_set.add(z)
                                unselected = [u for u in unselected if u not in (x, z)]
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        
        return sel, sel_set

    best, best_set = local_search(best, best_set, max_time=3.0)

    # ---------- Randomized restarts with kicks ----------
    rng = random.Random(42)
    
    for restart in range(50):
        if time_left() < 5:
            break
        
        # Create a perturbed version of best
        # Remove k random elements and re-add greedily
        k = rng.randint(2, min(6, len(best) // 3))
        removed = rng.sample(best, k)
        perturbed = [x for x in best if x not in set(removed)]
        perturbed_set = set(perturbed)
        
        # Add back some random elements
        candidates = [x for x in all_numbers if x not in perturbed_set]
        rng.shuffle(candidates)
        for x in candidates[:k]:
            if can_add(x, perturbed_set):
                perturbed.append(x)
                perturbed_set.add(x)
        
        # Local search on perturbed
        p_sel, p_set = local_search(perturbed, perturbed_set, max_time=1.5)
        
        if len(p_sel) > len(best):
            best = p_sel
            best_set = p_set

    # ---------- Final polish ----------
    best, best_set = local_search(best, best_set, max_time=3.0)

    # Final safety
    if not best:
        best = [1]
    return best
