import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    # Safe primes: primes > 20. Any product containing one is uniquely
    # divisible by it, so it cannot collide with products from [2..20].
    safe_primes = [23, 29, 31, 37]
    other_candidates = [x for x in range(2, 41) if x not in safe_primes]

    # Initialize product set with safe primes
    P_base = {1}
    for x in safe_primes:
        new_P = set(P_base)
        for p in P_base:
            new_P.add(p * x)
        P_base = new_P

    best_set = list(safe_primes)
    best_size = len(best_set)

    def is_valid(P, x):
        """Check if adding x creates a product collision."""
        for p in P:
            if p * x in P:
                return False
        return True

    def dfs(index, S, P, candidates, time_limit):
        nonlocal best_set, best_size
        if time.time() > time_limit:
            return

        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

        if index == len(candidates):
            return

        # Pruning: can't beat best even if we take all remaining
        if len(S) + (len(candidates) - index) <= best_size:
            return

        x = candidates[index]

        # Option 1: Include x (try first to build larger sets)
        if is_valid(P, x):
            new_P = set(P)
            for p in P:
                new_P.add(p * x)
            dfs(index + 1, S + [x], new_P, candidates, time_limit)

        # Option 2: Exclude x
        dfs(index + 1, S, P, candidates, time_limit)

    # Multiple orderings to explore different search paths
    # Order 1: descending (larger numbers first - less likely to conflict)
    order1 = sorted(other_candidates, reverse=True)

    # Order 2: ascending
    order2 = sorted(other_candidates)

    # Order 3: primes first, then composites (primes are "safer")
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True

    primes_in_range = [x for x in other_candidates if is_prime(x)]
    composites_in_range = [x for x in other_candidates if not is_prime(x)]
    order3 = sorted(primes_in_range, reverse=True) + sorted(composites_in_range, reverse=True)

    # Order 4: by number of prime factors (fewer = safer)
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

    order4 = sorted(other_candidates, key=lambda x: (num_prime_factors(x), -x))

    orderings = [order1, order2, order3, order4]

    # Run DFS with each ordering, allocating time dynamically
    total_time_budget = deadline - start_time - 5  # reserve 5s for polish
    time_per_ordering = total_time_budget / len(orderings)

    for i, ordering in enumerate(orderings):
        if time.time() > deadline - 5:
            break
        t_limit = min(start_time + time_per_ordering * (i + 1) + 2, deadline - 5)
        dfs(0, list(safe_primes), P_base, ordering, t_limit)

    # Final polish: try to add any remaining candidates that fit
    best_set_set = set(best_set)
    remaining = [x for x in range(2, 41) if x not in best_set_set]

    # Build P from current best_set
    P_best = {1}
    for x in best_set:
        new_P = set(P_best)
        for p in P_best:
            new_P.add(p * x)
        P_best = new_P

    # Try adding remaining candidates one by one
    for x in remaining:
        if time.time() > deadline:
            break
        if is_valid(P_best, x):
            new_P = set(P_best)
            for p in P_best:
                new_P.add(p * x)
            P_best = new_P
            best_set.append(x)

    # Ensure we return a list of Python ints
    return [int(x) for x in best_set]
