def construct():
    import time
    import random
    from math import gcd

    deadline = time.time() + 85  # safety margin for 90s limit

    N = 128
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Build conflict structure:
    # For each x, find all pairs (a,b) with a<b, a!=x, b!=x such that
    # gcd(x,a) == gcd(x,b) == gcd(a,b)
    CONFLICT = {x: [] for x in range(1, N + 1)}
    conflict_count = {x: 0 for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            g = gcd_table[x][a]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if gcd_table[x][b] == g and gcd_table[a][b] == g:
                    CONFLICT[x].append((a, b))
        conflict_count[x] = len(CONFLICT[x])

    def is_valid_triple_free(selected_set):
        """Check if selected_set has no bad triple. Returns True if valid."""
        elems = sorted(selected_set)
        n = len(elems)
        for i in range(n):
            for j in range(i + 1, n):
                g_ij = gcd_table[elems[i]][elems[j]]
                for k in range(j + 1, n):
                    if gcd_table[elems[i]][elems[k]] == g_ij and gcd_table[elems[j]][elems[k]] == g_ij:
                        return False
        return True

    def can_add(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        for a, b in CONFLICT[x]:
            if a in selected_set and b in selected_set:
                return False
        return True

    def greedy_build(order):
        """Build a valid set by greedily adding elements in the given order."""
        result = set()
        for x in order:
            if can_add(x, result):
                result.add(x)
        return result

    def local_search(current_set, time_budget):
        """Improve current_set via swap-based local search with conflict-aware moves."""
        if not is_valid_triple_free(current_set):
            return current_set

        best_set = set(current_set)
        best_size = len(current_set)
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 5000

        while time.time() - start_time < time_budget and no_improve_count < max_no_improve:
            improved = False
            in_set = list(current_set)
            out_set = [x for x in range(1, N + 1) if x not in current_set]

            # Try all swaps: remove r from current_set, add x
            for r in in_set:
                if time.time() - start_time > time_budget:
                    break
                temp = current_set - {r}
                for x in out_set:
                    # Check if x can be added to temp
                    valid = True
                    for a, b in CONFLICT[x]:
                        if a in temp and b in temp:
                            valid = False
                            break
                    if valid:
                        new_set = temp | {x}
                        if len(new_set) > best_size:
                            best_set = new_set
                            best_size = len(new_set)
                            current_set = new_set
                            improved = True
                            no_improve_count = 0
                            in_set = list(current_set)
                            out_set = [y for y in range(1, N + 1) if y not in current_set]
                            break
                if improved:
                    break

            if not improved:
                no_improve_count += 1

        return best_set

    def multi_start_local_search(time_budget_per_restart):
        """Multiple restarts with different ordering strategies."""
        global_best = set()
        global_best_size = 0

        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break

            # Different ordering strategies
            if restart == 0:
                # Sort by conflict count (fewest conflicts first)
                order = sorted(range(1, N + 1), key=lambda x: conflict_count[x])
            elif restart == 1:
                # Sort by conflict count (most conflicts first)
                order = sorted(range(1, N + 1), key=lambda x: -conflict_count[x])
            elif restart == 2:
                # Random order
                random.seed(restart * 1000 + 42)
                order = list(range(1, N + 1))
                random.shuffle(order)
            elif restart == 3:
                # Primes first, then composites
                def is_prime(n):
                    if n < 2:
                        return False
                    for d in range(2, int(n**0.5) + 1):
                        if n % d == 0:
                            return False
                    return True
                primes = [x for x in range(1, N + 1) if is_prime(x)]
                composites = [x for x in range(1, N + 1) if not is_prime(x)]
                order = primes + composites
            elif restart == 4:
                # Descending order
                order = list(range(N, 0, -1))
            elif restart == 5:
                # Ascending order
                order = list(range(1, N + 1))
            else:
                # Random with seed
                random.seed(restart * 7919 + 13)
                order = list(range(1, N + 1))
                random.shuffle(order)

            # Greedy build
            current_set = greedy_build(order)
            current_size = len(current_set)

            if current_size > global_best_size:
                global_best = set(current_set)
                global_best_size = current_size

            # Local search improvement
            improved_set = local_search(current_set, time_budget_per_restart)
            if len(improved_set) > global_best_size:
                global_best = improved_set
                global_best_size = len(improved_set)

        return global_best

    # Run multi-start local search
    best_set = multi_start_local_search(3.0)

    # Final verification
    if is_valid_triple_free(best_set):
        return list(best_set)
    else:
        # Fallback: return a known valid small set
        return [1]
