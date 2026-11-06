"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # safety margin before 90s kill

    N = 256
    nums = list(range(1, N + 1))
    n = N

    # Build conflict graph using bitmasks for speed
    # adj[i] is a bitmask where bit j is set if nums[i] conflicts with nums[j]
    adj = [0] * n
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            bj = nums[j]
            s = ai + bj
            p = ai * bj
            if p % s == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Identify odds and evens
    odds_mask = 0
    evens_mask = 0
    for i in range(n):
        if nums[i] % 2 == 1:
            odds_mask |= (1 << i)
        else:
            evens_mask |= (1 << i)

    # Precompute: for each even index, which odds does it conflict with?
    even_conf_odds = [0] * n  # even_conf_odds[i] = bitmask of odd indices conflicting with i
    for i in range(n):
        if nums[i] % 2 == 0:
            even_conf_odds[i] = adj[i] & odds_mask

    # Precompute: for each even index, which evens does it conflict with?
    even_conf_evens = [0] * n  # even_conf_evens[i] = bitmask of even indices conflicting with i
    for i in range(n):
        if nums[i] % 2 == 0:
            even_conf_evens[i] = adj[i] & evens_mask

    # An even is "odd-safe" if it has NO conflict with any odd
    odd_safe_evens_mask = 0
    for i in range(n):
        if nums[i] % 2 == 0 and even_conf_odds[i] == 0:
            odd_safe_evens_mask |= (1 << i)

    def popcount(x):
        return bin(x).count('1')

    def greedy_mis_on_mask(candidates_mask, adj_list, order):
        """Greedy MIS on the subgraph induced by candidates_mask, using given vertex order."""
        selected = 0
        for v in order:
            if not (candidates_mask & (1 << v)):
                continue
            if not (adj_list[v] & selected):
                selected |= (1 << v)
        return selected

    def try_greedy_full(order, base_mask=odds_mask):
        """Given an ordering of all evens, greedily add those compatible with base and selected evens."""
        selected = base_mask
        selected_evens = 0
        for v in order:
            if not (evens_mask & (1 << v)):
                continue
            # Check conflicts with base (odds)
            if even_conf_odds[v] & base_mask:
                continue
            # Check conflicts with selected evens
            if even_conf_evens[v] & selected_evens:
                continue
            selected |= (1 << v)
            selected_evens |= (1 << v)
        return selected

    # Get list of even indices
    even_indices = [i for i in range(n) if nums[i] % 2 == 0]
    odd_safe_even_indices = [i for i in even_indices if odd_safe_evens_mask & (1 << i)]

    # Strategy 1: Start with all odds, add odd-safe evens via greedy independent set
    best_mask = odds_mask
    best_size = popcount(best_mask)

    if odd_safe_even_indices:
        orderings = [
            sorted(odd_safe_even_indices, key=lambda i: popcount(even_conf_evens[i])),
            sorted(odd_safe_even_indices, key=lambda i: -popcount(even_conf_evens[i])),
            sorted(odd_safe_even_indices, key=lambda i: nums[i]),
            sorted(odd_safe_even_indices, key=lambda i: -nums[i]),
            sorted(odd_safe_even_indices, key=lambda i: nums[i].bit_count()),
            sorted(odd_safe_even_indices, key=lambda i: -nums[i].bit_count()),
        ]
        for order in orderings:
            if time.time() > DEADLINE - 5:
                break
            indep = greedy_mis_on_mask(odd_safe_evens_mask, even_conf_evens, order)
            candidate = odds_mask | indep
            sz = popcount(candidate)
            if sz > best_size:
                best_size = sz
                best_mask = candidate

    # Strategy 2: Full greedy over all evens with various orderings
    all_orders = [
        sorted(even_indices, key=lambda i: (popcount(even_conf_odds[i]), popcount(even_conf_evens[i]), nums[i])),
        sorted(even_indices, key=lambda i: (popcount(even_conf_evens[i]), popcount(even_conf_odds[i]), nums[i])),
        sorted(even_indices, key=lambda i: nums[i]),
        sorted(even_indices, key=lambda i: -nums[i]),
        sorted(even_indices, key=lambda i: nums[i].bit_count()),
        sorted(even_indices, key=lambda i: -nums[i].bit_count()),
    ]
    for order in all_orders:
        if time.time() > DEADLINE - 5:
            break
        s = try_greedy_full(order)
        sz = popcount(s)
        if sz > best_size:
            best_size = sz
            best_mask = s

    # Local search using bitmasks
    def local_search(mask, max_iter=200):
        if time.time() > DEADLINE - 5:
            return mask, popcount(mask)
        cur = mask
        cur_size = popcount(cur)
        cur_odds = cur & odds_mask
        cur_evens = cur & evens_mask
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < DEADLINE - 5:
            improved = False
            iterations += 1

            # Try adding each unselected even
            for e in even_indices:
                if cur & (1 << e):
                    continue
                if even_conf_odds[e] & cur_odds:
                    continue
                if even_conf_evens[e] & cur_evens:
                    continue
                cur |= (1 << e)
                cur_evens |= (1 << e)
                cur_size += 1
                improved = True
                break

            # Try 1-for-k swaps: remove one even, add multiple others
            if not improved and time.time() < DEADLINE - 5:
                for rem in even_indices:
                    if not (cur & (1 << rem)):
                        continue
                    trial_evens = cur_evens & ~(1 << rem)
                    added = 0
                    for e in even_indices:
                        if e == rem:
                            continue
                        if trial_evens & (1 << e):
                            continue
                        if even_conf_odds[e] & cur_odds:
                            continue
                        if even_conf_evens[e] & trial_evens:
                            continue
                        trial_evens |= (1 << e)
                        added += 1
                    if added > 1:
                        cur = cur_odds | trial_evens
                        cur_evens = trial_evens
                        cur_size = popcount(cur)
                        improved = True
                        break

            # Try 2-for-k swaps: remove two evens, add multiple others
            if not improved and time.time() < DEADLINE - 5:
                evens_list = [e for e in even_indices if cur & (1 << e)]
                for i in range(len(evens_list)):
                    for j in range(i + 1, len(evens_list)):
                        rem1, rem2 = evens_list[i], evens_list[j]
                        trial_evens = cur_evens & ~(1 << rem1) & ~(1 << rem2)
                        added = 0
                        for e in even_indices:
                            if e == rem1 or e == rem2:
                                continue
                            if trial_evens & (1 << e):
                                continue
                            if even_conf_odds[e] & cur_odds:
                                continue
                            if even_conf_evens[e] & trial_evens:
                                continue
                            trial_evens |= (1 << e)
                            added += 1
                        if added > 2:
                            cur = cur_odds | trial_evens
                            cur_evens = trial_evens
                            cur_size = popcount(cur)
                            improved = True
                            break
                    if improved:
                        break

        return cur, cur_size

    best_mask, best_size = local_search(best_mask)

    # Strategy 3: Random restarts with perturbation
    rng = random.Random(42)
    for restart in range(50):
        if time.time() > DEADLINE - 10:
            break
        # Create a perturbed ordering
        shuffled = even_indices[:]
        for i in range(len(shuffled) - 1, 0, -1):
            if rng.random() < 0.4:
                j = rng.randint(0, i)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        s = try_greedy_full(shuffled)
        # Local search
        s, sz = local_search(s, max_iter=30)
        if sz > best_size:
            best_size = sz
            best_mask = s

    # Final local search
    if time.time() < DEADLINE - 5:
        best_mask, best_size = local_search(best_mask, max_iter=100)

    # Convert mask to list of numbers
    result = [nums[i] for i in range(n) if best_mask & (1 << i)]

    # Verify the result
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fall back to odds only (always valid)
        return [nums[i] for i in range(n) if odds_mask & (1 << i)]

    return result
# EVOLVE-BLOCK-END
