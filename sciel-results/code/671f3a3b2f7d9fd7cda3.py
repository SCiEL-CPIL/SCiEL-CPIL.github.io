"""Combined strategy: bitmask all-odds base + greedy even addition + aggressive local search."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # safety margin before 90s kill

    N = 256
    nums = list(range(1, N + 1))
    n = N

    # Build adjacency bitmasks: adj[i] has bit j set if nums[i] and nums[j] conflict
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

    degrees = [bin(adj[i]).count('1') for i in range(n)]

    # Helper: count bits
    def popcount(x):
        return bin(x).count('1')

    # Helper: greedy fill - add all compatible vertices from a candidate set
    def greedy_fill(selected_mask, candidates_mask):
        """Add as many vertices from candidates as possible to selected."""
        result = selected_mask
        # Repeatedly find a compatible vertex and add it
        while True:
            # Find candidates that don't conflict with result
            compatible = 0
            temp = candidates_mask
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                if not (adj[v] & result):
                    compatible |= lb
                temp ^= lb
            if not compatible:
                break
            # Pick the one with lowest degree for better packing
            best_v = -1
            best_deg = n + 1
            temp = compatible
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                if degrees[v] < best_deg:
                    best_deg = degrees[v]
                    best_v = v
                temp ^= lb
            result |= (1 << best_v)
            candidates_mask ^= (1 << best_v)
        return result

    # Strategy 1: Start with all odds, greedily add compatible evens
    odds_mask = 0
    for i in range(n):
        if nums[i] % 2 == 1:
            odds_mask |= (1 << i)

    # Greedily add evens to the odds base
    evens_to_try = []
    for i in range(n):
        if nums[i] % 2 == 0:
            evens_to_try.append(i)

    # Sort evens by degree (ascending) for better greedy
    evens_to_try.sort(key=lambda x: (degrees[x], x))

    base_mask = odds_mask
    for i in evens_to_try:
        if not (adj[i] & base_mask):
            base_mask |= (1 << i)

    best_mask = base_mask
    best_size = popcount(base_mask)

    # Local search: try 1-for-1, 1-for-2, 2-for-3 swaps
    def local_search(mask, max_iter=200):
        nonlocal best_mask, best_size
        cur = mask
        cur_size = popcount(cur)
        improved = True
        it = 0
        while improved and it < max_iter and time.time() < DEADLINE - 5:
            improved = False
            it += 1

            # Phase 1: Try adding any compatible vertex
            for v in range(n):
                if cur & (1 << v):
                    continue
                if not (adj[v] & cur):
                    cur |= (1 << v)
                    cur_size += 1
                    improved = True
                    break

            # Phase 2: Try 1-for-1 swap (remove one, add one different)
            if not improved:
                for rem in range(n):
                    if not (cur & (1 << rem)):
                        continue
                    trial = cur & ~(1 << rem)
                    # Try to add any vertex not in trial
                    for add in range(n):
                        if trial & (1 << add):
                            continue
                        if not (adj[add] & trial):
                            # This would be size-neutral but might enable future improvements
                            # Only do it if it changes the set (add != rem)
                            if add != rem:
                                cur = trial | (1 << add)
                                improved = True
                                break
                    if improved:
                        break

            # Phase 3: Try 1-for-2 swap (remove one selected, add two unselected)
            if not improved and time.time() < DEADLINE - 5:
                for rem in range(n):
                    if not (cur & (1 << rem)):
                        continue
                    trial = cur & ~(1 << rem)
                    added = 0
                    for add1 in range(n):
                        if trial & (1 << add1):
                            continue
                        if adj[add1] & trial:
                            continue
                        for add2 in range(n):
                            if add2 == add1:
                                continue
                            if trial & (1 << add2):
                                continue
                            if adj[add2] & trial:
                                continue
                            if adj[add1] & (1 << add2):
                                continue
                            # Both can be added
                            new_size = cur_size - 1 + 2
                            if new_size > cur_size:
                                cur = trial | (1 << add1) | (1 << add2)
                                cur_size = new_size
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

            # Phase 4: Try 2-for-3 swap
            if not improved and time.time() < DEADLINE - 10:
                selected_list = [i for i in range(n) if cur & (1 << i)]
                n_sel = len(selected_list)
                for a in range(min(n_sel, 30)):
                    for b in range(a + 1, min(n_sel, 30)):
                        rem1, rem2 = selected_list[a], selected_list[b]
                        trial = cur & ~(1 << rem1) & ~(1 << rem2)
                        added = 0
                        for add1 in range(n):
                            if trial & (1 << add1):
                                continue
                            if adj[add1] & trial:
                                continue
                            for add2 in range(n):
                                if add2 == add1 or trial & (1 << add2):
                                    continue
                                if adj[add2] & trial:
                                    continue
                                if adj[add1] & (1 << add2):
                                    continue
                                for add3 in range(n):
                                    if add3 == add1 or add3 == add2 or trial & (1 << add3):
                                        continue
                                    if adj[add3] & trial:
                                        continue
                                    if adj[add1] & (1 << add3) or adj[add2] & (1 << add3):
                                        continue
                                    new_size = cur_size - 2 + 3
                                    if new_size > cur_size:
                                        cur = trial | (1 << add1) | (1 << add2) | (1 << add3)
                                        cur_size = new_size
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur

        return cur, cur_size

    # Run local search on the base
    best_mask, best_size = local_search(best_mask, max_iter=100)

    # Try multiple greedy orderings for even addition
    rng = random.Random(42)
    for restart in range(30):
        if time.time() > DEADLINE - 10:
            break

        # Different strategies for building the base
        if restart == 0:
            # Pure degree-ordered greedy on all vertices
            order = sorted(range(n), key=lambda x: (degrees[x], x))
            mask = 0
            for v in order:
                if not (adj[v] & mask):
                    mask |= (1 << v)
        elif restart == 1:
            # Odds first, then evens by degree
            mask = odds_mask
            evens_sorted = sorted([i for i in range(n) if nums[i] % 2 == 0],
                                  key=lambda x: (degrees[x], x))
            for v in evens_sorted:
                if not (adj[v] & mask):
                    mask |= (1 << v)
        elif restart == 2:
            # Odds first, then evens by value
            mask = odds_mask
            for v in range(n):
                if nums[v] % 2 == 0 and not (adj[v] & mask):
                    mask |= (1 << v)
        elif restart == 3:
            # All by value ascending
            mask = 0
            for v in range(n):
                if not (adj[v] & mask):
                    mask |= (1 << v)
        elif restart == 4:
            # All by value descending
            mask = 0
            for v in range(n - 1, -1, -1):
                if not (adj[v] & mask):
                    mask |= (1 << v)
        else:
            # Random order
            order = list(range(n))
            rng.shuffle(order)
            mask = 0
            for v in order:
                if not (adj[v] & mask):
                    mask |= (1 << v)

        sz = popcount(mask)
        if sz > best_size:
            best_size = sz
            best_mask = mask

        # Local search on this result
        mask, sz = local_search(mask, max_iter=50)
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Final aggressive local search
    if time.time() < DEADLINE - 5:
        best_mask, best_size = local_search(best_mask, max_iter=200)

    # Convert to result list
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(nums[i])
    result.sort()

    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break

    if not valid or not result:
        # Fallback: return all odds (should be valid)
        return [x for x in range(1, N + 1) if x % 2 == 1]

    return result
# EVOLVE-BLOCK-END
