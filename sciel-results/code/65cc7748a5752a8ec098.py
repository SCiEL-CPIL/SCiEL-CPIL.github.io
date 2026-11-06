"""Combined strategy: bitmask all-odds base + greedy even addition + aggressive local search + SA polish."""
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
        while True:
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

    evens_to_try = []
    for i in range(n):
        if nums[i] % 2 == 0:
            evens_to_try.append(i)

    evens_to_try.sort(key=lambda x: (degrees[x], x))

    base_mask = odds_mask
    for i in evens_to_try:
        if not (adj[i] & base_mask):
            base_mask |= (1 << i)

    best_mask = base_mask
    best_size = popcount(base_mask)

    # Improved local search with better swap strategies
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

            # Phase 2: Try 1-for-1 swap
            if not improved:
                for rem in range(n):
                    if not (cur & (1 << rem)):
                        continue
                    trial = cur & ~(1 << rem)
                    for add in range(n):
                        if trial & (1 << add):
                            continue
                        if not (adj[add] & trial):
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
                    # Collect candidates that can be added to trial
                    candidates = []
                    for add1 in range(n):
                        if trial & (1 << add1):
                            continue
                        if adj[add1] & trial:
                            continue
                        candidates.append(add1)
                    # Try all pairs from candidates
                    for i_c in range(len(candidates)):
                        a1 = candidates[i_c]
                        for j_c in range(i_c + 1, len(candidates)):
                            a2 = candidates[j_c]
                            if adj[a1] & (1 << a2):
                                continue
                            # Both can be added
                            cur = trial | (1 << a1) | (1 << a2)
                            cur_size += 1
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
                for a in range(min(n_sel, 40)):
                    for b in range(a + 1, min(n_sel, 40)):
                        rem1, rem2 = selected_list[a], selected_list[b]
                        trial = cur & ~(1 << rem1) & ~(1 << rem2)
                        # Collect candidates
                        candidates = []
                        for add in range(n):
                            if trial & (1 << add):
                                continue
                            if adj[add] & trial:
                                continue
                            candidates.append(add)
                        # Try all triples
                        found = False
                        for i_c in range(min(len(candidates), 30)):
                            if found:
                                break
                            a1 = candidates[i_c]
                            for j_c in range(i_c + 1, min(len(candidates), 30)):
                                if found:
                                    break
                                a2 = candidates[j_c]
                                if adj[a1] & (1 << a2):
                                    continue
                                for k_c in range(j_c + 1, min(len(candidates), 30)):
                                    a3 = candidates[k_c]
                                    if adj[a1] & (1 << a3) or adj[a2] & (1 << a3):
                                        continue
                                    cur = trial | (1 << a1) | (1 << a2) | (1 << a3)
                                    cur_size += 1
                                    improved = True
                                    found = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur

        return cur, cur_size

    # Simulated annealing polish
    def simulated_annealing(mask, max_iter=500):
        nonlocal best_mask, best_size
        rng_sa = random.Random(123)
        cur = mask
        cur_size = popcount(cur)
        best_local = cur
        best_local_size = cur_size
        temperature = 2.0
        cooling = 0.9995
        it = 0
        while it < max_iter and time.time() < DEADLINE - 5:
            it += 1
            # Pick a random move
            move_type = rng_sa.random()
            if move_type < 0.4:
                # Single add
                unselected = [v for v in range(n) if not (cur & (1 << v))]
                if not unselected:
                    continue
                v = rng_sa.choice(unselected)
                new_mask = cur | (1 << v)
                delta = 1 - popcount(adj[v] & cur)
            elif move_type < 0.7:
                # Single remove
                selected = [v for v in range(n) if cur & (1 << v)]
                if not selected:
                    continue
                v = rng_sa.choice(selected)
                new_mask = cur & ~(1 << v)
                delta = -1
            elif move_type < 0.9:
                # 1-for-2
                selected = [v for v in range(n) if cur & (1 << v)]
                unselected = [v for v in range(n) if not (cur & (1 << v))]
                if not selected or len(unselected) < 2:
                    continue
                rem = rng_sa.choice(selected)
                trial = cur & ~(1 << rem)
                # Find compatible unselected
                compat = [v for v in unselected if not (adj[v] & trial)]
                if len(compat) < 2:
                    continue
                a1, a2 = rng_sa.sample(compat, 2)
                if adj[a1] & (1 << a2):
                    continue
                new_mask = trial | (1 << a1) | (1 << a2)
                delta = 1
            else:
                # 2-for-3
                selected = [v for v in range(n) if cur & (1 << v)]
                unselected = [v for v in range(n) if not (cur & (1 << v))]
                if len(selected) < 2 or len(unselected) < 3:
                    continue
                rem1, rem2 = rng_sa.sample(selected, 2)
                trial = cur & ~(1 << rem1) & ~(1 << rem2)
                compat = [v for v in unselected if not (adj[v] & trial)]
                if len(compat) < 3:
                    continue
                a1, a2, a3 = rng_sa.sample(compat, 3)
                if adj[a1] & (1 << a2) or adj[a1] & (1 << a3) or adj[a2] & (1 << a3):
                    continue
                new_mask = trial | (1 << a1) | (1 << a2) | (1 << a3)
                delta = 1

            new_size = popcount(new_mask)
            if delta == 0:
                # Check if actually different
                if new_mask == cur:
                    continue

            # Acceptance criterion
            if delta > 0:
                cur = new_mask
                cur_size = new_size
                if cur_size > best_local_size:
                    best_local_size = cur_size
                    best_local = cur
            else:
                # Accept with probability exp(delta / temperature)
                if rng_sa.random() < pow(2.718281828, delta / temperature):
                    cur = new_mask
                    cur_size = new_size

            temperature *= cooling

        if best_local_size > best_size:
            best_size = best_local_size
            best_mask = best_local
        return best_local, best_local_size

    # Run local search on the base
    best_mask, best_size = local_search(best_mask, max_iter=100)

    # Try multiple greedy orderings for even addition
    rng = random.Random(42)
    for restart in range(30):
        if time.time() > DEADLINE - 10:
            break

        if restart == 0:
            order = sorted(range(n), key=lambda x: (degrees[x], x))
            mask = 0
            for v in order:
                if not (adj[v] & mask):
                    mask |= (1 << v)
        elif restart == 1:
            mask = odds_mask
            evens_sorted = sorted([i for i in range(n) if nums[i] % 2 == 0],
                                  key=lambda x: (degrees[x], x))
            for v in evens_sorted:
                if not (adj[v] & mask):
                    mask |= (1 << v)
        elif restart == 2:
            mask = odds_mask
            for v in range(n):
                if nums[v] % 2 == 0 and not (adj[v] & mask):
                    mask |= (1 << v)
        elif restart == 3:
            mask = 0
            for v in range(n):
                if not (adj[v] & mask):
                    mask |= (1 << v)
        elif restart == 4:
            mask = 0
            for v in range(n - 1, -1, -1):
                if not (adj[v] & mask):
                    mask |= (1 << v)
        else:
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

        mask, sz = local_search(mask, max_iter=50)
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Apply SA polish to the best solution
    if time.time() < DEADLINE - 10:
        best_mask, best_size = simulated_annealing(best_mask, max_iter=300)

    # Final aggressive local search
    if time.time() < DEADLINE - 5:
        best_mask, best_size = local_search(best_mask, max_iter=200)

    # Another SA pass if time permits
    if time.time() < DEADLINE - 5:
        best_mask, best_size = simulated_annealing(best_mask, max_iter=200)

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
        return [x for x in range(1, N + 1) if x % 2 == 1]

    return result
# EVOLVE-BLOCK-END
