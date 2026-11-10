import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses bitmask-accelerated local search with simulated annealing.
    """
    m = 205
    deadline = time.time() + 580  # Safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = sorted(forbidden)

    # For each element i, compute bitmask of elements that conflict with i
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for f in forbidden_list:
            j = (i - f) % m
            mask |= (1 << j)
            j2 = (i + f) % m
            mask |= (1 << j2)
        mask &= ~(1 << i)
        conflict_mask[i] = mask

    # Precompute: for each element, the set of non-conflicting elements (as bitmask)
    non_conflict_mask = [0] * m
    all_mask = (1 << m) - 1
    for i in range(m):
        non_conflict_mask[i] = all_mask & ~conflict_mask[i] & ~(1 << i)

    def count_conflicts_bitmask(S_mask, x):
        """Count how many elements in S_mask conflict with x using bitmask AND."""
        conflicts = conflict_mask[x] & S_mask
        return bin(conflicts).count('1')

    def can_add_bitmask(S_mask, x):
        """Check if x can be added to set represented by S_mask."""
        return (conflict_mask[x] & S_mask) == 0

    def greedy_init(order):
        """Greedy: process candidates in given order, add if no conflict with selected."""
        S_mask = 0
        for v in order:
            if can_add_bitmask(S_mask, v):
                S_mask |= (1 << v)
        return S_mask

    def mask_to_set(S_mask):
        """Convert bitmask to set of integers."""
        return {i for i in range(m) if S_mask & (1 << i)}

    def local_search(S_mask, time_limit=None, max_iters=50000):
        """
        Improve set via 1-add and 1-swap moves using bitmask operations.
        Returns best mask found.
        """
        start_time = time.time()
        best_S = S_mask
        best_size = bin(S_mask).count('1')
        iters = 0
        no_improve = 0
        max_no_improve = 50

        while iters < max_iters:
            iters += 1
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            if time.time() > deadline:
                break

            improved = False

            # Try 1-add
            for cand in range(m):
                if not (S_mask & (1 << cand)) and can_add_bitmask(S_mask, cand):
                    S_mask |= (1 << cand)
                    improved = True
                    break

            if improved:
                no_improve = 0
                sz = bin(S_mask).count('1')
                if sz > best_size:
                    best_size = sz
                    best_S = S_mask
                continue

            # Try 1-swap: remove one, add one
            s_list = mask_to_set(S_mask)
            random.shuffle(s_list)
            swap_found = False
            for rem in s_list:
                S_without = S_mask & ~(1 << rem)
                # Find candidates that can be added
                for cand in range(m):
                    if not (S_without & (1 << cand)) and can_add_bitmask(S_without, cand):
                        S_mask = S_without | (1 << cand)
                        swap_found = True
                        break
                if swap_found:
                    break

            if swap_found:
                no_improve = 0
                sz = bin(S_mask).count('1')
                if sz > best_size:
                    best_size = sz
                    best_S = S_mask
                continue

            # No improvement: kick
            no_improve += 1
            if no_improve >= max_no_improve or random.random() < 0.15:
                k = random.choice([1, 2, 3, 4])
                if len(s_list) < k:
                    k = max(1, len(s_list) - 1)
                if k > 0 and len(s_list) > 0:
                    to_remove = random.sample(s_list, min(k, len(s_list)))
                    for r in to_remove:
                        S_mask &= ~(1 << r)
                    # Re-greedy fill
                    for cand in range(m):
                        if not (S_mask & (1 << cand)) and can_add_bitmask(S_mask, cand):
                            S_mask |= (1 << cand)
                no_improve = 0
                sz = bin(S_mask).count('1')
                if sz > best_size:
                    best_size = sz
                    best_S = S_mask

        return best_S

    def simulated_annealing(S_mask, time_limit=None, max_iters=100000):
        """
        Simulated annealing to escape local optima.
        Accepts worse solutions with probability exp(-delta/T).
        """
        start_time = time.time()
        T = 1.0
        cooling = 0.9999
        best_S = S_mask
        best_size = bin(S_mask).count('1')
        iters = 0
        current_size = bin(S_mask).count('1')

        while iters < max_iters:
            iters += 1
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            if time.time() > deadline:
                break

            # Random move: either add, remove, or swap
            move_type = random.random()
            new_S = S_mask

            if move_type < 0.3:
                # Try to add a random element
                cand = random.randint(0, m - 1)
                if not (S_mask & (1 << cand)) and can_add_bitmask(S_mask, cand):
                    new_S = S_mask | (1 << cand)
                else:
                    continue  # Can't add, skip
            elif move_type < 0.5:
                # Try to remove a random element
                s_list = mask_to_set(S_mask)
                if not s_list:
                    continue
                rem = random.choice(s_list)
                new_S = S_mask & ~(1 << rem)
            else:
                # Swap: remove one, add one
                s_list = mask_to_set(S_mask)
                if not s_list:
                    continue
                rem = random.choice(s_list)
                S_without = S_mask & ~(1 << rem)
                # Find a candidate to add
                candidates = [c for c in range(m) if not (S_without & (1 << c)) and can_add_bitmask(S_without, c)]
                if not candidates:
                    continue
                cand = random.choice(candidates)
                new_S = S_without | (1 << cand)

            new_size = bin(new_S).count('1')
            delta = new_size - current_size

            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                S_mask = new_S
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_S = S_mask

            T *= cooling

        return best_S

    # Main search
    best_global = 0
    best_global_size = 0

    # Strategy 1: Deterministic greedy
    S_det = 0
    for v in range(m):
        if can_add_bitmask(S_det, v):
            S_det |= (1 << v)
    S_det = local_search(S_det, time_limit=20.0)
    sz = bin(S_det).count('1')
    if sz > best_global_size:
        best_global_size = sz
        best_global = S_det

    # Strategy 2: Multi-start randomized greedy + local search
    rng = random.Random(42)
    n_starts = 300
    for trial in range(n_starts):
        if time.time() > deadline:
            break
        order = list(range(m))
        rng.shuffle(order)
        S_init = greedy_init(order)
        time_left = deadline - time.time()
        if time_left < 5:
            break
        ls_time = min(3.0, time_left * 0.7 / max(1, n_starts - trial))
        S_best = local_search(S_init, time_limit=ls_time)
        sz = bin(S_best).count('1')
        if sz > best_global_size:
            best_global_size = sz
            best_global = S_best

    # Strategy 3: Simulated annealing from best solution
    time_left = deadline - time.time()
    if time_left > 30:
        sa_time = min(120.0, time_left * 0.6)
        S_sa = simulated_annealing(best_global, time_limit=sa_time)
        sz = bin(S_sa).count('1')
        if sz > best_global_size:
            best_global_size = sz
            best_global = S_sa
        # Polish after SA
        time_left = deadline - time.time()
        if time_left > 10:
            S_polish = local_search(best_global, time_limit=time_left - 5)
            sz = bin(S_polish).count('1')
            if sz > best_global_size:
                best_global_size = sz
                best_global = S_polish

    # Strategy 4: Perturb best solution and re-search
    n_perturb = 100
    for trial in range(n_perturb):
        if time.time() > deadline:
            break
        S = best_global
        s_list = mask_to_set(S)
        if len(s_list) < 2:
            break
        n_remove = min(len(s_list), rng.randint(1, 3))
        to_remove = rng.sample(s_list, n_remove)
        for r in to_remove:
            S &= ~(1 << r)
        # Re-greedy fill
        for cand in range(m):
            if not (S & (1 << cand)) and can_add_bitmask(S, cand):
                S |= (1 << cand)
        time_left = deadline - time.time()
        if time_left < 5:
            break
        ls_time = min(2.0, time_left * 0.7 / max(1, n_perturb - trial))
        S_best = local_search(S, time_limit=ls_time)
        sz = bin(S_best).count('1')
        if sz > best_global_size:
            best_global_size = sz
            best_global = S_best

    # Final polish
    time_left = deadline - time.time()
    if time_left > 10:
        S_final = local_search(best_global, time_limit=time_left - 5)
        sz = bin(S_final).count('1')
        if sz > best_global_size:
            best_global_size = sz
            best_global = S_final

    # Convert best mask to sorted array
    result_list = sorted([i for i in range(m) if best_global & (1 << i)])
    result = np.array(result_list, dtype=np.int64)
    return result
