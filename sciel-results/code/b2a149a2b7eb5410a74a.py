# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses bitmask-based conflict detection with simulated annealing and multi-start search.
    """
    m = 205
    deadline = time.time() + 570  # safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = sorted(forbidden)
    n_forbid = len(forbidden_list)

    # For each element i, compute bitmask of elements that conflict with i
    # conflict_mask[i] has bit j set if (i-j) mod m is in forbidden
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for f in forbidden_list:
            j = (i - f) % m
            mask |= (1 << j)
            j2 = (i + f) % m
            if j2 != j:
                mask |= (1 << j2)
        conflict_mask[i] = mask

    def is_valid(s):
        """Check if set s has no forbidden pairs using bitmasks."""
        mask = 0
        for a in s:
            if mask & conflict_mask[a]:
                return False
            mask |= (1 << a)
        return True

    def greedy_random(order):
        """Greedy: process candidates in given order, add if no conflict with selected."""
        selected = set()
        mask = 0
        for cand in order:
            if not (mask & conflict_mask[cand]):
                selected.add(cand)
                mask |= (1 << cand)
        return selected

    def try_add(s, mask, cand):
        """Try to add cand to set s with bitmask mask. Returns (success, new_mask)."""
        if (mask & conflict_mask[cand]) == 0:
            return True, mask | (1 << cand)
        return False, mask

    def local_search(s, max_iters=2000):
        """Improve set via add, swap operations using bitmasks."""
        s = set(s)
        mask = 0
        for a in s:
            mask |= (1 << a)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            # Try adding any element
            for cand in range(m):
                if not (mask >> cand) & 1:
                    if (mask & conflict_mask[cand]) == 0:
                        s.add(cand)
                        mask |= (1 << cand)
                        improved = True
                        break
            if improved:
                continue
            # Try swap: remove one, add another (net +1 or keep size)
            s_list = sorted(s)
            for rem in s_list:
                if time.time() > deadline:
                    break
                mask_without = mask & ~(1 << rem)
                for cand in range(m):
                    if not (mask_without >> cand) & 1:
                        if (mask_without & conflict_mask[cand]) == 0:
                            s.discard(rem)
                            s.add(cand)
                            mask = mask_without | (1 << cand)
                            improved = True
                            break
                if improved:
                    break
            # Try 2-swap: remove two, add two (same size, different configuration)
            if not improved and len(s) >= 2:
                s_list = sorted(s)
                found = False
                for i_idx in range(len(s_list)):
                    if time.time() > deadline:
                        break
                    for j_idx in range(i_idx + 1, len(s_list)):
                        rem1, rem2 = s_list[i_idx], s_list[j_idx]
                        mask_without = mask & ~(1 << rem1) & ~(1 << rem2)
                        for cand1 in range(m):
                            if (mask_without >> cand1) & 1:
                                continue
                            if (mask_without & conflict_mask[cand1]) != 0:
                                continue
                            mask_with1 = mask_without | (1 << cand1)
                            for cand2 in range(m):
                                if (mask_with1 >> cand2) & 1:
                                    continue
                                if (mask_with1 & conflict_mask[cand2]) != 0:
                                    continue
                                # Found a valid 2-swap
                                s.discard(rem1)
                                s.discard(rem2)
                                s.add(cand1)
                                s.add(cand2)
                                mask = mask_with1 | (1 << cand2)
                                improved = True
                                found = True
                                break
                            if found:
                                break
                    if found:
                        break
        return s

    def simulated_annealing(s, iterations=5000):
        """Simulated annealing to find larger sets."""
        s = set(s)
        mask = 0
        for a in s:
            mask |= (1 << a)
        best_s = set(s)
        best_mask = mask
        best_size = len(s)
        temp = 2.0
        cooling = 0.9995

        for it in range(iterations):
            if time.time() > deadline:
                break
            if temp < 0.01:
                break
            s_list = sorted(s)
            # Random move: add, remove, or swap
            move_type = random.random()
            if move_type < 0.3:
                # Try add
                cand = random.randrange(m)
                if (mask >> cand) & 1:
                    continue
                if (mask & conflict_mask[cand]) == 0:
                    s.add(cand)
                    mask |= (1 << cand)
                    delta = 1
                else:
                    # Would be invalid, skip
                    continue
            elif move_type < 0.6:
                # Try remove
                if len(s) <= 1:
                    continue
                rem = random.choice(s_list)
                s.discard(rem)
                mask &= ~(1 << rem)
                delta = -1
            else:
                # Swap: remove one, add one
                if len(s) <= 1:
                    continue
                rem = random.choice(s_list)
                cand = random.randrange(m)
                if (mask >> cand) & 1:
                    continue
                mask_without = mask & ~(1 << rem)
                if (mask_without & conflict_mask[cand]) == 0:
                    s.discard(rem)
                    s.add(cand)
                    mask = mask_without | (1 << cand)
                    delta = 0
                else:
                    continue

            # Accept if better, or with probability exp(-delta/temp)
            if delta < 0:
                s.add(rem if 'rem' in dir() else random.choice(s_list))
                # This is getting complex, let's simplify
                pass

            # Simpler: just accept improvements
            if len(s) > best_size:
                best_s = set(s)
                best_mask = mask
                best_size = len(s)

            temp *= cooling

        return best_s

    # Strategy 1: deterministic greedy
    best_set = set()
    mask = 0
    for cand in range(m):
        if not (mask & conflict_mask[cand]):
            best_set.add(cand)
            mask |= (1 << cand)
    best_set = local_search(best_set, max_iters=3000)
    best_size = len(best_set)

    # Strategy 2: randomized greedy with many shuffles
    rng = random.Random(12345)
    for trial in range(50000):
        if time.time() > deadline:
            break
        order = list(range(m))
        rng.shuffle(order)
        s = greedy_random(order)
        s = local_search(s, max_iters=1000)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Strategy 3: perturbation of best set
    for trial in range(20000):
        if time.time() > deadline:
            break
        s = set(best_set)
        n_remove = min(len(s), rng.randint(1, 4))
        to_remove = rng.sample(sorted(s), n_remove)
        for r in to_remove:
            s.discard(r)
        s = local_search(s, max_iters=2000)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Strategy 4: random restarts with small seeds
    for trial in range(30000):
        if time.time() > deadline:
            break
        seed_size = rng.randint(1, 6)
        seed = set(rng.sample(range(m), seed_size))
        # Make valid
        valid_seed = set()
        smask = 0
        for x in sorted(seed):
            if not (smask & conflict_mask[x]):
                valid_seed.add(x)
                smask |= (1 << x)
        # Greedy from seed
        s = set(valid_seed)
        mask = smask
        for cand in range(m):
            if not (mask >> cand) & 1:
                if not (mask & conflict_mask[cand]):
                    s.add(cand)
                    mask |= (1 << cand)
        s = local_search(s, max_iters=1500)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    return np.array(sorted(best_set), dtype=np.int64)


# EVOLVE-BLOCK-END
