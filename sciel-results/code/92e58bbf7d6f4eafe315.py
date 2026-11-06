"""Sum-free subset finder using bitmask-based local search with diverse multi-start greedy."""
import time
import random
from collections import defaultdict


def construct():
    deadline = time.time() + 145

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(elems):
        s = set(elems)
        for a in elems:
            for b in elems:
                if a + b in s:
                    return False
        return True

    def find_sum_free(elements, time_budget=45.0):
        if not elements:
            return []

        n = len(elements)
        elem_set = set(elements)
        idx_of = {x: i for i, x in enumerate(elements)}
        my_deadline = time.time() + time_budget

        # Precompute conflict structure
        # For each index k: pairs (i,j) where elements[i]+elements[j]==elements[k]
        as_sum = [set() for _ in range(n)]   # as_sum[k] = set of (i,j)
        as_left = [set() for _ in range(n)]  # as_left[i] = set of (j,k)
        as_right = [set() for _ in range(n)] # as_right[j] = set of (i,k)

        for i in range(n):
            for j in range(i, n):
                s_val = elements[i] + elements[j]
                k = idx_of.get(s_val)
                if k is not None:
                    as_sum[k].add((i, j))
                    as_left[i].add((j, k))
                    as_right[j].add((i, k))

        def count_conflicts(active):
            """Count violated sum-free constraints in active set."""
            count = 0
            for k in active:
                for (i, j) in as_sum[k]:
                    if i in active and j in active:
                        count += 1
            return count

        def can_add(idx, active):
            """Check if adding idx to active creates any conflict."""
            for (i, j) in as_sum[idx]:
                if i in active and j in active:
                    return False
            for (j, k) in as_left[idx]:
                if j in active and k in active:
                    return False
            for (i, k) in as_right[idx]:
                if i in active and k in active:
                    return False
            return True

        def greedy_completion(active):
            """Greedy add any element that fits, until no more can be added."""
            active = set(active)
            improved = True
            while improved:
                improved = False
                for idx in range(n):
                    if idx not in active and can_add(idx, active):
                        active.add(idx)
                        improved = True
            return active

        # ========== MODEL-BASED INITIALIZATION ==========
        # Try multiple parameterized models to find good starting points

        def model_sign_partition():
            """Split by sign: take all positive or all negative (whichever is larger)."""
            pos = [i for i in range(n) if elements[i] > 0]
            neg = [i for i in range(n) if elements[i] < 0]
            best = []
            for subset in [pos, neg, range(n)]:
                active = set(subset)
                active = greedy_completion(active)
                if len(active) > len(best):
                    best = active
            return best

        def model_residue(m):
            """Try all residue classes mod m, keep the largest sum-free one."""
            best = []
            for r in range(m):
                active = set(i for i in range(n) if elements[i] % m == r)
                active = greedy_completion(active)
                if len(active) > len(best):
                    best = active
            return best

        def model_magnitude_split():
            """Split by magnitude: take elements in a range [lo, hi]."""
            abs_vals = sorted(range(n), key=lambda i: abs(elements[i]))
            best = []
            for frac_lo in [0, 0.25, 0.5]:
                for frac_hi in [0.75, 1.0]:
                    lo_idx = int(frac_lo * n)
                    hi_idx = int(frac_hi * n)
                    active = set(abs_vals[lo_idx:hi_idx])
                    active = greedy_completion(active)
                    if len(active) > len(best):
                        best = active
            return best

        def model_greedy_random(trial_seed):
            """Randomized greedy addition."""
            rng = random.Random(trial_seed)
            order = list(range(n))
            rng.shuffle(order)
            active = set()
            for idx in order:
                if can_add(idx, active):
                    active.add(idx)
            return active

        def model_greedy_weighted(trial_seed):
            """Greedy with random weights - pick element with best weight/conflict ratio."""
            rng = random.Random(trial_seed)
            remaining = set(range(n))
            active = set()
            weights = {i: rng.random() for i in range(n)}
            while remaining and time.time() < my_deadline:
                best_idx = None
                best_score = -1
                for idx in list(remaining):
                    if can_add(idx, active):
                        # Score: prefer elements with fewer future conflicts
                        conflict_deg = len(as_left[idx]) + len(as_right[idx]) + len(as_sum[idx])
                        score = weights[idx] / (1 + conflict_deg)
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                if best_idx is None:
                    break
                active.add(best_idx)
                remaining.discard(best_idx)
            return active

        # ========== LOCAL SEARCH ==========
        def local_search_1flip(initial_active, max_iters=5000):
            """1-flip local search: remove elements to fix conflicts, add to grow."""
            active = set(initial_active)
            current_conflicts = count_conflicts(active)
            best_active = set(active)
            best_size = len(active)
            rng = random.Random(99999)

            for iteration in range(max_iters):
                if time.time() > my_deadline:
                    break

                if current_conflicts == 0:
                    # Try to add elements
                    added = False
                    # Try in random order for diversity
                    for idx in range(n):
                        if idx not in active and can_add(idx, active):
                            active.add(idx)
                            added = True
                            break
                    if not added:
                        # Local optimum - try random restarts of removal
                        if len(active) > best_size:
                            best_size = len(active)
                            best_active = set(active)
                        # Try removing a random element and re-adding
                        if len(active) > 2:
                            to_remove = rng.choice(list(active))
                            active.discard(to_remove)
                            # Re-add greedily
                            active = greedy_completion(active)
                            current_conflicts = count_conflicts(active)
                    else:
                        current_conflicts = count_conflicts(active)
                        if len(active) > best_size:
                            best_size = len(active)
                            best_active = set(active)
                    continue

                # Find a conflict and resolve it
                conflict_found = None
                for k in active:
                    for (i, j) in as_sum[k]:
                        if i in active and j in active:
                            conflict_found = (i, j, k)
                            break
                    if conflict_found:
                        break

                if not conflict_found:
                    break

                i, j, k = conflict_found
                candidates = [i, j, k]
                rng.shuffle(candidates)

                best_new_conflicts = current_conflicts
                best_remove = None
                for c in candidates:
                    new_active = active - {c}
                    new_conflicts = count_conflicts(new_active)
                    if new_conflicts < best_new_conflicts:
                        best_new_conflicts = new_conflicts
                        best_remove = c

                if best_remove is not None:
                    active.remove(best_remove)
                    current_conflicts = best_new_conflicts
                    # Greedy complete
                    active = greedy_completion(active)
                    current_conflicts = count_conflicts(active)
                    if len(active) > best_size and current_conflicts == 0:
                        best_size = len(active)
                        best_active = set(active)
                else:
                    # No single removal helps; try swap
                    swapped = False
                    for c in candidates:
                        new_active = active - {c}
                        for add_c in range(n):
                            if add_c not in new_active:
                                test = new_active | {add_c}
                                if count_conflicts(test) < current_conflicts:
                                    active = test
                                    current_conflicts = count_conflicts(active)
                                    active = greedy_completion(active)
                                    current_conflicts = count_conflicts(active)
                                    swapped = True
                                    break
                        if swapped:
                            break
                    if not swapped:
                        # Stuck, try random perturbation
                        if len(active) > 1:
                            to_remove = rng.choice(list(active))
                            active.discard(to_remove)
                            active = greedy_completion(active)
                            current_conflicts = count_conflicts(active)

            return best_active

        def local_search_2swap(initial_active, max_iters=2000):
            """2-swap local search: remove 1, add 2 (net gain of 1)."""
            active = set(initial_active)
            best_active = set(active)
            best_size = len(active)
            rng = random.Random(77777)

            for _ in range(max_iters):
                if time.time() > my_deadline:
                    break

                improved = False
                inactive = [x for x in range(n) if x not in active]
                if len(inactive) < 2 or len(active) < 1:
                    break

                # Try removing each active element and adding two inactive ones
                for remove_c in list(active):
                    if time.time() > my_deadline:
                        break
                    new_active = active - {remove_c}
                    # Find pairs of inactive that can both be added
                    addable = [x for x in inactive if can_add(x, new_active)]
                    if len(addable) >= 2:
                        # Try pairs
                        for a in range(min(len(addable), 20)):
                            for b in range(a + 1, min(len(addable), 20)):
                                i1, i2 = addable[a], addable[b]
                                test = new_active | {i1, i2}
                                if count_conflicts(test) == 0:
                                    active = test
                                    if len(active) > best_size:
                                        best_size = len(active)
                                        best_active = set(active)
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break

                if not improved:
                    # Try 1-for-1 swap
                    for remove_c in list(active):
                        new_active = active - {remove_c}
                        for add_c in inactive:
                            if can_add(add_c, new_active):
                                test = new_active | {add_c}
                                if count_conflicts(test) == 0 and len(test) > len(active) - 1:
                                    # Same size but different composition - try if it helps future
                                    # Only accept if it enables growth
                                    active = test
                                    improved = True
                                    break
                        if improved:
                            break
                    if not improved:
                        break

            return best_active

        # ========== RUN ALL MODELS AND SEARCH ==========
        best_result_indices = set()

        # Model 1: Sign partition
        if time.time() < my_deadline:
            m1 = model_sign_partition()
            if len(m1) > len(best_result_indices):
                best_result_indices = set(m1)

        # Model 2: Residue classes mod small m
        if time.time() < my_deadline:
            for m in [3, 5, 7]:
                m2 = model_residue(m)
                if len(m2) > len(best_result_indices):
                    best_result_indices = set(m2)

        # Model 3: Magnitude split
        if time.time() < my_deadline:
            m3 = model_magnitude_split()
            if len(m3) > len(best_result_indices):
                best_result_indices = set(m3)

        # Model 4: Randomized greedy (multiple trials)
        num_random_trials = 10 if n <= 120 else 5
        for t in range(num_random_trials):
            if time.time() > my_deadline:
                break
            m4 = model_greedy_random(1000 + t)
            if len(m4) > len(best_result_indices):
                best_result_indices = set(m4)

        # Model 5: Weighted greedy
        for t in range(3):
            if time.time() > my_deadline:
                break
            m5 = model_greedy_weighted(2000 + t)
            if len(m5) > len(best_result_indices):
                best_result_indices = set(m5)

        # Local search from best model
        if best_result_indices and time.time() < my_deadline:
            ls_result = local_search_1flip(best_result_indices)
            if len(ls_result) > len(best_result_indices):
                best_result_indices = set(ls_result)

        # 2-swap local search
        if best_result_indices and time.time() < my_deadline:
            sw_result = local_search_2swap(best_result_indices)
            if len(sw_result) > len(best_result_indices):
                best_result_indices = set(sw_result)

        # More local search iterations from improved best
        if best_result_indices and time.time() < my_deadline:
            ls_result2 = local_search_1flip(best_result_indices, max_iters=3000)
            if len(ls_result2) > len(best_result_indices):
                best_result_indices = set(ls_result2)

        # Final: greedy complete
        if best_result_indices:
            best_result_indices = greedy_completion(best_result_indices)

        # Convert to values
        best_result = [elements[i] for i in best_result_indices]

        # Final verification - fix any remaining conflicts
        result_set = set(best_result)
        for _ in range(3):  # Iterate a few times to catch cascading issues
            to_remove = set()
            for a in best_result:
                for b in best_result:
                    s = a + b
                    if s in result_set:
                        to_remove.add(s)
            if to_remove:
                result_set -= to_remove
                best_result = list(result_set)
            else:
                break

        # If verification failed, fall back to a safe subset
        if not is_sum_free(best_result):
            # Aggressive removal
            result_set = set(best_result)
            changed = True
            while changed:
                changed = False
                for a in list(result_set):
                    for b in list(result_set):
                        if a + b in result_set:
                            result_set.discard(a + b)
                            changed = True
                best_result = list(result_set)
            if not is_sum_free(best_result):
                # Last resort: take only the largest elements (they can't be sums of smaller ones in a bounded set)
                best_result = [x for x in sorted(best_result, key=lambda x: -abs(x))[:len(best_result)//2]]

        return best_result

    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = deadline - time.time()
        if remaining < 5:
            # Fallback: use odd numbers or top half
            elems = instance(size, seed)
            # Simple heuristic: take elements with odd absolute values
            results.append([x for x in elems if abs(x) % 2 == 1])
            continue

        # Allocate time proportional to problem size
        time_budget = min(remaining - 3, 50.0)
        elems = instance(size, seed)
        result = find_sum_free(elems, time_budget=time_budget)
        results.append(result)

    return results
