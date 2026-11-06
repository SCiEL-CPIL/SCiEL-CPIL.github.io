import time
import random

def construct():
    start = time.time()
    DEADLINE = 140.0  # safety margin (limit is 150s)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                if a + b in Sset:
                    return False
        return True

    def can_add(x, Sset, S_list):
        """Check if x can be added to S without creating a sum-free violation."""
        for a in S_list:
            if (x + a) in Sset or (x - a) in Sset:
                return False
        return True

    def conflicts_count(S_list):
        """Return list of conflict counts per element."""
        Sset = set(S_list)
        val_to_idx = {v: i for i, v in enumerate(S_list)}
        n = len(S_list)
        conflicts = [0] * n
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                s = a + b
                if s in Sset:
                    conflicts[i] += 1
                    if j != i:
                        conflicts[j] += 1
                    k = val_to_idx.get(s)
                    if k is not None:
                        conflicts[k] += 1
        return conflicts

    def greedy_remove(V, time_limit, start_time):
        """Remove elements with most conflicts until sum-free."""
        S = set(V)
        while not is_sum_free(S):
            if time.time() - start_time > time_limit:
                break
            S_list = sorted(S)
            conflicts = conflicts_count(S_list)
            if not any(conflicts):
                break
            max_c = max(conflicts)
            idx = conflicts.index(max_c)
            S.remove(S_list[idx])
        return S

    def hill_climb(S, removed, time_limit, start_time, rng=None):
        """Try to add back removed elements."""
        if rng is None:
            rng = random.Random()
        S = set(S)
        S_list = list(S)
        removed = list(removed)
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            still_removed = []
            rng.shuffle(removed)
            for x in removed:
                if (time.time() - start_time) > time_limit:
                    still_removed.append(x)
                    continue
                if can_add(x, S, S_list):
                    S.add(x)
                    S_list.append(x)
                    improved = True
                else:
                    still_removed.append(x)
            removed = still_removed
        return S, removed

    def try_single_swap(S, removed, time_limit, start_time, rng=None):
        """Try to swap: remove one element from S, add one or more from removed."""
        if rng is None:
            rng = random.Random()
        S = set(S)
        S_list = sorted(S)
        removed = list(removed)
        n = len(S_list)
        if n == 0 or not removed:
            return S, removed, False
        
        improved = False
        indices = list(range(n))
        rng.shuffle(indices)
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            a = S_list[i]
            # Tentatively remove a
            S_tentative = set(S_list[:i] + S_list[i+1:])
            S_list_tentative = S_list[:i] + S_list[i+1:]
            
            # Try to add elements from removed
            added = []
            for x in removed:
                if time.time() - start_time > time_limit:
                    break
                if can_add(x, S_tentative, S_list_tentative):
                    S_tentative.add(x)
                    S_list_tentative.append(x)
                    added.append(x)
            
            # If we added more than we removed, accept the swap
            if len(added) > 1:
                S = S_tentative
                removed = [x for x in removed if x not in added]
                improved = True
                S_list = sorted(S)
                n = len(S_list)
        
        return S, removed, improved

    def try_double_swap(S, removed, time_limit, start_time, rng=None):
        """Try to swap: remove two elements from S, add multiple from removed."""
        if rng is None:
            rng = random.Random()
        S = set(S)
        S_list = sorted(S)
        removed = list(removed)
        n = len(S_list)
        if n < 2 or not removed:
            return S, removed, False
        
        improved = False
        indices = list(range(n))
        rng.shuffle(indices)
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            for j in indices:
                if j <= i or time.time() - start_time > time_limit:
                    continue
                a = S_list[i]
                b = S_list[j]
                # Tentatively remove a and b
                S_tentative = set(S_list[:i] + S_list[i+1:j] + S_list[j+1:])
                S_list_tentative = S_list[:i] + S_list[i+1:j] + S_list[j+1:]
                
                # Try to add elements from removed
                added = []
                for x in removed:
                    if time.time() - start_time > time_limit:
                        break
                    if can_add(x, S_tentative, S_list_tentative):
                        S_tentative.add(x)
                        S_list_tentative.append(x)
                        added.append(x)
                
                # If we added more than we removed, accept the swap
                if len(added) > 2:
                    S = S_tentative
                    removed = [x for x in removed if x not in added]
                    improved = True
                    S_list = sorted(S)
                    n = len(S_list)
                    break
        
        return S, removed, improved

    def ensure_sum_free(S, time_limit, start_time):
        """Ensure S is sum-free by removing conflicting elements."""
        S = set(S)
        if is_sum_free(S):
            return S
        
        S_list = sorted(S)
        while not is_sum_free(S_list):
            if time.time() - start_time > time_limit:
                break
            S_list = sorted(S_list)
            conflicts = conflicts_count(S_list)
            if not any(conflicts):
                break
            max_c = max(conflicts)
            idx = conflicts.index(max_c)
            S_list.pop(idx)
        
        return set(S_list)

    def greedy_add(V, order=None):
        """Greedily add elements in given order (or sorted order)."""
        S = set()
        S_list = []
        if order is None:
            order = V
        for x in order:
            if can_add(x, S, S_list):
                S.add(x)
                S_list.append(x)
        return S

    def multi_greedy(V, n_trials, time_limit, start_time, rng=None):
        """Try multiple greedy constructions with different orderings."""
        if rng is None:
            rng = random.Random()
        best_S = set()
        best_removed = list(V)
        
        for trial in range(n_trials):
            if time.time() - start_time > time_limit:
                break
            perm = list(V)
            rng.shuffle(perm)
            S_trial = greedy_add(V, perm)
            if len(S_trial) > len(best_S):
                best_S = set(S_trial)
                best_removed = [x for x in V if x not in S_trial]
        
        return best_S, best_removed

    def local_search(S, removed, time_limit, start_time, rng=None):
        """Combined local search: hill climb + swaps."""
        if rng is None:
            rng = random.Random()
        S = set(S)
        removed = list(removed)
        
        # Phase 1: Hill climb
        if (time.time() - start_time) < time_limit * 0.4:
            S, removed = hill_climb(S, removed, time_limit * 0.4, start_time, rng)
        
        # Phase 2: Single swaps
        if (time.time() - start_time) < time_limit * 0.7:
            S, removed, _ = try_single_swap(S, removed, time_limit * 0.3, start_time, rng)
        
        # Phase 3: Double swaps
        if (time.time() - start_time) < time_limit * 0.9:
            S, removed, _ = try_double_swap(S, removed, time_limit * 0.1, start_time, rng)
        
        # Phase 4: Final hill climb
        if (time.time() - start_time) < time_limit:
            S, removed = hill_climb(S, removed, time_limit * 0.1, start_time, rng)
        
        return S, removed

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining_total = DEADLINE - elapsed
        
        if remaining_total < 5:
            # Fallback: Greedy construction from scratch to ensure validity
            S_fallback = greedy_add(V)
            results.append(sorted(S_fallback))
            continue
        
        # Allocate time per instance based on remaining
        instances_left = 3 - len(results)
        time_per_instance = remaining_total / instances_left - 2
        
        best_size = 0
        best_set = set()
        
        # Strategy 1: Multi-greedy with local search
        t1 = time_per_instance * 0.35
        rng1 = random.Random(seed + 100)
        S1, removed1 = multi_greedy(V, max(3, int(t1 / 3.0)), t1 * 0.5, start, rng1)
        if (time.time() - start) < start + t1:
            S1, removed1 = local_search(S1, removed1, t1 * 0.5, time.time(), rng1)
        if is_sum_free(S1):
            if len(S1) > best_size:
                best_size = len(S1)
                best_set = set(S1)
        else:
            S1_fixed = ensure_sum_free(S1, 1.0, time.time())
            if len(S1_fixed) > best_size:
                best_size = len(S1_fixed)
                best_set = set(S1_fixed)
        
        # Strategy 2: Greedy remove + local search
        elapsed = time.time() - start
        if DEADLINE - elapsed > 5:
            t2 = min(time_per_instance * 0.25, DEADLINE - elapsed - 3)
            S2 = greedy_remove(V, t2 * 0.5, start)
            removed2 = [x for x in V if x not in S2]
            if (time.time() - start) < start + t2:
                S2, removed2 = local_search(S2, removed2, t2 * 0.5, time.time(), random.Random(seed + 200))
            if is_sum_free(S2):
                if len(S2) > best_size:
                    best_size = len(S2)
                    best_set = set(S2)
            else:
                S2_fixed = ensure_sum_free(S2, 1.0, time.time())
                if len(S2_fixed) > best_size:
                    best_size = len(S2_fixed)
                    best_set = set(S2_fixed)
        
        # Strategy 3: Multiple random restarts with local search
        elapsed = time.time() - start
        if DEADLINE - elapsed > 5:
            t3 = min(time_per_instance * 0.35, DEADLINE - elapsed - 3)
            rng_master = random.Random(seed + 300)
            n_trials = max(2, int(t3 / 5.0))
            for trial in range(n_trials):
                if DEADLINE - (time.time() - start) < 3:
                    break
                trial_time = t3 / (n_trials - trial)
                rng = random.Random(seed + trial * 1000 + 42)
                
                # Random permutation
                perm = list(V)
                rng.shuffle(perm)
                
                # Greedy add
                S_trial = greedy_add(V, perm)
                removed_trial = [x for x in V if x not in S_trial]
                
                # Local search
                S_trial, removed_trial = local_search(S_trial, removed_trial, min(trial_time * 0.8, 8.0), time.time(), rng)
                
                if is_sum_free(S_trial):
                    if len(S_trial) > best_size:
                        best_size = len(S_trial)
                        best_set = set(S_trial)
                else:
                    S_trial_fixed = ensure_sum_free(S_trial, 1.0, time.time())
                    if len(S_trial_fixed) > best_size:
                        best_size = len(S_trial_fixed)
                        best_set = set(S_trial_fixed)
        
        # Strategy 4: Final polish on best set
        elapsed = time.time() - start
        if DEADLINE - elapsed > 3 and best_set:
            t4 = min(5.0, DEADLINE - elapsed - 1)
            removed_final = [x for x in V if x not in best_set]
            if removed_final:
                rng_polish = random.Random(seed + 7777)
                best_set, removed_final = local_search(best_set, removed_final, t4, time.time(), rng_polish)
                if is_sum_free(best_set):
                    if len(best_set) > best_size:
                        best_size = len(best_set)
                else:
                    best_set_fixed = ensure_sum_free(best_set, 1.0, time.time())
                    if len(best_set_fixed) > best_size:
                        best_size = len(best_set_fixed)
                        best_set = best_set_fixed
        
        # Final safety check before appending
        if not is_sum_free(best_set):
            # Last resort: greedy add from scratch
            S_final = greedy_add(V)
            best_set = S_final
            
        results.append(sorted(best_set))
    
    return results
