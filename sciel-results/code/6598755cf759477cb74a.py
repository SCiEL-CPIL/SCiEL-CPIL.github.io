import time
import random
import numpy as np

def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin (limit is 150s)

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
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i, len(S_list)):
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

    def greedy_remove(V, time_limit, start_time):
        """Remove elements with most conflicts until sum-free."""
        S = set(V)
        S_list = sorted(S)
        
        while not is_sum_free(S):
            if time.time() - start_time > time_limit:
                break
            S_list = sorted(S)
            n = len(S_list)
            Sset = set(S_list)
            
            # Count conflicts for each element
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
                        # Find index of s
                        for k in range(n):
                            if S_list[k] == s:
                                conflicts[k] += 1
                                break
            
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

    def try_swap(S, S_list, removed, time_limit, start_time, rng=None):
        """Try to swap: remove one element from S, add one or more from removed."""
        if rng is None:
            rng = random.Random()
        S = set(S)
        S_list = list(S)
        removed = list(removed)
        n = len(S_list)
        if n == 0 or not removed:
            return S, S_list, removed, False
        
        improved = False
        # Try removing each element and see if we can add more
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
            if len(added) > 1:  # removed 1, added >1
                S = S_tentative
                S_list = S_list_tentative
                removed = [x for x in removed if x not in added]
                improved = True
        
        return S, S_list, removed, improved

    def modular_seeded(V, time_limit, start_time):
        """
        DIRECT construction: use modular sum-free intervals.
        """
        n = len(V)
        min_v = min(V)
        max_v = max(V)
        
        best_S = set()
        best_removed = list(V)
        
        # Try multiple modular constructions with different moduli
        range_size = max_v - min_v + 1
        candidates_mods = list(set([n, 2*n, 3*n, 4*n, 5*n, range_size, range_size // 2, range_size // 3, range_size // 4, range_size // 5]))
        
        for M in candidates_mods:
            if M <= 0:
                continue
            lower = M // 3 + 1
            upper = (2 * M) // 3
            S = set()
            for x in V:
                r = (x - min_v) % M
                if lower <= r <= upper:
                    S.add(x)
            if len(S) > len(best_S):
                best_S = set(S)
                best_removed = [x for x in V if x not in S]
        
        # Construction: sign-based
        pos = [x for x in V if x > 0]
        neg = [x for x in V if x < 0]
        if len(pos) > len(best_S):
            best_S = set(pos)
            best_removed = neg
        if len(neg) > len(best_S):
            best_S = set(neg)
            best_removed = pos
        
        # Construction: parity (odd numbers are sum-free)
        odds = [x for x in V if x % 2 != 0]
        if len(odds) > len(best_S):
            best_S = set(odds)
            best_removed = [x for x in V if x % 2 == 0]
        
        # Now optimize from the best seed
        if not is_sum_free(best_S):
            best_S = greedy_remove(list(best_S), time_limit * 0.3, start_time)
            best_removed = [x for x in V if x not in best_S]
        
        # Hill climb to add more
        if (time.time() - start_time) < time_limit:
            best_S, best_removed = hill_climb(best_S, best_removed, time_limit * 0.7, start_time)
        
        return best_S

    def ensure_sum_free(S, time_limit, start_time):
        """Ensure S is sum-free by removing conflicting elements."""
        S = set(S)
        if is_sum_free(S):
            return S
        
        S_list = sorted(S)
        while not is_sum_free(S):
            if time.time() - start_time > time_limit:
                break
            S_list = sorted(S)
            n = len(S_list)
            Sset = set(S_list)
            
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
                        for k in range(n):
                            if S_list[k] == s:
                                conflicts[k] += 1
                                break
            
            if not any(conflicts):
                break
            max_c = max(conflicts)
            idx = conflicts.index(max_c)
            S.remove(S_list[idx])
        
        return S

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
        
        # Strategy 1: modular seeded
        t1 = time_per_instance * 0.4
        S1 = modular_seeded(V, t1, start)
        if is_sum_free(S1):
            if len(S1) > best_size:
                best_size = len(S1)
                best_set = set(S1)
        else:
            S1_fixed = ensure_sum_free(S1, 1.0, time.time())
            if len(S1_fixed) > best_size:
                best_size = len(S1_fixed)
                best_set = set(S1_fixed)
        
        # Strategy 2: greedy remove
        elapsed = time.time() - start
        if DEADLINE - elapsed > 5:
            t2 = min(time_per_instance * 0.3, DEADLINE - elapsed - 3)
            S2 = greedy_remove(V, t2, start)
            removed2 = [x for x in V if x not in S2]
            S2, removed2 = hill_climb(S2, removed2, min(2.0, DEADLINE - (time.time() - start) - 1), start)
            if is_sum_free(S2):
                if len(S2) > best_size:
                    best_size = len(S2)
                    best_set = set(S2)
            else:
                S2_fixed = ensure_sum_free(S2, 1.0, time.time())
                if len(S2_fixed) > best_size:
                    best_size = len(S2_fixed)
                    best_set = set(S2_fixed)
        
        # Strategy 3: Multiple random restarts with greedy add + hill climb
        elapsed = time.time() - start
        if DEADLINE - elapsed > 5:
            t3 = min(time_per_instance * 0.4, DEADLINE - elapsed - 3)
            rng_master = random.Random(seed + 999)
            n_trials = max(2, int(t3 / 2.0))
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
                
                # Hill climb
                S_trial, removed_trial = hill_climb(S_trial, removed_trial, min(trial_time * 0.6, 5.0), start, rng)
                
                # Swap-based improvement
                if removed_trial and (time.time() - start) < DEADLINE - 2:
                    S_trial, S_list_trial, removed_trial, _ = try_swap(
                        S_trial, list(S_trial), removed_trial,
                        min(trial_time * 0.3, 3.0), start, rng
                    )
                
                if is_sum_free(S_trial):
                    if len(S_trial) > best_size:
                        best_size = len(S_trial)
                        best_set = set(S_trial)
                else:
                    S_trial_fixed = ensure_sum_free(S_trial, 1.0, time.time())
                    if len(S_trial_fixed) > best_size:
                        best_size = len(S_trial_fixed)
                        best_set = set(S_trial_fixed)
        
        # Strategy 4: Final polish - try individual swaps on the best set
        elapsed = time.time() - start
        if DEADLINE - elapsed > 3 and best_set:
            t4 = min(3.0, DEADLINE - elapsed - 1)
            removed_final = [x for x in V if x not in best_set]
            if removed_final:
                rng_polish = random.Random(seed + 7777)
                best_set, S_list_best, removed_final, _ = try_swap(
                    best_set, list(best_set), removed_final, t4, start, rng_polish
                )
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
