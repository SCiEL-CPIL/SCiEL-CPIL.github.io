"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import random
    start_time = time.time()
    DEADLINE = 145.0  # wall-clock limit; keep margin
    MARGIN = 3.0      # safety margin seconds

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c for a,b,c in S."""
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def can_add(x, chosen_set):
        """Check if x can be added to chosen_set while maintaining sum-free."""
        for a in chosen_set:
            if (x - a) in chosen_set:
                return False
            if a + x in chosen_set:
                return False
        if 2 * x in chosen_set:
            return False
        return True

    def compute_conflict_matrix(S):
        """
        Precompute conflicts: conflict[i][j] = True if S[i] and S[j] cannot
        both be in a sum-free set (i.e., there exists k in S such that
        S[i]+S[j]=S[k], or 2*S[i]=S[j], or 2*S[j]=S[i]).
        Also compute conflict_degree[i] = number of elements conflicting with S[i].
        """
        n = len(S)
        Sset = set(S)
        # conflict[i] is a set of indices j such that S[i] and S[j] conflict
        conflict = [set() for _ in range(n)]
        for i in range(n):
            xi = S[i]
            for j in range(i + 1, n):
                xj = S[j]
                # Check if xi and xj conflict: xi+xj in S, 2*xi==xj, or 2*xj==xi
                if xi + xj in Sset or 2 * xi == xj or 2 * xj == xi:
                    conflict[i].add(j)
                    conflict[j].add(i)
        conflict_degree = [len(c) for c in conflict]
        return conflict, conflict_degree

    def greedy_build(S, order, conflict=None, conflict_degree=None):
        """Greedy: add elements in given order if sum-free is maintained."""
        chosen = []
        chosen_set = set()
        chosen_idx_set = set()
        for idx in order:
            x = S[idx]
            if idx in chosen_idx_set:
                continue
            if can_add(x, chosen_set):
                chosen.append(x)
                chosen_set.add(x)
                chosen_idx_set.add(idx)
        return chosen

    def local_search_remove_retry(S, initial, conflict, conflict_degree, max_rounds=50):
        """
        Local search: remove one element, then greedily add back as many as possible.
        Candidates are sorted by conflict degree (fewer conflicts first).
        """
        current = list(initial)
        current_set = set(current)
        best = len(current)
        n = len(S)
        
        # Map value to index
        val_to_idx = {v: i for i, v in enumerate(S)}
        
        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            
            improved = False
            # Try removing each element and see if we can add more back
            for i in range(len(current)):
                if time.time() - start_time > DEADLINE - MARGIN:
                    break
                x = current[i]
                # Remove x
                new_set = set(current)
                new_set.discard(x)
                
                # Now try to add back elements not in new_set, greedily
                # Sort candidates by how few conflicts they have with new_set
                candidates = []
                for y in S:
                    if y not in new_set:
                        # Count conflicts with new_set
                        cnt = 0
                        for a in new_set:
                            if (y - a) in new_set or a + y in new_set:
                                cnt += 1
                        if 2 * y in new_set:
                            cnt += 1
                        candidates.append((cnt, y))
                
                candidates.sort(key=lambda t: t[0])
                
                trial_set = set(new_set)
                for cnt, y in candidates:
                    if can_add(y, trial_set):
                        trial_set.add(y)
                
                if len(trial_set) > best:
                    best = len(trial_set)
                    current = list(trial_set)
                    current_set = set(current)
                    improved = True
                    break
            
            if not improved:
                break
        
        return current

    def local_search_swap_add(S, initial, conflict, conflict_degree, max_rounds=30):
        """
        Local search: try to add elements that can be added without removing anything.
        Then try swaps: remove one, add two.
        """
        current = list(initial)
        current_set = set(current)
        best = len(current)
        n = len(S)
        
        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            
            improved = False
            
            # Phase 1: Try to add any element that fits
            for y in S:
                if y not in current_set:
                    if can_add(y, current_set):
                        current_set.add(y)
                        current = list(current_set)
                        best = len(current)
                        improved = True
            
            # Phase 2: Try swaps - remove one, add up to 2
            if not improved:
                for i in range(len(current)):
                    if time.time() - start_time > DEADLINE - MARGIN:
                        break
                    x = current[i]
                    new_set = set(current)
                    new_set.discard(x)
                    
                    added = 0
                    for y in S:
                        if y not in new_set and added < 2:
                            if can_add(y, new_set):
                                new_set.add(y)
                                added += 1
                    
                    if added > 1:  # Net gain of at least 1
                        current = list(new_set)
                        current_set = set(current)
                        best = len(current)
                        improved = True
                        break
            
            if not improved:
                break
        
        return current

    def simulated_annealing_improve(S, initial, time_budget=10):
        """
        Simulated annealing over the binary inclusion vector.
        Move: add a random element, remove a random element, or swap.
        Objective: maximize |S| subject to S being sum-free.
        """
        rng = random.Random(42)
        s = set(initial)
        remaining = [x for x in S if x not in s]
        
        best_s = set(s)
        best_size = len(s)
        
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time and time.time() - start_time < DEADLINE - MARGIN:
            move_type = rng.random()
            new_s = set(s)
            
            if move_type < 0.3 and remaining:
                # Add a random element from remaining
                idx = rng.randrange(len(remaining))
                x = remaining[idx]
                if can_add(x, new_s):
                    new_s.add(x)
                else:
                    # Try removing a conflicting element first
                    conflicts = []
                    if 2 * x in new_s:
                        conflicts.append(2 * x)
                    for y in new_s:
                        if y != x:
                            if x + y in new_s:
                                conflicts.append(x + y)
                            if x - y in new_s:
                                conflicts.append(x - y)
                    if conflicts:
                        for c in conflicts:
                            if rng.random() < 0.5:
                                new_s.discard(c)
                        if can_add(x, new_s):
                            new_s.add(x)
            elif move_type < 0.5 and s:
                # Remove a random element
                x = list(s)[rng.randrange(len(s))]
                new_s.discard(x)
            else:
                # Swap: remove one, try to add another
                if s and remaining:
                    x = list(s)[rng.randrange(len(s))]
                    new_s.discard(x)
                    for _ in range(min(3, len(remaining))):
                        idx = rng.randrange(len(remaining))
                        y = remaining[idx]
                        if can_add(y, new_s):
                            new_s.add(y)
                            break
            
            old_size = len(s)
            new_size = len(new_s)
            delta = new_size - old_size
            
            if delta > 0 or (delta == 0 and rng.random() < 0.3):
                s = new_s
                remaining = [x for x in S if x not in s]
                if new_size > best_size:
                    best_size = new_size
                    best_s = set(s)
            elif delta < 0:
                if rng.random() < math.exp(delta / T):
                    s = new_s
                    remaining = [x for x in S if x not in s]
            
            T = max(T_min, T * cooling)
        
        return list(best_s)

    def solve_instance(S, time_budget):
        """Solve one instance: multiple greedy starts + local search."""
        n = len(S)
        if n == 0:
            return []
        
        # Baseline: odd numbers (always sum-free)
        odds = [x for x in S if x % 2 != 0]
        best_result = odds
        best_size = len(odds)
        
        # Compute conflicts
        conflict, conflict_degree = compute_conflict_matrix(S)
        
        # Try multiple greedy orderings
        orderings = []
        
        # Natural order
        orderings.append(list(range(n)))
        # Reverse order
        orderings.append(list(range(n - 1, -1, -1)))
        
        # Sorted by conflict count (least conflicted first)
        orderings.append(sorted(range(n), key=lambda i: conflict_degree[i]))
        # Sorted by conflict count (most conflicted first)
        orderings.append(sorted(range(n), key=lambda i: -conflict_degree[i]))
        
        # Sorted by absolute value
        orderings.append(sorted(range(n), key=lambda i: abs(S[i])))
        # Sorted by -absolute value
        orderings.append(sorted(range(n), key=lambda i: -abs(S[i])))
        # Sorted by value
        orderings.append(sorted(range(n), key=lambda i: S[i]))
        # Sorted by -value
        orderings.append(sorted(range(n), key=lambda i: -S[i]))
        
        # Random shuffles
        rng = random.Random(42)
        num_random = min(20, max(5, n // 5))
        for _ in range(num_random):
            perm = list(range(n))
            rng.shuffle(perm)
            orderings.append(perm)
        
        # Run greedy on all orderings
        for order in orderings:
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            result = greedy_build(S, order)
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
        
        # Local search on best
        if best_result:
            # Remove-and-retry
            if time.time() - start_time < DEADLINE - MARGIN - 5:
                improved = local_search_remove_retry(S, best_result, conflict, conflict_degree, max_rounds=30)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_result = improved
            
            # Swap-add
            if time.time() - start_time < DEADLINE - MARGIN - 3:
                improved = local_search_swap_add(S, best_result, conflict, conflict_degree, max_rounds=20)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_result = improved
        
        # Final polish with simulated annealing
        if best_result and time.time() - start_time < DEADLINE - MARGIN - 2:
            remaining_time = min(time_budget - (time.time() - start_time), 10)
            if remaining_time > 2:
                polished = simulated_annealing_improve(S, best_result, time_budget=remaining_time)
                if len(polished) > best_size:
                    best_size = len(polished)
                    best_result = polished
        
        return best_result if best_result else []

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE - MARGIN:
            results.append([])
            continue
            
        elems = instance(size, seed)
        
        # Allocate time proportionally to problem size
        if size <= 60:
            time_budget = 15
        elif size <= 120:
            time_budget = 25
        else:
            time_budget = 40
        
        result = solve_instance(elems, time_budget=time_budget)
        results.append(result if result is not None else [])

    return results
# EVOLVE-BLOCK-END
