"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import random
    start_time = time.time()
    timeout = 115  # Safety margin for 120s limit

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add_element(x, s):
        """Check if x can be added to sum-free set s without creating conflicts."""
        if 2 * x in s:
            return False
        for y in s:
            if y != x:
                if x + y in s:
                    return False
                if x - y in s:
                    return False
        return True

    def get_conflicts(x, s):
        """Return list of elements in s that conflict with x."""
        conflicts = []
        if 2 * x in s:
            conflicts.append(2 * x)
        for y in s:
            if y != x:
                if x + y in s:
                    conflicts.append(x + y)
                if x - y in s:
                    conflicts.append(x - y)
        return list(set(conflicts))

    def greedy_sum_free(elements, order_key=None):
        """
        Strategy:
        1. Start with 'High Magnitude' candidates.
        2. Resolve conflicts in this initial set.
        3. Greedily add remaining elements using specified ordering.
        """
        if not elements:
            return []
        
        pos = [x for x in elements if x > 0]
        neg = [x for x in elements if x < 0]
        
        max_pos = max(pos) if pos else 0
        min_neg = min(neg) if neg else 0
        
        candidates = []
        for x in pos:
            if 2 * x > max_pos:
                candidates.append(x)
        for x in neg:
            if 2 * x < min_neg:
                candidates.append(x)
        
        selected_set = set(candidates)
        
        # Resolve conflicts in selected_set
        def has_conflict(s):
            for a in s:
                if 2*a in s:
                    return True
                for b in s:
                    if b != a and a+b in s:
                        return True
            return False

        while has_conflict(selected_set):
            removed = False
            for a in list(selected_set):
                if 2*a in selected_set:
                    selected_set.remove(2*a)
                    removed = True
                    break
                for b in selected_set:
                    if b != a:
                        s_val = a + b
                        if s_val in selected_set:
                            selected_set.remove(s_val)
                            removed = True
                            break
                if removed:
                    break
            if not removed:
                break

        remaining = [x for x in elements if x not in selected_set]
        if order_key:
            remaining.sort(key=order_key)
        else:
            remaining.sort(key=lambda x: -abs(x))
        
        for x in remaining:
            if can_add_element(x, selected_set):
                selected_set.add(x)
        
        return list(selected_set)

    def remove_and_retry_improve(elements, initial_subset, max_rounds=200):
        """
        Remove-and-retry: for each element in the subset, try removing it and
        then greedily adding back as many elements as possible. If the result
        is larger, accept it.
        """
        s = set(initial_subset)
        remaining = [x for x in elements if x not in s]
        
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            
            best_gain = 0
            best_new_set = None
            best_removed = None
            
            for x in list(s):
                s_no_x = set(s)
                s_no_x.discard(x)
                
                # Greedily add from remaining + {x}
                candidates = [y for y in remaining if y not in s_no_x]
                candidates.append(x)
                candidates.sort(key=lambda z: -abs(z))
                
                s_candidate = set(s_no_x)
                for y in candidates:
                    if can_add_element(y, s_candidate):
                        s_candidate.add(y)
                
                gain = len(s_candidate) - len(s)
                if gain > best_gain:
                    best_gain = gain
                    best_new_set = s_candidate
                    best_removed = x
            
            if best_gain > 0 and best_new_set is not None:
                s = best_new_set
                remaining = [x for x in elements if x not in s]
                improved = True
        
        return list(s)

    def simulated_annealing_improve(elements, initial_subset, time_budget=10):
        """
        Simulated annealing over the binary inclusion vector.
        Move: add a random element, remove a random element, or swap.
        Objective: maximize |S| subject to S being sum-free.
        """
        rng = random.Random(42)
        s = set(initial_subset)
        remaining = [x for x in elements if x not in s]
        
        best_s = set(s)
        best_size = len(s)
        
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        end_time = time.time() + time_budget
        steps = 0
        
        while time.time() < end_time:
            steps += 1
            
            # Random move
            move_type = rng.random()
            new_s = set(s)
            
            if move_type < 0.3 and remaining:
                # Add a random element from remaining
                idx = rng.randrange(len(remaining))
                x = remaining[idx]
                if can_add_element(x, new_s):
                    new_s.add(x)
                else:
                    # Try removing a conflicting element first
                    conflicts = get_conflicts(x, new_s)
                    if conflicts:
                        for c in conflicts:
                            if rng.random() < 0.5:
                                new_s.discard(c)
                        if can_add_element(x, new_s):
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
                    # Try to add a random remaining element
                    for _ in range(min(3, len(remaining))):
                        idx = rng.randrange(len(remaining))
                        y = remaining[idx]
                        if can_add_element(y, new_s):
                            new_s.add(y)
                            break
            
            old_size = len(s)
            new_size = len(new_s)
            delta = new_size - old_size
            
            if delta > 0 or (delta == 0 and rng.random() < 0.3):
                # Accept
                s = new_s
                remaining = [x for x in elements if x not in s]
                if new_size > best_size:
                    best_size = new_size
                    best_s = set(s)
            elif delta < 0:
                # Maybe accept with probability exp(delta / T)
                if rng.random() < math.exp(delta / T):
                    s = new_s
                    remaining = [x for x in elements if x not in s]
            
            T = max(T_min, T * cooling)
        
        return list(best_s)

    def branch_and_bound_max_independent_set(elements, time_budget=5):
        """
        Branch and bound for maximum independent set on the conflict graph.
        The conflict graph has an edge between a and b if a+b is in elements,
        or 2a is in elements (self-loop handled separately).
        
        Actually, the sum-free constraint is: no a, b, c in S with a+b=c.
        This is a 3-uniform hypergraph constraint, not a simple graph.
        
        Let's reformulate: for each triple (a,b,c) with a+b=c, we can't have
        all three in S. This is a 3-SAT-like constraint.
        
        For branch and bound, we'll use a simpler approach:
        - Build the set of "forbidden pairs" given a partial solution.
        - Use greedy coloring for upper bounds.
        """
        # This is complex for 3-uniform constraints. Let's use a simpler B&B:
        # Branch on each element: include or exclude.
        # Upper bound: current_size + number of remaining elements that don't
        # conflict with current set.
        
        rng = random.Random(123)
        elems = list(elements)
        n = len(elems)
        
        # Sort by degree (number of conflicts) descending for better pruning
        def conflict_count(x):
            count = 0
            for y in elems:
                if y != x and (x + y in set(elems) or x - y in set(elems) or 2*x in set(elems)):
                    count += 1
            return count
        
        elem_set = set(elems)
        elems_sorted = sorted(elems, key=lambda x: -conflict_count(x))
        
        best_size = 0
        best_set = set()
        
        def bb(idx, current_set, current_size):
            nonlocal best_size, best_set
            
            if time.time() > start_time + time_budget:
                return
            
            # Upper bound
            ub = current_size + (n - idx)
            if ub <= best_size:
                return
            
            if idx == n:
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                return
            
            x = elems_sorted[idx]
            
            # Option 1: include x if possible
            if can_add_element(x, current_set):
                bb(idx + 1, current_set | {x}, current_size + 1)
            
            # Option 2: exclude x
            bb(idx + 1, current_set, current_size)
        
        # This is exponential. Let's use a heuristic B&B with limited depth.
        # Instead, let's do a smarter local search with more diverse restarts.
        
        return list(best_set)

    def multi_start_local_search(elements, time_budget=20):
        """
        Run multiple greedy starts with different orderings, then improve
        each with remove-and-retry, and pick the best.
        """
        orderings = [
            lambda x: -abs(x),
            lambda x: x,
            lambda x: -x,
            lambda x: abs(x),
            lambda x: (0 if x > 0 else 1, -abs(x)),
            lambda x: (1 if x > 0 else 0, abs(x)),
            lambda x: (0 if x > 0 else 1, abs(x)),
            lambda x: (1 if x > 0 else 0, -abs(x)),
        ]
        
        best_subset = None
        best_size = -1
        
        per_order_budget = time_budget / (len(orderings) + 2)
        
        for order in orderings:
            if time.time() - start_time > timeout:
                break
            t0 = time.time()
            subset = greedy_sum_free(elements, order_key=order)
            
            # Improve with remove-and-retry
            if time.time() - t0 < per_order_budget * 0.5:
                improved = remove_and_retry_improve(elements, subset, max_rounds=50)
                if len(improved) > len(subset):
                    subset = improved
            
            if len(subset) > best_size:
                best_size = len(subset)
                best_subset = subset
        
        # Final polish with simulated annealing on the best
        if best_subset is not None:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 5:
                sa_budget = min(remaining_time - 2, 15)
                if sa_budget > 2:
                    polished = simulated_annealing_improve(elements, best_subset, time_budget=sa_budget)
                    if len(polished) > best_size:
                        best_subset = polished
                        best_size = len(polished)
        
        return best_subset if best_subset is not None else []

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > timeout:
            results.append([])
            continue
            
        elems = instance(size, seed)
        
        # Allocate time proportionally to problem size
        # n=60: ~15s, n=120: ~25s, n=240: ~40s
        if size <= 60:
            time_budget = 15
        elif size <= 120:
            time_budget = 25
        else:
            time_budget = 40
        
        result = multi_start_local_search(elems, time_budget=time_budget)
        results.append(result if result is not None else [])

    return results
# EVOLVE-BLOCK-END
