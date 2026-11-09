"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if len(S) <= 1:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        """Check if x can be added to S_set while maintaining sum-freeness."""
        # Need: no a,b in S_set with a+b=x, no a in S_set with a+x in S_set, and 2x not in S_set
        for a in S_set:
            if x - a in S_set and a <= x - a:
                return False
            if a + x in S_set:
                return False
        if 2 * x in S_set:
            return False
        return True

    def greedy_by_order(S, order_key, start_time=None):
        """Greedy: add elements in given order if sum-free with current set."""
        result = []
        result_set = set()
        for x in sorted(S, key=order_key):
            if can_add(x, result_set):
                result.append(x)
                result_set.add(x)
        return result

    def local_search(S, initial, start_time=None):
        """Improve sum-free subset by trying additions and 1-for-2 swaps."""
        if start_time is None:
            start_time = time.time()
        current = list(initial)
        current_set = set(current)
        S_list = list(S)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Try adding each element not in current
            for x in S_list:
                if x in current_set:
                    continue
                if can_add(x, current_set):
                    current.append(x)
                    current_set.add(x)
                    improved = True
            # Try removing one element and adding up to two
            if time.time() > deadline:
                break
            for a in list(current_set):
                if time.time() > deadline:
                    break
                temp_set = set(current_set)
                temp_set.discard(a)
                added = 0
                for x in S_list:
                    if x in temp_set:
                        continue
                    if can_add(x, temp_set):
                        temp_set.add(x)
                        added += 1
                        if added >= 2:
                            break
                if added > 1:
                    current = list(temp_set)
                    current_set = temp_set
                    improved = True
        return current

    def simulated_annealing(S, initial, start_time=None):
        """Simulated annealing on conflict graph: accept worse states with decaying probability."""
        if start_time is None:
            start_time = time.time()
        current = list(initial)
        current_set = set(current)
        best = list(current)
        best_set = set(current)
        S_list = list(S)
        n = len(S_list)
        
        temp = 5.0
        cooling = 0.995
        min_temp = 0.1
        
        while temp > min_temp:
            if time.time() > deadline:
                break
            
            # Pick a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random element
                candidates = [x for x in S_list if x not in current_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if can_add(x, current_set):
                    current.append(x)
                    current_set.add(x)
                    if len(current) > len(best):
                        best = list(current)
                        best_set = set(current)
                else:
                    # Accept worse state with probability exp(-delta/temp)
                    # delta = 0 (no size change) but we break sum-freeness, so reject
                    pass
            elif move_type < 0.7:
                # Try to remove a random element
                if not current:
                    continue
                x = random.choice(current)
                current.remove(x)
                current_set.discard(x)
                if len(current) > len(best):
                    best = list(current)
                    best_set = set(current)
            else:
                # Swap: remove one, try to add another
                if not current:
                    continue
                x_remove = random.choice(current)
                current_set.discard(x_remove)
                candidates = [x for x in S_list if x not in current_set]
                if candidates:
                    x_add = random.choice(candidates)
                    if can_add(x_add, current_set):
                        current.append(x_add)
                        current_set.add(x_add)
                        if len(current) > len(best):
                            best = list(current)
                            best_set = set(current)
                    else:
                        # Reject swap, restore
                        current_set.add(x_remove)
                        if x_remove not in current:
                            current.append(x_remove)
                else:
                    current_set.add(x_remove)
                    if x_remove not in current:
                        current.append(x_remove)
            
            temp *= cooling
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Multiple starting points
        candidates = []
        
        # Odd numbers (guaranteed sum-free)
        odd_subset = [x for x in S if x % 2]
        candidates.append(odd_subset)
        
        # Greedy by absolute value
        greedy_abs = greedy_by_order(S, lambda x: (abs(x), x))
        candidates.append(greedy_abs)
        
        # Greedy by value (positive first)
        greedy_val = greedy_by_order(S, lambda x: (-x if x > 0 else x))
        candidates.append(greedy_val)
        
        # Greedy by -abs (large magnitude first)
        greedy_negabs = greedy_by_order(S, lambda x: (-abs(x), x))
        candidates.append(greedy_negabs)
        
        # Pick best candidate
        best = max(candidates, key=len)
        
        # Local search
        if time.time() < deadline:
            improved = local_search(S, best)
            if len(improved) > len(best):
                best = improved
        
        # Simulated annealing with time budget
        if time.time() < deadline - 10:
            sa_result = simulated_annealing(S, best)
            if len(sa_result) > len(best):
                best = sa_result
        
        # Verify sum-freeness
        if not is_sum_free(best):
            best = odd_subset
        
        results.append(best)
    
    return results
# EVOLVE-BLOCK-END
