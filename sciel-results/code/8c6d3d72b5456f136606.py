import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit, leaving 5s margin
    
    max_val = 4096
    
    best_set = []
    best_size = 0
    best_sum_set = set()
    
    def is_valid_addition(x, current_set, sum_set):
        new_sums = [3 * x]
        for y in current_set:
            new_sums.append(2 * x + y)
            new_sums.append(x + 2 * y)
        for y, z in combinations(current_set, 2):
            new_sums.append(x + y + z)
        
        new_sums_set = set(new_sums)
        if len(new_sums_set) != len(new_sums):
            return False, None
        if not new_sums_set.isdisjoint(sum_set):
            return False, None
        return True, new_sums_set

    def run_greedy(candidates):
        current_set = []
        sum_set = set()
        
        for x in candidates:
            valid, new_sums = is_valid_addition(x, current_set, sum_set)
            if valid:
                current_set.append(x)
                sum_set.update(new_sums)
                
        return current_set, sum_set

    # 1. Deterministic Increasing Greedy
    s, ss = run_greedy(range(1, max_val + 1))
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        best_sum_set = ss

    # 2. Randomized Greedy
    rng = random.Random(42)
    for i in range(50):
        if time.time() > deadline - 20.0:
            break
            
        candidates = list(range(1, max_val + 1))
        rng.shuffle(candidates)
        
        s, ss = run_greedy(candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_sum_set = ss

    # 3. Local Search with efficient operations
    if time.time() < deadline - 3.0:
        S = list(best_set)
        sum_set = set(best_sum_set)
        S_set = set(S)
        current_size = len(S)
        
        def compute_new_sums_for_add(x, S):
            """Compute all new triple sums when adding x to set S"""
            new_sums = [3 * x]
            for y in S:
                new_sums.append(2 * x + y)
                new_sums.append(x + 2 * y)
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    new_sums.append(x + S[i] + S[j])
            return new_sums

        def remove_sums_for_element(x, S):
            """Compute all triple sums involving x in set S (including x itself)"""
            to_remove = {3 * x}
            others = [y for y in S if y != x]
            for y in others:
                to_remove.add(2 * x + y)
                to_remove.add(x + 2 * y)
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    to_remove.add(x + others[i] + others[j])
            return to_remove

        # Simulated Annealing
        T = 50.0
        T_min = 0.5
        cooling_rate = 0.998
        
        while time.time() < deadline - 1.5:
            move_type = rng.random()
            
            if move_type < 0.25:
                # Try Add
                c = rng.randint(1, max_val)
                if c in S_set:
                    continue
                
                new_sums = compute_new_sums_for_add(c, S)
                new_sums_set = set(new_sums)
                
                if len(new_sums_set) != len(new_sums):
                    continue
                if not new_sums_set.isdisjoint(sum_set):
                    continue
                
                # Valid addition
                S.append(c)
                S_set.add(c)
                sum_set.update(new_sums_set)
                current_size += 1
                
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(S)
                    best_sum_set = set(sum_set)
                    
            elif move_type < 0.40:
                # Try Remove
                if not S:
                    continue
                a = rng.choice(S)
                
                to_remove = remove_sums_for_element(a, S)
                
                S.remove(a)
                S_set.remove(a)
                sum_set.difference_update(to_remove)
                current_size -= 1
                
                # Accept with probability exp(-1/T)
                if rng.random() >= (2.718281828 ** (-1.0 / T)):
                    # Revert
                    S.append(a)
                    S_set.add(a)
                    sum_set.update(to_remove)
                    current_size += 1
                    
            elif move_type < 0.75:
                # Try Swap
                if not S:
                    continue
                a = rng.choice(S)
                b = rng.randint(1, max_val)
                if b in S_set:
                    continue
                
                # Remove a
                to_remove_a = remove_sums_for_element(a, S)
                S.remove(a)
                S_set.remove(a)
                sum_set.difference_update(to_remove_a)
                
                # Try add b
                new_sums_b = compute_new_sums_for_add(b, S)
                new_sums_b_set = set(new_sums_b)
                
                if len(new_sums_b_set) == len(new_sums_b) and new_sums_b_set.isdisjoint(sum_set):
                    S.append(b)
                    S_set.add(b)
                    sum_set.update(new_sums_b_set)
                else:
                    # Revert
                    S.append(a)
                    S_set.add(a)
                    sum_set.update(to_remove_a)
                    
            else:
                # Try Compression: shift a random element to a lower value
                if not S:
                    continue
                a = rng.choice(S)
                if a <= 1:
                    continue
                
                # Try shifting down by small amount
                shift = rng.choice([1, 2, 3, 5, 8, 13])
                b = a - shift
                if b < 1 or b in S_set:
                    continue
                
                # Remove a
                to_remove_a = remove_sums_for_element(a, S)
                S.remove(a)
                S_set.remove(a)
                sum_set.difference_update(to_remove_a)
                
                # Try add b
                new_sums_b = compute_new_sums_for_add(b, S)
                new_sums_b_set = set(new_sums_b)
                
                if len(new_sums_b_set) == len(new_sums_b) and new_sums_b_set.isdisjoint(sum_set):
                    S.append(b)
                    S_set.add(b)
                    sum_set.update(new_sums_b_set)
                    # This is a neutral move (size unchanged) but compresses the set
                else:
                    # Revert
                    S.append(a)
                    S_set.add(a)
                    sum_set.update(to_remove_a)
            
            T *= cooling_rate
            if T < T_min:
                T = T_min

    return best_set
