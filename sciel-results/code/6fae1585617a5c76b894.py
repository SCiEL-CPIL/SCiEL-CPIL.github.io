import time
import random
import math

_START = None

def _calc_cost(marks, D_target=30):
    """
    Calculate cost for a set of marks.
    Returns (dup_penalty, miss_penalty, max_mark)
    """
    if not marks:
        return 0, 0, 0
    marks = sorted(set(marks))
    if 0 not in marks:
        # Invalid state, high penalty
        return 10000, 10000, max(marks)
    
    diffs = []
    n = len(marks)
    for i in range(n):
        for j in range(i+1, n):
            d = marks[j] - marks[i]
            if d <= 0:
                continue
            diffs.append(d)
            
    # Duplicate differences
    dup_count = 0
    diff_set = set()
    for d in diffs:
        if d in diff_set:
            dup_count += 1
        else:
            diff_set.add(d)
            
    # Missing differences 1..D_target
    miss_count = 0
    for d in range(1, D_target + 1):
        if d not in diff_set:
            miss_count += 1
            
    return dup_count, miss_count, max(marks)

def _sa_search(D_limit, time_limit):
    """
    Simulated Annealing search for a Sidon set with max element <= D_limit
    that covers 1..30.
    """
    # Initial state: random set containing 0
    # Size heuristic: Need enough elements to cover 30 diffs.
    # Min size approx sqrt(30) ~ 6-8. Let's start with 10.
    current_size = 10
    marks = [0] + random.sample(range(1, D_limit + 1), current_size - 1)
    marks = sorted(set(marks))
    
    cur_dup, cur_miss, cur_max = _calc_cost(marks)
    cur_cost = cur_dup * 1000 + cur_miss * 100 + cur_max
    
    best_dup, best_miss, best_max = cur_dup, cur_miss, cur_max
    best_marks = list(marks)
    
    T = 100.0
    cooling_rate = 0.999
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if T < 0.1:
            break
            
        # Generate neighbor
        neighbor = list(marks)
        move_type = random.randint(0, 3)
        
        if move_type == 0 and len(neighbor) < 32:
            # Add random element
            x = random.randint(1, D_limit)
            if x not in neighbor:
                neighbor.append(x)
        elif move_type == 1 and len(neighbor) > 2:
            # Remove random element (not 0)
            non_zero = [i for i, v in enumerate(neighbor) if v != 0]
            if non_zero:
                idx = random.choice(non_zero)
                neighbor.pop(idx)
        elif move_type == 2 and len(neighbor) > 0:
            # Replace random element (not 0)
            non_zero = [i for i, v in enumerate(neighbor) if v != 0]
            if non_zero:
                idx = random.choice(non_zero)
                new_val = random.randint(1, D_limit)
                if new_val not in neighbor:
                    neighbor[idx] = new_val
        elif move_type == 3 and len(neighbor) > 0:
            # Perturb random element (not 0)
            non_zero = [i for i, v in enumerate(neighbor) if v != 0]
            if non_zero:
                idx = random.choice(non_zero)
                delta = random.randint(-5, 5)
                new_val = neighbor[idx] + delta
                if 1 <= new_val <= D_limit and new_val not in neighbor:
                    neighbor[idx] = new_val
        else:
            # No change
            pass
            
        neighbor = sorted(set(neighbor))
        if 0 not in neighbor:
            neighbor.append(0)
            neighbor = sorted(neighbor)
            
        n_dup, n_miss, n_max = _calc_cost(neighbor)
        n_cost = n_dup * 1000 + n_miss * 100 + n_max
        
        delta_cost = n_cost - cur_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            marks = neighbor
            cur_dup, cur_miss, cur_max = n_dup, n_miss, n_max
            cur_cost = n_cost
            
            if n_dup == 0 and n_miss == 0:
                if n_max < best_max:
                    best_dup, best_miss, best_max = n_dup, n_miss, n_max
                    best_marks = list(marks)
                    # If we found a valid set, we can try to reduce max further
                    # by restarting with tighter limit or just continue cooling
                    # But for now, record it.
            
        T *= cooling_rate
        
    if best_dup == 0 and best_miss == 0:
        return best_marks
    return None

def _seed_greedy():
    # Keep the original greedy as fallback
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        placed = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                placed = True
                break
        if not placed:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # Try Simulated Annealing for increasing D limits
    # We want to minimize the max element.
    # Start from a reasonable lower bound.
    # Min diameter for 30 diffs is approx sqrt(30*4) ~ 11? No, N(N-1)/2 >= 30 -> N~8.
    # Diameter D must be at least 30.
    
    best_result = None
    best_val = 10**6
    
    # Try SA with D limits
    for D in range(30, 100):
        if time.time() - _START > 110:
            break
        # Run SA for a short time for each D
        res = _sa_search(D, time_limit=5)
        if res is not None:
            max_val = max(res)
            if max_val < best_val:
                best_val = max_val
                best_result = res
    
    if best_result is not None:
        return best_result
    
    # Fallback to greedy
    return _seed_greedy()
