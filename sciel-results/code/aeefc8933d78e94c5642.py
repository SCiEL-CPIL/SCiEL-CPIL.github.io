import random
import time
import math

def construct():
    # 1. Generate initial valid solution using greedy
    a = [0]
    
    def get_diffs_list(s):
        s = sorted(s)
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        
    def is_valid_full(s):
        if len(s) > 32:
            return False
        if 0 not in s:
            return False
        d = get_diffs_list(s)
        if len(d) != len(set(d)):
            return False
        d_set = set(d)
        for i in range(1, 31):
            if i not in d_set:
                return False
        return True

    # Greedy Init
    for d in range(1, 31):
        if d in get_diffs_list(sorted(a)):
            continue
        found = False
        candidates = set()
        for v in a:
            candidates.add(v + d)
            candidates.add(v - d)
        candidates = [c for c in candidates if c >= 0 and c not in a]
        random.shuffle(candidates)
        for x in candidates:
            trial = sorted(a + [x])
            if is_valid_full(trial):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if is_valid_full(trial):
                    a = trial
                    break
                y += 1
                
    # 2. Simulated Annealing
    best_a = sorted(a)
    best_max = max(best_a)
    
    current_a = best_a[:]
    current_diffs = set(get_diffs_list(current_a))
    current_max = best_max
    total_pairs = len(current_a) * (len(current_a) - 1) // 2
    
    start_time = time.time()
    timeout = 110.0
    
    T = 10.0 # Temperature
    T_min = 0.1
    cooling = 0.999
    
    def remove_elem(a_list, x, diffs_set):
        for v in a_list:
            if v == x: continue
            if v > x:
                diffs_set.discard(v - x)
            else:
                diffs_set.discard(x - v)
                
    def add_elem(a_list, y, diffs_set):
        for v in a_list:
            if v == y: continue
            if v > y:
                diffs_set.add(v - y)
            else:
                diffs_set.add(y - v)

    def check_coverage(diffs_set):
        for i in range(1, 31):
            if i not in diffs_set:
                return False
        return True

    while time.time() - start_time < timeout:
        if len(current_a) <= 1:
            break
        idx = random.randint(1, len(current_a) - 1)
        x = current_a[idx]
        
        if current_max <= 1:
            break
            
        # Sample y
        # To allow escape, we can sample y up to current_max + T
        # But we want to minimize max.
        # Let's sample y in [1, current_max + int(T)]
        upper = current_max + int(T)
        if upper < 1:
            upper = 1
        y = random.randint(1, upper)
        
        if y == x:
            continue
        if y in current_a:
            continue

        # Perform move
        remove_elem(current_a, x, current_diffs)
        current_a[idx] = y
        add_elem(current_a, y, current_diffs)
        
        if len(current_diffs) != total_pairs:
            remove_elem(current_a, y, current_diffs)
            current_a[idx] = x
            add_elem(current_a, x, current_diffs)
            continue
            
        if not check_coverage(current_diffs):
            remove_elem(current_a, y, current_diffs)
            current_a[idx] = x
            add_elem(current_a, x, current_diffs)
            continue
            
        new_max = max(current_a)
        delta = new_max - current_max
        
        if delta < 0:
            current_max = new_max
            if current_max < best_max:
                best_a = sorted(current_a)
                best_max = current_max
        else:
            # Accept with probability
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_max = new_max
                else:
                    remove_elem(current_a, y, current_diffs)
                    current_a[idx] = x
                    add_elem(current_a, x, current_diffs)
            else:
                remove_elem(current_a, y, current_diffs)
                current_a[idx] = x
                add_elem(current_a, x, current_diffs)
                
        T *= cooling
        if T < T_min:
            T = T_min

    return best_a
