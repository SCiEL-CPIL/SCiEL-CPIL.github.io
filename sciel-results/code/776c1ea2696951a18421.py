import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    prescribed = [1, 2, 5, 11]
    max_val = 284
    k_prescribed = len(prescribed)
    
    def get_conflict_count(a):
        sums = set()
        c = 0
        for i in range(len(a)):
            ai = a[i]
            for j in range(i, len(a)):
                s = ai + a[j]
                if s in sums:
                    c += 1
                else:
                    sums.add(s)
        return c

    def solve_for_k(k, time_limit):
        fixed = prescribed[:]
        available = [x for x in range(1, max_val + 1) if x not in fixed]
        random.shuffle(available)
        var_init = available[:k - k_prescribed]
        
        current_set = fixed + var_init
        current_values = set(current_set)
        current_conflicts = get_conflict_count(current_set)
        
        if current_conflicts == 0:
            return current_set

        T = 1.0
        cooling = 0.9995
        min_T = 1e-6
        best_set_found = None
        best_conflicts = current_conflicts
        
        var_indices = list(range(k_prescribed, k))
        
        while time.time() < time_limit:
            if T < min_T:
                T = 1.0
                
            idx = random.choice(var_indices)
            old_val = current_set[idx]
            
            new_val = random.randint(1, max_val)
            
            if new_val in current_values and new_val != old_val:
                continue
            
            # Update
            current_values.remove(old_val)
            current_values.add(new_val)
            current_set[idx] = new_val
            
            new_conflicts = get_conflict_count(current_set)
            delta = new_conflicts - current_conflicts
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_conflicts = new_conflicts
                if current_conflicts == 0:
                    return current_set
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_set_found = list(current_set)
            else:
                current_set[idx] = old_val
                current_values.remove(new_val)
                current_values.add(old_val)
            
            T *= cooling
        
        return best_set_found

    sizes_to_try = [20, 19, 18, 17, 16, 15]
    time_allocs = {20: 60, 19: 40, 18: 20, 17: 10, 16: 5, 15: 5}
    
    for k in sizes_to_try:
        if time.time() > deadline - 1.0:
            break
        t_limit = time.time() + time_allocs.get(k, 5)
        if t_limit > deadline:
            t_limit = deadline - 0.5
            
        result = solve_for_k(k, t_limit)
        if result:
            if len(result) == k and all(x in result for x in prescribed):
                sums = set()
                valid = True
                for i in range(len(result)):
                    for j in range(i, len(result)):
                        s = result[i] + result[j]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    return result

    return prescribed
