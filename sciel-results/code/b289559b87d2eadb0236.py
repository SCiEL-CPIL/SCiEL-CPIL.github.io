import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    chosen = []
    chosen_set = set()
    sum_counts = {}
    
    # Greedy Init
    for x in range(1, 1001):
        valid = True
        for y in chosen:
            s = x + y
            if sum_counts.get(s, 0) >= 2:
                valid = False
                break
        if valid:
            s = 2 * x
            if sum_counts.get(s, 0) >= 2:
                valid = False
        
        if valid:
            chosen.append(x)
            chosen_set.add(x)
            for y in chosen:
                if y != x: # y is always != x here as x is new
                    s = x + y
                    sum_counts[s] = sum_counts.get(s, 0) + 1
            s = 2 * x
            sum_counts[s] = sum_counts.get(s, 0) + 1

    best_size = len(chosen)
    best_chosen = chosen.copy()
    best_sum_counts = sum_counts.copy()
    best_chosen_set = chosen_set.copy()
    
    while time.time() < deadline:
        if random.random() < 0.7:
            # Add
            x = random.randint(1, 1000)
            if x in chosen_set:
                continue
            
            valid = True
            for y in chosen:
                s = x + y
                if sum_counts.get(s, 0) >= 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if sum_counts.get(s, 0) >= 2:
                    valid = False
            
            if valid:
                chosen.append(x)
                chosen_set.add(x)
                for y in chosen:
                    if y != x:
                        s = x + y
                        sum_counts[s] = sum_counts.get(s, 0) + 1
                s = 2 * x
                sum_counts[s] = sum_counts.get(s, 0) + 1
                
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_chosen = chosen.copy()
                    best_sum_counts = sum_counts.copy()
                    best_chosen_set = chosen_set.copy()
        else:
            # Swap
            if not chosen:
                continue
            u = random.choice(chosen)
            v = random.randint(1, 1000)
            if v in chosen_set:
                continue
            
            # Remove u
            for y in chosen:
                if y != u:
                    s = u + y
                    sum_counts[s] -= 1
                    if sum_counts[s] == 0:
                        del sum_counts[s]
            s = 2 * u
            sum_counts[s] -= 1
            if sum_counts[s] == 0:
                del sum_counts[s]
            
            # Check add v
            valid = True
            for y in chosen:
                if y != u:
                    s = v + y
                    if sum_counts.get(s, 0) >= 2:
                        valid = False
                        break
            if valid:
                s = 2 * v
                if sum_counts.get(s, 0) >= 2:
                    valid = False
            
            if valid:
                for y in chosen:
                    if y != u:
                        s = v + y
                        sum_counts[s] = sum_counts.get(s, 0) + 1
                s = 2 * v
                sum_counts[s] = sum_counts.get(s, 0) + 1
                
                chosen_set.remove(u)
                chosen_set.add(v)
                # Update list
                idx = chosen.index(u)
                chosen[idx] = v
            else:
                # Rollback
                for y in chosen:
                    if y != u:
                        s = u + y
                        sum_counts[s] = sum_counts.get(s, 0) + 1
                s = 2 * u
                sum_counts[s] = sum_counts.get(s, 0) + 1

    return best_chosen
