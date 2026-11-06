import random
import time
import math

def construct():
    N = 1000
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Greedy initialization
    def greedy_init():
        chosen = []
        counts = [0] * (2 * N + 1)
        for x in range(1, N + 1):
            # Check if adding x violates constraints
            # New sums: x + y for y in chosen, and 2x
            # We need to check if any count becomes 3
            ok = True
            # 2x
            if counts[2 * x] == 2:
                ok = False
            else:
                for y in chosen:
                    s = x + y
                    if counts[s] == 2:
                        ok = False
                        break
            if ok:
                chosen.append(x)
                counts[2 * x] += 1
                for y in chosen[:-1]: # chosen[:-1] is the old chosen
                    counts[x + y] += 1
        return chosen, counts

    best_set, best_counts = greedy_init()
    best_size = len(best_set)
    
    # SA State
    in_set = [False] * (N + 1)
    for x in best_set:
        in_set[x] = True
    
    current_set = list(best_set)
    current_counts = list(best_counts)
    current_size = best_size
    
    # Calculate initial violation (should be 0)
    def calc_violation(counts):
        v = 0
        for s in range(2, 2 * N + 1):
            if counts[s] > 2:
                v += counts[s] - 2
        return v

    current_violation = calc_violation(current_counts)
    
    lambda_pen = 10.0
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    # Precompute indices for random access?
    # Just use random.choice for removal, random.randint for addition
    
    while time.time() - start_time < time_limit:
        # Generate move
        move_type = random.random()
        if move_type < 0.4:
            # Add
            candidates = [i for i in range(1, N + 1) if not in_set[i]]
            if not candidates:
                continue
            x = random.choice(candidates)
            # Simulate add
            delta_size = 1
            delta_viol = 0
            # Check 2x
            if current_counts[2 * x] == 2:
                delta_viol += 1
            for y in current_set:
                s = x + y
                if current_counts[s] == 2:
                    delta_viol += 1
            
            delta_E = -delta_size + lambda_pen * delta_viol
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                in_set[x] = True
                current_set.append(x)
                current_size += delta_size
                current_counts[2 * x] += 1
                for y in current_set[:-1]:
                    current_counts[x + y] += 1
                current_violation += delta_viol
                
                if current_violation == 0 and current_size > best_size:
                    best_size = current_size
                    best_set = list(current_set)
                    best_counts = list(current_counts)
                    
        elif move_type < 0.8:
            # Remove
            if not current_set:
                continue
            x = random.choice(current_set)
            delta_size = -1
            delta_viol = 0
            if current_counts[2 * x] == 3:
                delta_viol -= 1
            for y in current_set:
                if y == x: continue
                s = x + y
                if current_counts[s] == 3:
                    delta_viol -= 1
            
            delta_E = -delta_size + lambda_pen * delta_viol
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                in_set[x] = False
                current_set.remove(x)
                current_size += delta_size
                current_counts[2 * x] -= 1
                for y in current_set:
                    current_counts[x + y] -= 1
                current_violation += delta_viol
                
        else:
            # Swap
            if not current_set:
                continue
            x = random.choice(current_set)
            candidates = [i for i in range(1, N + 1) if not in_set[i]]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # Remove x, Add y
            # Delta size 0
            delta_viol = 0
            
            # Remove x effects
            if current_counts[2 * x] == 3:
                delta_viol -= 1
            for z in current_set:
                if z == x: continue
                s = x + z
                if current_counts[s] == 3:
                    delta_viol -= 1
            
            # Add y effects (on the set after removing x)
            # Note: counts are currently for set with x.
            # We need to check counts after removing x, then adding y.
            # This is tricky to do incrementally without updating.
            # Simpler: Just update, check, and revert if rejected?
            # Or calculate delta carefully.
            
            # Let's do a "try and revert" approach for swaps to be safe and simple, 
            # though it's slightly slower.
            # Or just stick to Add/Remove which are simpler to delta.
            # Given the budget, Add/Remove might be sufficient.
            # Let's disable Swap for now to keep it robust and fast.
            pass
            
        T *= cooling
        if T < T_min:
            # Restart or just keep going?
            # If we are stuck, we might want to perturb.
            # For now, just keep going with low T (greedy local search)
            pass
            
    return best_set
