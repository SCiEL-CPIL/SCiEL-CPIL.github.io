import random
import math

def calculate_score(U):
    if not U or len(U) < 2:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
        
    val = 1 + (math.log(ratio) / denom)
    return val

def construct():
    best_set = [0, 1, 3]
    best_score = calculate_score(best_set)
    
    # Set of allowed values
    max_val = 50
    possible_values = list(range(1, max_val + 1))
    
    seeds = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 7],
        [0, 1, 2, 4, 8]
    ]
    
    # Add some random seeds
    for _ in range(5):
        n = random.randint(3, 6)
        elems = random.sample(possible_values, n-1)
        seeds.append([0] + sorted(elems))
        
    for seed in seeds:
        U = list(seed)
        current_score = calculate_score(U)
        
        # Local search
        for _ in range(1000):
            improved = False
            
            # Move 1: Swap an element
            if len(U) > 1:
                i = random.randint(0, len(U)-1)
                if U[i] == 0:
                    continue # Don't swap 0
                old_val = U[i]
                new_val = random.choice(possible_values)
                if new_val not in U:
                    U[i] = new_val
                    new_score = calculate_score(U)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        U[i] = old_val
            
            # Move 2: Add an element
            if len(U) < 10:
                new_val = random.choice(possible_values)
                if new_val not in U:
                    U.append(new_val)
                    new_score = calculate_score(U)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        U.pop()
                        
            # Move 3: Remove an element
            if len(U) > 2:
                i = random.randint(0, len(U)-1)
                if U[i] == 0:
                    continue
                removed_val = U.pop(i)
                new_score = calculate_score(U)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    U.insert(i, removed_val)
            
            if not improved:
                # Try a few more random moves or break?
                # To avoid getting stuck, we can just continue but it might be slow.
                # Let's just break if no improvement in a row? 
                # Actually, simple hill climbing can get stuck. 
                # But with 1000 iters and random restarts, it should be okay.
                pass

        if current_score > best_score:
            best_score = current_score
            best_set = U
            
    return best_set
