import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin for 150s limit
    
    def compute_score(U):
        if not U or len(U) <= 1:
            return 1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        u_arr = np.array(U, dtype=np.int64)
        n = len(u_arr)
        
        # Use broadcasting for efficient computation
        # Sumset: all pairwise sums
        sumset = set()
        diffset = set()
        
        # For efficiency with small n, use nested loops but with early exit
        # For larger n, we could use FFT-based convolution, but for n < 100, direct is fine
        for i in range(n):
            for j in range(i, n):
                s = u_arr[i] + u_arr[j]
                sumset.add(s)
                if i != j:
                    d = u_arr[i] - u_arr[j]
                    diffset.add(d)
                    diffset.add(-d)
        
        # Add 0 to diffset (a-a = 0)
        diffset.add(0)
        
        size_sum = len(sumset)
        size_diff = len(diffset)
        
        if size_sum == 0:
            return 1.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
        
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    # Known optimal Golomb rulers (these have all distinct positive differences)
    # These maximize |U-U| for given max(U), which helps the ratio
    golomb_rulers = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 25],
        8: [0, 1, 4, 10, 12, 17, 25, 29],
        9: [0, 1, 4, 10, 12, 17, 25, 29, 34],
        10: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41],
    }
    
    # Additional candidates: subsets of Golomb rulers, and some hand-crafted sets
    initial_sets = []
    for n in range(1, 11):
        if n in golomb_rulers:
            initial_sets.append(golomb_rulers[n])
    
    # Add some variations
    extra_sets = [
        [0, 1, 3, 7],
        [0, 1, 4, 6, 10],
        [0, 2, 5, 11],
        [0, 1, 5, 11],
        [0, 3, 7, 12],
        [0, 1, 4, 9, 11, 18],
        [0, 2, 7, 12, 18],
        [0, 1, 5, 10, 16],
    ]
    initial_sets.extend(extra_sets)
    
    best_score = 0.0
    best_set = [0]
    
    for s in initial_sets:
        score = compute_score(s)
        if score > best_score:
            best_score = score
            best_set = list(s)
    
    # Local search with multiple restarts
    max_restarts = 5
    best_overall_score = best_score
    best_overall_set = best_set
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Start from best set or a random Golomb ruler
        if restart == 0:
            current_set = list(best_overall_set)
        else:
            # Pick a random Golomb ruler as starting point
            n = random.randint(3, 10)
            if n in golomb_rulers:
                current_set = list(golomb_rulers[n])
            else:
                current_set = list(best_overall_set)
        
        current_score = compute_score(current_set)
        
        # Local search phase
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Generate a neighbor
            neighbor = list(current_set)
            op = random.randint(0, 3)
            
            if op == 0 and len(neighbor) > 1:
                # Remove a random element (not 0)
                idx = random.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
                if len(neighbor) == 0:
                    continue
                if 0 not in neighbor:
                    neighbor.append(0)
                    neighbor.sort()
            elif op == 1:
                # Add a random element
                max_val = max(neighbor) if neighbor else 0
                upper_bound = max(2 * max_val + 15, 25)
                new_val = random.randint(1, upper_bound)  # Avoid 0
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
            elif op == 2:
                # Modify a random element (not 0) to a nearby value
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    old_val = neighbor[idx]
                    # Try small perturbations
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        new_val = old_val + delta
                        if new_val >= 0 and new_val not in neighbor:
                            neighbor[idx] = new_val
                            neighbor.sort()
                            break
                    else:
                        continue
            else:
                # Swap: remove one element and add another
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    removed_val = neighbor.pop(idx)
                    max_val = max(neighbor) if neighbor else 0
                    upper_bound = max(2 * max_val + 15, 25)
                    new_val = random.randint(1, upper_bound)
                    if new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor.sort()
                    else:
                        # Revert
                        neighbor.append(removed_val)
                        neighbor.sort()
                        continue
            
            if not neighbor or 0 not in neighbor:
                continue
                
            new_score = compute_score(neighbor)
            
            if new_score > current_score:
                current_set = neighbor
                current_score = new_score
                no_improve_count = 0
                if current_score > best_overall_score:
                    best_overall_score = current_score
                    best_overall_set = list(current_set)
            else:
                no_improve_count += 1
                # Small probability of accepting worse move to escape local minima
                if random.random() < 0.005:
                    current_set = neighbor
                    current_score = new_score
    
    return best_overall_set
# EVOLVE-BLOCK-END
