import random
import itertools
import time

def is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(arr):
    n = len(arr)
    # Check size 4 first, if found return 4 (since we know 3 is min possible for 12 elems? No, 3 is min for any set? No, 3 is min for sets with no 3-AP? No.)
    # We just want the max.
    # If we find a 5, return 5.
    # If we find a 4, return 4.
    # If no 4, return 3.
    
    # Check for 5
    for combo in itertools.combinations(arr, 5):
        if is_sidon(combo):
            return 5
    # Check for 4
    for combo in itertools.combinations(arr, 4):
        if is_sidon(combo):
            return 4
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # Leave 10s margin
    
    # Try to find a set with max sidon size 3
    best_set = None
    best_val = 12 # Max possible is 12
    
    # Initialize with random sets
    for _ in range(1000):
        if time.time() > deadline:
            break
        s = set(random.sample(range(129), 12))
        arr = list(s)
        val = max_sidon_size(arr)
        if val < best_val:
            best_val = val
            best_set = arr
            if best_val == 3:
                return best_set
                
    # If we didn't find 3, try local search on the best found
    if best_set is not None:
        current = best_set[:]
        current_val = best_val
        for _ in range(10000):
            if time.time() > deadline:
                break
            if current_val == 3:
                return current
                
            # Pick a random 4-Sidon subset to break
            # Find one
            sidon_4 = None
            for combo in itertools.combinations(current, 4):
                if is_sidon(combo):
                    sidon_4 = combo
                    break
            
            if sidon_4 is None:
                # Should not happen if current_val >= 4
                return current
                
            # Try to move one element of sidon_4
            idx_to_move = random.choice([current.index(x) for x in sidon_4])
            old_val = current[idx_to_move]
            
            # Try new values
            candidates = [v for v in range(129) if v not in current]
            random.shuffle(candidates)
            
            improved = False
            for new_val in candidates:
                if time.time() > deadline:
                    break
                trial = current[:]
                trial[idx_to_move] = new_val
                trial_val = max_sidon_size(trial)
                if trial_val < current_val:
                    current = trial
                    current_val = trial_val
                    improved = True
                    break
            if not improved:
                # Maybe try moving a different element or random move
                # Just do a random move to escape local minima
                idx = random.randint(0, 11)
                candidates = [v for v in range(129) if v not in current]
                new_val = random.choice(candidates)
                trial = current[:]
                trial[idx] = new_val
                trial_val = max_sidon_size(trial)
                if trial_val <= current_val:
                    current = trial
                    current_val = trial_val
                    
    if best_set is not None:
        return best_set
    else:
        return list(range(12))
