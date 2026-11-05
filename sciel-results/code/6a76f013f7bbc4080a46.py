import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s kill (or 150s if that's the limit, but 120s is stated in prompt "120 s")
    # Prompt says: "Each evaluation must finish within 120 s."
    # "The evaluator KILLS the program after 150s wall-clock."
    # So 120s is the hard limit for the function to return.
    
    prescribed = [1, 2, 5, 11]
    N = 284
    
    def compute_conflicts(s):
        """Returns number of duplicate sums (conflicts) and the set of sums."""
        sums = {}
        conflicts = 0
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                val = s[i] + s[j]
                if val in sums:
                    conflicts += 1
                else:
                    sums[val] = True
        return conflicts, sums

    def is_valid(s):
        conflicts, _ = compute_conflicts(s)
        return conflicts == 0

    best_set = prescribed[:]
    best_size = len(prescribed)
    
    # Initial check
    if is_valid(best_set):
        pass # It is valid
    
    # Simulated Annealing Parameters
    # We start with a set that contains prescribed and some random elements.
    # We want to maximize size, but we penalize conflicts.
    # Score = size - penalty * conflicts
    
    penalty = 10.0 # High penalty to ensure we prioritize validity
    
    def get_score(s):
        conflicts, _ = compute_conflicts(s)
        return len(s) - penalty * conflicts

    # Initialize SA
    current_set = prescribed[:]
    # Add some random elements to start with a larger set, even if invalid
    random.seed(42)
    candidates = list(range(1, N+1))
    random.shuffle(candidates)
    for x in candidates:
        if x in current_set:
            continue
        if len(current_set) >= 20:
            break
        current_set.append(x)
        
    current_score = get_score(current_set)
    best_sa_set = current_set[:]
    best_sa_score = current_score
    
    if current_score > get_score(best_set):
         # This score is size - penalty*conflicts. 
         # If conflicts=0, score=size. 
         # best_set is valid, so its score is its size.
         # So we compare directly if valid, or score if invalid.
         # But we only care about valid sets for the final answer.
         # We track the best *valid* set found.
         pass

    # Track best valid set found during SA
    # Initial best valid is prescribed
    
    temperature = 5.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    # Pre-compute sums for speed? 
    # compute_conflicts is O(N^2). With N=20, it's 210 ops. Very fast.
    # SA can do millions of steps.
    
    steps = 0
    while time.time() < deadline:
        if temperature < min_temp:
            temperature = 5.0 # Restart
            # Maybe reset current_set to a new random state
            current_set = prescribed[:]
            random.shuffle(candidates)
            for x in candidates:
                if x in current_set:
                    continue
                if len(current_set) >= 20:
                    break
                current_set.append(x)
            current_score = get_score(current_set)
            
        # Generate a neighbor
        # 1. Add
        # 2. Remove
        # 3. Swap
        
        move_type = random.random()
        
        if len(current_set) < 20 and move_type < 0.4:
            # Add
            x = random.randint(1, N)
            if x not in current_set:
                new_set = current_set + [x]
            else:
                continue
        elif len(current_set) > 4 and move_type < 0.7:
            # Remove
            idx = random.randint(0, len(current_set)-1)
            # Do not remove prescribed elements? 
            # If we remove prescribed, we must add them back eventually.
            # It's safer to only remove non-prescribed elements.
            non_prescribed_idx = [i for i in range(len(current_set)) if current_set[i] not in prescribed]
            if non_prescribed_idx:
                idx = non_prescribed_idx[random.randint(0, len(non_prescribed_idx)-1)]
                new_set = current_set[:idx] + current_set[idx+1:]
            else:
                continue
        else:
            # Swap
            # Pick an element to remove and one to add
            non_prescribed_idx = [i for i in range(len(current_set)) if current_set[i] not in prescribed]
            if not non_prescribed_idx:
                continue
            idx = non_prescribed_idx[random.randint(0, len(non_prescribed_idx)-1)]
            old_val = current_set[idx]
            new_val = random.randint(1, N)
            if new_val == old_val or new_val in current_set:
                continue
            new_set = current_set[:]
            new_set[idx] = new_val
        
        new_score = get_score(new_set)
        
        # Check if new set is valid and better than best valid
        if is_valid(new_set):
            if len(new_set) > best_size:
                best_size = len(new_set)
                best_set = new_set[:]
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0 or random.random() < pow(2.71828, delta / temperature):
            current_set = new_set
            current_score = new_score
            
        temperature *= cooling_rate
        steps += 1
        
        # Periodically try to "clean up" or just let SA handle it.
        
    return best_set
