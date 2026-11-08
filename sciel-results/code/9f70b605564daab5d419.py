import time
import math
import random

def score_set(U):
    if not U or 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    # Compute U+U and U-U
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    if num <= 1:
        return 1.0
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 1.0
    return 1 + math.log(num) / den

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    # Initial best: start with a known good small set
    best_set = [0, 1, 3, 7, 13]
    best_score = score_set(best_set)
    
    # Parameters for Simulated Annealing
    # We explore sets of size 3 to 20 and values up to 200
    # This is a structural change: wider search space + SA acceptance
    
    random.seed(42) # Fixed seed for reproducibility in this context, though random restarts use it
    
    def get_neighbor(current_set, current_score):
        """
        Generate a random neighbor by:
        1. Swapping an element
        2. Adding an element
        3. Removing an element
        """
        n = len(current_set)
        # Choose operation: 60% swap, 20% add, 20% remove
        op = random.random()
        
        # Define search range for new values
        # Scale range with current max to allow exploration
        max_val = max(current_set) if current_set else 0
        search_range = max(50, int(max_val * 1.5) + 10)
        
        if op < 0.6 and n > 0:
            # Swap
            idx = random.randint(0, n - 1)
            old_val = current_set[idx]
            new_val = random.randint(0, search_range)
            
            new_set = current_set[:]
            new_set[idx] = new_val
            
            # Handle 0 constraint
            if 0 not in new_set and new_val != 0:
                # If we removed 0 and didn't add it back, force 0 in
                # But we are swapping, so if old_val was 0, we must keep 0 or add it elsewhere?
                # Easier: if 0 is not in the set, replace a random non-zero element with 0?
                # Or just reject if 0 is lost?
                # Let's try to fix it: if 0 not in new_set, replace a random element with 0
                if 0 not in new_set:
                    # Find a non-zero element to replace with 0
                    non_zero_indices = [i for i, v in enumerate(new_set) if v != 0]
                    if non_zero_indices:
                        replace_idx = random.choice(non_zero_indices)
                        new_set[replace_idx] = 0
            
            # Remove duplicates and sort
            new_set = sorted(set(new_set))
            
            # Ensure 0 is in set
            if 0 not in new_set:
                return None, None
            
            # Ensure size is valid (3-20)
            if len(new_set) < 3 or len(new_set) > 20:
                return None, None
                
            return new_set, score_set(new_set)
            
        elif op < 0.8 and n < 20:
            # Add
            new_val = random.randint(0, search_range)
            if new_val in current_set:
                return None, None
            new_set = sorted(current_set + [new_val])
            if 0 not in new_set:
                return None, None
            return new_set, score_set(new_set)
            
        else:
            # Remove
            if n <= 3:
                return None, None
            # Don't remove 0
            removable = [i for i, v in enumerate(current_set) if v != 0]
            if not removable:
                return None, None
            idx = random.choice(removable)
            new_set = current_set[:idx] + current_set[idx+1:]
            if 0 not in new_set:
                return None, None
            return new_set, score_set(new_set)

    # Multi-start Simulated Annealing
    num_restarts = 10
    steps_per_restart = 500
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize current state
        # For first restart, use best_set. For others, random initialization.
        if r == 0:
            current_set = best_set[:]
            current_score = best_score
        else:
            # Random initialization
            size = random.randint(3, 10)
            vals = random.sample(range(0, 100), size)
            if 0 not in vals:
                vals[0] = 0
            current_set = sorted(set(vals))
            if len(current_set) < 3:
                continue
            current_score = score_set(current_set)
            
        # Simulated Annealing Loop
        T = 0.1  # Initial temperature
        T_min = 0.001
        cooling = 0.99
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            neighbor_set, neighbor_score = get_neighbor(current_set, current_score)
            
            if neighbor_set is None:
                continue
                
            delta = neighbor_score - current_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta > 0 or random.random() < math.exp(-delta / T):
                current_set = neighbor_set
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
                    
            T *= cooling
            if T < T_min:
                T = T_min

    # Final local polish on the best found
    # Do a quick hill climb on best_set to ensure no easy gains are missed
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(len(best_set)):
            old_val = best_set[i]
            # Try small swaps around current values
            for offset in range(-5, 6):
                new_val = old_val + offset
                if new_val < 0:
                    continue
                if new_val == old_val:
                    continue
                
                new_set = best_set[:]
                new_set[i] = new_val
                if old_val == 0 and new_val != 0:
                    continue # Can't move 0 out
                if 0 not in new_set and new_val != 0:
                    continue
                    
                new_set = sorted(set(new_set))
                if 0 not in new_set or len(new_set) < 3 or len(new_set) > 20:
                    continue
                
                s = score_set(new_set)
                if s > best_score + 1e-9:
                    best_score = s
                    best_set = new_set
                    improved = True
                    break # Restart loop for this i? Or just continue. Let's break inner and restart outer.
            if improved:
                break # Restart outer loop

    return best_set

if __name__ == "__main__":
    # Quick test
    s = construct()
    print(f"Result: {s}")
    print(f"Score: {score_set(s)}")
