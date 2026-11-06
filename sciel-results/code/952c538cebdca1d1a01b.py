import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    fixed = [1, 2, 5, 11]
    fixed_set = set(fixed)
    max_val = 284
    
    # Target size
    target_size = 20
    
    best_set = None
    best_size = 0
    
    def get_sums(a):
        sums = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
        return sums

    def count_collisions(a):
        # Returns number of duplicate sums
        sums = []
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.append(a[i] + a[j])
        return len(sums) - len(set(sums))

    # We will try to find a set of size 20.
    # If we find one, we return it.
    # If not, we keep the best valid set found.
    
    # To speed up collision counting, we can maintain the set of sums incrementally.
    # But for simplicity and robustness, let's use a helper class or just recompute if N is small (20).
    # N=20, pairs=210. Recomputing is fast.
    
    def evaluate(a):
        return count_collisions(a)

    def random_initial_state(size):
        # Start with fixed
        a = list(fixed)
        remaining = [x for x in range(1, max_val + 1) if x not in fixed_set]
        random.shuffle(remaining)
        a.extend(remaining[:size - len(fixed)])
        random.shuffle(a)
        return a

    # Simulated Annealing parameters
    T_start = 100.0
    T_end = 0.01
    cooling_rate = 0.999
    
    # We will run several restarts
    # Each restart tries to minimize collisions for a specific size
    # We start with size 20. If we fail to get 0 collisions, we might try 19, etc.
    # Or just run SA on size 20, and if it finds 0, great.
    # If it doesn't, maybe it gets stuck at 1 collision.
    # A "kick" strategy: if stuck, remove an element, then add a different one.
    
    for restart in range(50): # Max restarts
        if time.time() > deadline:
            break
            
        current_size = 20
        # Maybe try smaller sizes if 20 fails repeatedly?
        # But 20 is the target. Let's stick to 20 first.
        
        a = random_initial_state(current_size)
        current_cost = evaluate(a)
        
        if current_cost == 0:
            best_set = a
            best_size = current_size
            break # Found optimal
        
        T = T_start
        stuck_count = 0
        best_in_run = current_cost
        best_a_in_run = list(a)
        
        while T > T_end:
            if time.time() > deadline:
                break
                
            # Choose a move
            move_type = random.random()
            new_a = list(a)
            
            if move_type < 0.7:
                # Move a non-fixed element
                non_fixed_indices = [i for i, x in enumerate(a) if x not in fixed_set]
                if not non_fixed_indices:
                    T *= cooling_rate
                    continue
                idx = random.choice(non_fixed_indices)
                old_val = a[idx]
                new_val = random.randint(1, max_val)
                # Ensure distinctness
                if new_val in a and new_val != old_val:
                    # Try again or skip
                    pass 
                else:
                    new_a[idx] = new_val
            elif move_type < 0.85:
                # Remove a non-fixed element
                non_fixed_indices = [i for i, x in enumerate(a) if x not in fixed_set]
                if non_fixed_indices:
                    idx = random.choice(non_fixed_indices)
                    new_a.pop(idx)
            else:
                # Add a random element
                if len(a) < 25: # Allow growing a bit to escape?
                    # But we want size 20.
                    # If we add, we must remove later?
                    # Variable size search:
                    new_val = random.randint(1, max_val)
                    if new_val not in a:
                        new_a.append(new_val)
            
            if new_a == a:
                T *= cooling_rate
                continue
                
            new_cost = evaluate(new_a)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                a = new_a
                current_cost = new_cost
                
                if current_cost < best_in_run:
                    best_in_run = current_cost
                    best_a_in_run = list(a)
                    
                if current_cost == 0:
                    # Found a valid set
                    if len(a) > best_size:
                        best_size = len(a)
                        best_set = list(a)
                    # If we found size 20, we can break out
                    if best_size >= 20:
                        T = 0
                        break
                    # If we found smaller, keep searching for larger?
                    # Actually, if we are in size 20 restart, and we found valid size 20, we are done.
                    # If we found valid size < 20, we should try to add elements.
                    # The move 'Add' helps this.
                    
            else:
                stuck_count += 1
                # If stuck for too long, kick
                if stuck_count > 100:
                    # Remove a random non-fixed element to reduce size and break local min
                    non_fixed_indices = [i for i, x in enumerate(a) if x not in fixed_set]
                    if non_fixed_indices:
                        idx = random.choice(non_fixed_indices)
                        a.pop(idx)
                        current_cost = evaluate(a)
                    stuck_count = 0
            
            T *= cooling_rate
        
        # After SA run, update best
        if best_in_run == 0 and len(best_a_in_run) > best_size:
            best_size = len(best_a_in_run)
            best_set = best_a_in_run
            
        # If we found size 20, we can stop
        if best_size >= 20:
            break
            
    if best_set:
        return sorted(best_set)
    else:
        # Fallback: Return the fixed set if nothing better found
        return sorted(fixed)
