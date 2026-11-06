import time
import random
import math

def construct():
    start_time = time.time()
    # Use a safer deadline to account for overhead and potential slowness
    deadline = start_time + 100.0
    
    # Helper to generate initial sets
    def generate_initial_sets():
        sets = []
        
        # 1. Simple interval
        sets.append(set(range(1, 33)))
        
        # 2. Sums of two squares
        candidates = set()
        for i in range(1, 20):
            for j in range(1, 20):
                val = i*i + j*j
                if 1 <= val <= 512:
                    candidates.add(val)
        if len(candidates) >= 32:
            sets.append(set(random.sample(sorted(candidates), 32)))
            
        # 3. Grid-like structure
        grid_set = set()
        for i in range(1, 6):
            for j in range(1, 7):
                grid_set.add(i * 10 + j)
        if len(grid_set) >= 32:
             sets.append(set(random.sample(sorted(grid_set), 32)))
             
        # 4. Random
        sets.append(set(random.sample(range(1, 513), 32)))
        
        # 5. Another random
        sets.append(set(random.sample(range(1, 513), 32)))
        
        return sets

    # Optimized score calculation
    def get_score(S):
        S_set = S
        S_sorted = sorted(S)
        d_set = set()
        n = len(S_sorted)
        contains = S_set.__contains__
        for i in range(n):
            a = S_sorted[i]
            for j in range(i + 1, n):
                b = S_sorted[j]
                d = b - a
                c = b + d
                if contains(c):
                    d_set.add(d)
        return len(d_set)

    best_set = None
    best_score = -1
    
    num_restarts = 4
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        # Check time
        if time.time() >= deadline - 2.0:
            break
            
        # Select an initial set
        if r == 0:
            init_sets = generate_initial_sets()
            current_set = max(init_sets, key=get_score)
        else:
            current_set = set(random.sample(range(1, 513), 32))
            
        current_score = get_score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            
        # Simulated Annealing Parameters
        start_temp = 5.0
        end_temp = 0.01
        cooling_rate = 0.995
        
        restart_deadline = start_time + (r + 1) * time_per_restart
        
        temp = start_temp
        
        # Convert to list for faster random access, but we need to update it if we accept moves
        # Since N is small, list(current_set) is cheap.
        
        while time.time() < restart_deadline:
            # Generate neighbor
            # Use list for choice to avoid set iteration overhead
            current_list = list(current_set)
            elem_to_replace = random.choice(current_list)
            
            if random.random() < 0.5:
                # Random replacement
                new_val = random.randint(1, 512)
                if new_val in current_set:
                    continue
            else:
                # Local nudge
                delta = random.randint(-5, 5)
                if delta == 0:
                    continue
                new_val = elem_to_replace + delta
                if new_val < 1 or new_val > 512:
                    continue
                if new_val in current_set:
                    continue
            
            # Apply move
            # Instead of creating a new set, modify in place and revert if rejected?
            # Or create new set. N=32 is small.
            new_set = set(current_set)
            new_set.remove(elem_to_replace)
            new_set.add(new_val)
            
            new_score = get_score(new_set)
            delta_score = new_score - current_score
            
            if delta_score >= 0:
                current_set = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            else:
                if temp > 0:
                    # Use a faster approximation for exp if needed, but math.exp is fine
                    prob = math.exp(delta_score / temp)
                    if random.random() < prob:
                        current_set = new_set
                        current_score = new_score
            
            # Cool down
            temp *= cooling_rate
            if temp < end_temp:
                temp = end_temp

    return sorted(best_set)
