import time
import random
import math

def construct():
    deadline = time.time() + 140
    prescribed = [1, 2, 5, 11]
    max_val = 284
    target_size = 20
    
    # Function to count conflicts
    def count_conflicts(s):
        sums = set()
        conflicts = 0
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                val = lst[i] + lst[j]
                if val in sums:
                    conflicts += 1
                else:
                    sums.add(val)
        return conflicts

    # Try to find a valid set of size target_size using Simulated Annealing
    available = [x for x in range(1, max_val + 1) if x not in prescribed]
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        random.seed(restart * 12345 + 678)
        
        # Create initial set of size target_size
        current_set = set(prescribed)
        # Pick random elements
        # To improve success rate, we can try to pick elements that are spread out
        # But random is okay with restarts
        random.shuffle(available)
        for x in available[:target_size - len(prescribed)]:
            current_set.add(x)
            
        current_conflicts = count_conflicts(current_set)
        
        if current_conflicts == 0:
            return list(current_set)
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.995
        min_T = 1e-5
        
        while T > min_T and time.time() < deadline:
            # Pick a random non-prescribed element to swap
            non_prescribed = [x for x in current_set if x not in prescribed]
            if not non_prescribed:
                break
                
            x_to_remove = random.choice(non_prescribed)
            
            # Pick a random candidate to add
            candidates = [x for x in range(1, max_val + 1) if x not in current_set]
            if not candidates:
                break
            x_to_add = random.choice(candidates)
            
            # Perform swap
            new_set = current_set - {x_to_remove} | {x_to_add}
            
            new_conflicts = count_conflicts(new_set)
            
            delta = new_conflicts - current_conflicts
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_set = new_set
                current_conflicts = new_conflicts
                
                if current_conflicts == 0:
                    return list(current_set)
                    
            T *= cooling
            
    # Fallback: Greedy construction
    a, sums = [1, 2, 5, 11], set()
    for i, x in enumerate(a):
        for y in a[i:]:
            sums.add(x + y)
    for x in range(12, 285):
        new = {x + y for y in a} | {2 * x}
        if len(new) == len(a) + 1 and not new & sums:
            a.append(x)
            sums |= new
    return a
