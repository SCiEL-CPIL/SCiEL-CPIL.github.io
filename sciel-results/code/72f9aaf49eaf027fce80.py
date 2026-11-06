import random
import time

# EVOLVE-BLOCK-START
def construct():
    MAX_SUM = 2000
    
    def greedy_construct(order):
        counts = [0] * (MAX_SUM + 1)
        chosen = []
        chosen_set = set()
        for x in order:
            valid = True
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            if valid:
                if counts[2 * x] >= 2:
                    valid = False
            
            if valid:
                for y in chosen:
                    counts[x + y] += 1
                counts[2 * x] += 1
                chosen.append(x)
                chosen_set.add(x)
        return chosen, chosen_set, counts

    best_size = 0
    best_chosen = []
    best_counts = []
    best_set = set()
    
    candidates = list(range(1, 1001))
    
    # 1. Find a good initial solution via multiple greedy restarts
    for _ in range(50):
        random.shuffle(candidates)
        chosen, chosen_set, counts = greedy_construct(candidates)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = chosen
            best_counts = counts
            best_set = chosen_set
            
    # 2. Local Search
    current_chosen = best_chosen[:]
    current_set = best_set.copy()
    current_counts = best_counts[:]
    excluded = [x for x in range(1, 1001) if x not in current_set]
    
    for step in range(5000):
        if not excluded:
            break
            
        c = random.choice(excluded)
        
        # Check if c can be added
        valid = True
        for y in current_chosen:
            if current_counts[c + y] >= 2:
                valid = False
                break
        if valid and current_counts[2 * c] >= 2:
            valid = False
            
        if valid:
            for y in current_chosen:
                current_counts[c + y] += 1
            current_counts[2 * c] += 1
            current_chosen.append(c)
            current_set.add(c)
            excluded.remove(c)
            if len(current_chosen) > best_size:
                best_size = len(current_chosen)
                best_chosen = current_chosen[:]
                best_counts = current_counts[:]
                best_set = current_set.copy()
        else:
            # Try a swap
            if random.random() < 0.5 and current_chosen:
                idx = random.randint(0, len(current_chosen) - 1)
                r = current_chosen[idx]
                
                # Remove r
                for y in current_chosen:
                    if y != r:
                        current_counts[r + y] -= 1
                current_counts[2 * r] -= 1
                current_chosen.pop(idx)
                current_set.remove(r)
                
                # Try to add c
                valid_c = True
                for y in current_chosen:
                    if current_counts[c + y] >= 2:
                        valid_c = False
                        break
                if valid_c and current_counts[2 * c] >= 2:
                    valid_c = False
                    
                if valid_c:
                    for y in current_chosen:
                        current_counts[c + y] += 1
                    current_counts[2 * c] += 1
                    current_chosen.append(c)
                    current_set.add(c)
                    
                    # Update excluded
                    if r not in excluded:
                        excluded.append(r)
                    if c in excluded:
                        excluded.remove(c)
                else:
                    # Restore r
                    for y in current_chosen:
                        current_counts[r + y] += 1
                    current_counts[2 * r] += 1
                    current_chosen.append(r)
                    current_set.add(r)
                    
    return best_chosen
# EVOLVE-BLOCK-END
