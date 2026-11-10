import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    denominators = list(range(2, 49))
    n = len(denominators)
    values = [1.0 / d for d in denominators]
    
    best_deficit = 1.0
    best_indices = None
    
    # Multiple restarts
    for restart in range(100):
        if time.time() > deadline:
            break
            
        indices = list(range(n))
        random.shuffle(indices)
        current_indices = []
        current_sum = 0.0
        for i in indices:
            if current_sum + values[i] < 1.0:
                current_indices.append(i)
                current_sum += values[i]
                
        local_best_deficit = 1.0 - current_sum
        local_best_indices = current_indices.copy()
        
        # Local search
        for _ in range(2000):
            if time.time() > deadline:
                break
                
            move_type = random.randint(0, 2)
            
            if move_type == 0:
                candidates = [i for i in range(n) if i not in current_indices]
                if not candidates: continue
                i = random.choice(candidates)
                new_sum = current_sum + values[i]
                if new_sum < 1.0:
                    current_indices.append(i)
                    current_sum = new_sum
                    deficit = 1.0 - current_sum
                    if deficit < local_best_deficit:
                        local_best_deficit = deficit
                        local_best_indices = current_indices.copy()
            elif move_type == 1:
                if not current_indices: continue
                i = random.choice(current_indices)
                current_indices.remove(i)
                current_sum -= values[i]
            else:
                if not current_indices: continue
                i_remove = random.choice(current_indices)
                candidates = [i for i in range(n) if i not in current_indices]
                if not candidates: continue
                i_add = random.choice(candidates)
                new_sum = current_sum - values[i_remove] + values[i_add]
                if new_sum < 1.0:
                    current_indices.remove(i_remove)
                    current_indices.append(i_add)
                    current_sum = new_sum
                    deficit = 1.0 - current_sum
                    if deficit < local_best_deficit:
                        local_best_deficit = deficit
                        local_best_indices = current_indices.copy()
                        
        if local_best_deficit < best_deficit:
            best_deficit = local_best_deficit
            best_indices = local_best_indices
            
    return best_indices
