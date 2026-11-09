def construct():
    import random
    import time
    from fractions import Fraction
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [1.0 / d for d in denominators]
    
    best_deficit = 1.0
    best_indices = None
    
    start_time = time.time()
    time_limit = 55.0
    
    def local_search(current_indices):
        current_set = set(current_indices)
        current_sum = sum(recips[i] for i in current_set)
        
        if current_sum >= 1.0:
            return None, 0.0
            
        improved = True
        # Limit iterations to prevent infinite loops due to float precision issues
        max_iters = 100
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try adding
            for i in range(n):
                if i not in current_set:
                    new_sum = current_sum + recips[i]
                    if new_sum < 1.0:
                        current_set.add(i)
                        current_sum = new_sum
                        improved = True
                        break
                        
            # 2. Try swapping
            if not improved:
                for i in list(current_set):
                    sum_without_i = current_sum - recips[i]
                    for j in range(n):
                        if j not in current_set:
                            new_sum = sum_without_i + recips[j]
                            if new_sum < 1.0 and new_sum > current_sum:
                                current_set.remove(i)
                                current_set.add(j)
                                current_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
                        
        return list(current_set), current_sum

    # Restart 0: Greedy (Largest to Smallest)
    greedy_indices = []
    s = 0.0
    for i in range(n):
        if s + recips[i] < 1.0:
            greedy_indices.append(i)
            s += recips[i]
            
    if greedy_indices:
        res, val = local_search(greedy_indices)
        if res and val < 1.0:
            if 1.0 - val < best_deficit:
                best_deficit = 1.0 - val
                best_indices = res

    # Random Restarts
    for r in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        indices = list(range(n))
        random.shuffle(indices)
        current_sum = 0.0
        current_set = []
        for i in indices:
            if current_sum + recips[i] < 1.0:
                current_set.append(i)
                current_sum += recips[i]
        
        if not current_set:
            continue
            
        res, val = local_search(current_set)
        if res and val < 1.0:
            if 1.0 - val < best_deficit:
                best_deficit = 1.0 - val
                best_indices = res

    # Final verification with Fraction to ensure strict inequality
    if best_indices:
        final_sum = sum(Fraction(1, denominators[i]) for i in best_indices)
        if final_sum < 1:
            return best_indices
    
    # Fallback: return a known valid small set if something went wrong
    return [0]
