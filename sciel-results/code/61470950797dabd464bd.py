import math
import time

def construct():
    def calc_score(u):
        if not u:
            return 0
        max_u = u[-1]
        sum_mask = 0
        diff_mask = 0
        
        for i in u:
            for j in u:
                sum_mask |= (1 << (i + j))
                diff_mask |= (1 << (i - j + max_u))
                
        num_sums = bin(sum_mask).count('1')
        num_diffs = bin(diff_mask).count('1')
        
        if num_sums == 0:
            return 0
            
        ratio = num_diffs / num_sums
        if ratio <= 1:
            return 0
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 0
            
        return 1 + math.log2(ratio) / denom

    # Start with a known good configuration
    # {0, 1, 3, 7, 15} gives ~1.098
    best_u = [0, 1, 3, 7, 15]
    best_score = calc_score(best_u)
    
    start_time = time.time()
    max_time = 100  # Leave 20s buffer
    
    # Hill Climbing
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        # Try adding an element
        max_u = best_u[-1]
        # Try elements up to max_u + 20 or so
        for new_elem in range(1, max_u + 20):
            if new_elem in best_u:
                continue
            candidate = sorted(best_u + [new_elem])
            s = calc_score(candidate)
            if s > best_score:
                best_score = s
                best_u = candidate
                improved = True
                break  # Restart loop with new best
        
        # Try removing an element (except 0)
        for i in range(1, len(best_u)):
            candidate = best_u[:i] + best_u[i+1:]
            s = calc_score(candidate)
            if s > best_score:
                best_score = s
                best_u = candidate
                improved = True
                break
        
        # Try swapping an element with a new one
        if not improved and (time.time() - start_time) < max_time:
             for i in range(1, len(best_u)):
                 old_elem = best_u[i]
                 # Try swapping with elements in a range
                 for new_elem in range(1, max_u + 10):
                     if new_elem in best_u:
                         continue
                     if new_elem == old_elem:
                         continue
                     candidate = best_u[:i] + [new_elem] + best_u[i+1:]
                     candidate = sorted(candidate)
                     s = calc_score(candidate)
                     if s > best_score:
                         best_score = s
                         best_u = candidate
                         improved = True
                         break
                 if improved:
                     break

    return best_u
