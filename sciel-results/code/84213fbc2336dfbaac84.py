import time
import heapq
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    target_num = 5
    target_den = 31
    
    # Initial known solution
    best_solution = [8, 31, 248]
    best_max_den = 248
    
    # Heap elements: (max_den_so_far, counter, num, den, min_d, path)
    # counter is used for tie-breaking to avoid comparing lists
    counter = 0
    heap = []
    
    # Start state
    heapq.heappush(heap, (0, counter, target_num, target_den, 1, []))
    counter += 1
    
    while heap:
        if time.time() > deadline:
            break
            
        max_d, _, num, den, min_d, path = heapq.heappop(heap)
        
        # If the current max denominator is already >= best known, 
        # no solution from this branch can be better.
        if max_d >= best_max_den:
            break
            
        # If remainder is 0, we found a valid solution
        if num == 0:
            # The max denominator of this solution is max_d
            # Since we pop in increasing order of max_d, 
            # this is the optimal solution.
            return path
            
        # Calculate bounds for next denominator d
        # 1/d <= num/den  => d >= den/num
        d_min = max(min_d + 1, (den + num - 1) // num)
        
        # d must be < best_max_den to improve
        d_max = best_max_den - 1
        
        if d_min > d_max:
            continue
            
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                break
                
            # new_rem = num/den - 1/d = (num*d - den) / (den*d)
            new_num = num * d - den
            new_den = den * d
            
            if new_num < 0:
                # 1/d > num/den, so remainder becomes negative.
                # Since d is increasing, new_num will become more negative?
                # No, new_num = num*d - den. As d increases, new_num increases.
                # So if new_num < 0, it might become positive later.
                # Wait, d_min ensures 1/d <= num/den, so new_num >= 0.
                # So new_num < 0 should not happen if d_min is correct.
                continue
                
            if new_num == 0:
                # Exact match
                # The solution is path + [d]
                # Max den is max(max_d, d)
                heapq.heappush(heap, (max(max_d, d), counter, 0, 1, d, path + [d]))
                counter += 1
                continue
                
            # Simplify
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            new_max_d = max(max_d, d)
            
            # Optional pruning: 
            # If new_max_d >= best_max_den, skip
            if new_max_d >= best_max_den:
                continue
                
            heapq.heappush(heap, (new_max_d, counter, new_num, new_den, d, path + [d]))
            counter += 1
            
    return best_solution
