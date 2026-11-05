import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    
    def calc_cost(signs):
        max_abs = 0
        for d in range(1, n):
            for r in range(d):
                max_ending_here = 0
                max_so_far = 0
                min_ending_here = 0
                min_so_far = 0
                idx = r
                while idx < n:
                    x = signs[idx]
                    # Max subarray
                    max_ending_here = max(0, max_ending_here + x)
                    if max_ending_here > max_so_far:
                        max_so_far = max_ending_here
                    # Min subarray
                    min_ending_here = min(0, min_ending_here + x)
                    if min_ending_here < min_so_far:
                        min_so_far = min_ending_here
                    idx += d
                curr_max = max_so_far if max_so_far > -min_so_far else -min_so_far
                if curr_max > max_abs:
                    max_abs = curr_max
        return max_abs

    # Initialize
    random.seed(42)
    signs = [random.choice([-1, 1]) for _ in range(n)]
    
    current_cost = calc_cost(signs)
    best_cost = current_cost
    best_signs = signs.copy()
    
    T = 5.0
    T_min = 0.5
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        i = random.randint(0, n - 1)
        signs[i] = -signs[i]
        
        new_cost = calc_cost(signs)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs.copy()
        else:
            delta = new_cost - current_cost
            if random.random() < math.exp(-delta / T):
                current_cost = new_cost
            else:
                signs[i] = -signs[i]
                
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_signs
