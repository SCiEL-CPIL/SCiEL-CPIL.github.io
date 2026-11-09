import time
import random

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Heuristic orders
    candidates_base = list(range(1, 257))
    
    # Try descending, ascending, and several random shuffles
    orders = [
        sorted(candidates_base, reverse=True),
        sorted(candidates_base)
    ]
    
    # Add random shuffles
    for _ in range(50):
        random.shuffle(candidates_base)
        orders.append(candidates_base[:])

    for order in orders:
        if time.time() - start_time > 55:
            break
        
        current_set = []
        sums_list = []
        sums_set = set()
        
        for x in order:
            # Generate new sums
            # New sums are {s + x for s in sums_list} U {x}
            # Note: sums_list contains non-empty subset sums of current_set
            new_sums_list = [s + x for s in sums_list]
            new_sums_list.append(x)
            
            valid = True
            
            # Check interactions between new_sums and existing sums
            # 1. Does any s_new divide any s_old? (s_old % s_new == 0)
            # 2. Does any s_old divide any s_new? (s_new % s_old == 0)
            
            # We can check 2 by finding divisors of s_new in sums_set
            # We can check 1 by finding multiples of s_new in sums_set
            
            for s_new in new_sums_list:
                # Check divisors of s_new in sums_set
                # If d is in sums_set and d < s_new, then d divides s_new.
                # We iterate divisors of s_new
                limit = int(s_new ** 0.5)
                divisors = []
                for i in range(1, limit + 1):
                    if s_new % i == 0:
                        divisors.append(i)
                        if i != s_new // i:
                            divisors.append(s_new // i)
                
                for d in divisors:
                    if d in sums_set:
                        valid = False
                        break
                if not valid:
                    break
                
                # Check multiples of s_new in sums_set
                # If m is in sums_set and m > s_new, then s_new divides m.
                # m = k * s_new
                # Max sum is roughly 256 * 15 = 3840
                # So k goes up to 3840 // s_new
                if s_new > 0:
                    max_k = 3840 // s_new
                    for k in range(2, max_k + 1):
                        m = k * s_new
                        if m in sums_set:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Check interactions within new_sums_list
                new_sums_set = set(new_sums_list)
                for s_new in new_sums_list:
                    # Check divisors of s_new in new_sums_set (excluding itself)
                    limit = int(s_new ** 0.5)
                    divisors = []
                    for i in range(1, limit + 1):
                        if s_new % i == 0:
                            if i != s_new:
                                divisors.append(i)
                            if i != s_new // i and s_new // i != s_new:
                                divisors.append(s_new // i)
                    
                    for d in divisors:
                        if d in new_sums_set:
                            valid = False
                            break
                    if not valid:
                        break
                    
                    # Check multiples of s_new in new_sums_set (excluding itself)
                    if s_new > 0:
                        max_k = 3840 // s_new
                        for k in range(2, max_k + 1):
                            m = k * s_new
                            if m in new_sums_set:
                                valid = False
                                break
                    if not valid:
                        break
            
            if valid:
                current_set.append(x)
                sums_list.extend(new_sums_list)
                sums_set.update(new_sums_list)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:].copy()
            
    # Ensure return type is list of ints
    return best_set
