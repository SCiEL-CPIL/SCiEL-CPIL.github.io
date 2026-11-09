import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    # Seed solution
    best_set = [31, 32, 34]
    best_size = 3
    
    # Max possible size is 12
    for k in range(12, best_size, -1):
        attempts = 0
        found = False
        while time.time() < deadline and not found:
            attempts += 1
            
            # Sample k distinct integers from 2 to 256
            # Using sample is efficient
            try:
                candidates = random.sample(range(2, 257), k)
            except ValueError:
                break # Not enough elements (shouldn't happen)
            
            # Check element divisibility
            valid_elem = True
            for i in range(k):
                ci = candidates[i]
                for j in range(i + 1, k):
                    cj = candidates[j]
                    if cj % ci == 0:
                        valid_elem = False
                        break
                if not valid_elem:
                    break
            
            if not valid_elem:
                continue
                
            # Generate subset sums
            sums = {0}
            for x in candidates:
                new_sums = set()
                for s in sums:
                    new_sums.add(s + x)
                sums.update(new_sums)
            sums.discard(0)
            
            # Check distinctness
            if len(sums) != (1 << k) - 1:
                continue
                
            # Check divisibility of sums
            valid_div = True
            max_sum = max(sums)
            # Iterate through sums
            # To speed up, we can iterate over smaller sums first?
            # If s is small, it has more multiples.
            # If s is large, it has few.
            # But we just need to find ONE conflict.
            # Small sums are more likely to be divisors.
            sorted_sums = sorted(sums)
            for s in sorted_sums:
                m = 2 * s
                while m <= max_sum:
                    if m in sums:
                        valid_div = False
                        break
                    m += s
                if not valid_div:
                    break
            
            if valid_div:
                best_set = sorted(candidates)
                best_size = k
                found = True
                break
            
            # If we've tried too many times for this k, give up on this k
            if attempts > 5000:
                break

    return best_set
