import random
import time

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    # Try sizes from 15 down to 1
    for k in range(15, 0, -1):
        if time.time() > deadline:
            break
            
        # Max sum estimate for limit
        max_val = 256
        # We will pick random distinct ints from 1..256
        
        # To increase chance of validity, we might prefer larger numbers?
        # Or just uniform random.
        
        attempts = 5000
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            # Generate random set
            # Using a range of 1..256
            # To avoid small numbers which cause divisibility issues,
            # maybe bias towards larger numbers?
            # Let's try uniform first.
            cand = random.sample(range(1, 257), k)
            
            # Quick check: no element divides another
            valid = True
            for i in range(k):
                for j in range(k):
                    if i == j: continue
                    if cand[j] % cand[i] == 0:
                        valid = False
                        break
                if not valid: break
            if not valid:
                continue
            
            # Compute subset sums
            sums = {0}
            for x in cand:
                new_sums = []
                for s in sums:
                    new_sums.append(s + x)
                # Add to set
                for ns in new_sums:
                    if ns in sums:
                        # Duplicate sum found
                        valid = False
                        break
                    sums.add(ns)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Remove 0 for divisibility check?
            # "no two distinct resulting values may divide one another"
            # 0 is not a value? "NONEMPTY subset sum". So 0 is not included.
            non_zero_sums = [s for s in sums if s > 0]
            
            # Check divisibility
            # Use a set for fast lookup
            sum_set = set(non_zero_sums)
            max_sum = max(non_zero_sums)
            
            div_valid = True
            for s in non_zero_sums:
                # Check multiples
                # If s divides m, and m is in sum_set, and m != s
                # We can just iterate multiples
                m = 2 * s
                while m <= max_sum:
                    if m in sum_set:
                        div_valid = False
                        break
                    m += s
                if not div_valid:
                    break
            
            if div_valid:
                return cand
                
    # Fallback: return a known small valid set
    # {2, 3} is valid?
    # Sums: 2, 3, 5.
    # 2|? No. 3|? No. 5|? No.
    return [2, 3]
