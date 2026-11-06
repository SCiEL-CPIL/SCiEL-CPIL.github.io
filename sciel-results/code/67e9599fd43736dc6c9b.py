import time
import math

def construct():
    deadline = time.time() + 80
    target = 123457
    
    # Direct approach: Check for small smoothness
    # Primes to check sequentially
    # We want to find min k such that N = a+b, P(a)<=k, P(b)<=k.
    # This is equivalent to N = a+b where a,b are k-smooth.
    
    # We can check k=2,3,5,7,11,13...
    # We stop when we find a solution.
    
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
    
    # Helper to generate k-smooth numbers
    def get_smooth(k_index):
        # k_index is number of primes used
        # Primes: [2, 3, 5, 7, ...]
        if k_index == 0:
            return {1}
        
        primes = primes_list[:k_index]
        # Use a dictionary or set of generated numbers
        # For k up to 11 (5 primes), simple recursion works.
        
        numbers = {1}
        # We can update the set iteratively
        # Start with {1}
        # For each new prime p:
        #   new_numbers = {n * p^i for n in numbers for i >= 1 if n * p^i <= limit}
        #   numbers = numbers | new_numbers
        
        limit = target
        # Initialize with 1
        current_set = {1}
        for i in range(k_index):
            p = primes_list[i]
            new_adds = set()
            for n in list(current_set):
                val = n
                while val <= limit:
                    val *= p
                    if val <= limit:
                        new_adds.add(val)
                    else:
                        break
            current_set.update(new_adds)
        return current_set

    best_pair = [1, target - 1]
    
    # Check k=1 to len(primes_list)
    for k in range(1, len(primes_list) + 1):
        if time.time() > deadline - 1:
            break
            
        smooth_set = get_smooth(k)
        
        # Check if target is a sum of two elements in smooth_set
        # one must be even, one odd?
        # Just check all x in smooth_set
        # if target - x in smooth_set
        
        found = False
        for x in smooth_set:
            y = target - x
            if y in smooth_set:
                # Found
                # P(x) <= prime_list[k-1], P(y) <= prime_list[k-1]
                # Max prime factor is at most primes_list[k-1]
                # But actually, we just need to return the pair.
                # The evaluator calculates P(a) and P(b).
                # If we return (x,y), P(x) and P(y) will be calculated.
                # Since x,y are k-smooth, P(x), P(y) <= primes_list[k-1].
                # This will yield a score based on this value.
                # We assume smaller k is better.
                return [x, y]
                
        # If not found, continue to next k
        # We don't have a fallback pair yet other than 1, target-1
        # But we know a solution exists (e.g. 1, 123456).
        # However, 123456 is not 1-smooth, not 2-smooth, etc.
        # So we will eventually hit a k where 123456 is smooth?
        # 123456 = 2^6 * 3 * 643. 643 is prime.
        # So 123456 is 643-smooth.
        # So for k such that prime_list[k-1] >= 643, the set will contain 123456.
        # And 1 is in the set.
        # So we will always find a solution eventually.
        
        # But we want to find the MINIMUM k.
        # The loop increases k. So the first one found is the minimum.
        
    # Fallback if loop finishes without return (should not happen for k up to 30)
    # Do a full scan to be safe?
    # Or just return the best found so far?
    # Since we return immediately on first find, this block is only reached if
    # no solution found in primes_list.
    
    # Given primes_list goes up to 31, and 123456 is 643-smooth,
    # we might exit the loop without returning if 123456 is not smooth
    # with primes <= 31.
    # 643 is > 31. So 123456 is NOT in smooth_set for k where prime < 643.
    # So if the minimum k is 11, we return in k=11.
    # If the minimum k was 643, we would need to go up to 643.
    # But we suspect min is 11.
    
    # If we exit the loop, it means no solution found with primes <= 31.
    # This would imply min k >= 37 (next prime).
    # To be safe, let's add a full scan fallback.
    
    # Full scan (as in original, but we can skip if we are confident)
    # For robustness, include the scan.
    
    max_n = target - 1
    lpf = [0] * (max_n + 1)
    lpf[1] = 1
    for i in range(2, max_n + 1):
        if lpf[i] == 0:
            for j in range(i, max_n + 1, i):
                if lpf[j] == 0 or i > lpf[j]:
                    lpf[j] = i

    best_max = float('inf')
    best_a = 1
    
    for a in range(1, target):
        b = target - a
        cur_max = max(lpf[a], lpf[b])
        if cur_max < best_max:
            best_max = cur_max
            best_a = a
            if best_max <= 3:
                break

    return [best_a, target - best_a]
