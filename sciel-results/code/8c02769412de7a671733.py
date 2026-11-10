import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    MAX_N = 10000000
    
    # We need to find a, b <= 10^7 such that sigma(a) = sigma(b) and |a/b - 3/2| is minimized.
    # a/b = 3/2 => 2a = 3b.
    # If we can find a pair where 2a = 3b exactly, the error is 0.
    # However, the task says "A raw value of exactly 0 cannot be scored and counts as invalid."
    # So we must find the pair with the smallest NON-ZERO error.
    # This implies we should look for pairs where 2a is close to 3b, but not equal.
    
    # Strategy:
    # 1. Efficiently compute sigma(n) for n up to MAX_N.
    #    A pure Python sieve for sigma is O(N log N) which might be too slow for 10^7 in Python within 85s.
    #    Let's estimate: 10^7 * (1/1 + 1/2 + ... + 1/10^7) ≈ 10^7 * ln(10^7) ≈ 1.6 * 10^8 operations.
    #    In Python, 1.6 * 10^8 operations is likely 10-20 seconds. This is feasible.
    #    We can optimize by only iterating over divisors up to sqrt(n)? No, sigma requires all divisors.
    #    The standard sieve approach:
    #    sigma = [0] * (MAX_N + 1)
    #    for i in range(1, MAX_N + 1):
    #        for j in range(i, MAX_N + 1, i):
    #            sigma[j] += i
    #    This is the standard way.
    
    # Let's try to optimize the inner loop or use a faster method if possible.
    # Since we cannot use NumPy for the sieve easily without a lot of overhead in the outer loop (10^7 iterations),
    # and the inner loop is vectorized in NumPy but the outer loop is Python, it might be similar speed.
    # Let's stick to pure Python but optimize the inner loop by local variable binding.
    
    sigma = [0] * (MAX_N + 1)
    
    # Optimization: local variables
    sig = sigma
    max_n = MAX_N
    
    for i in range(1, max_n + 1):
        for j in range(i, max_n + 1, i):
            sig[j] += i
            
    if time.time() > deadline - 10:
        # Fallback if sieve is too slow
        # We need a valid pair.
        # sigma(1)=1, sigma(2)=3, sigma(3)=4, sigma(4)=7, sigma(5)=6, sigma(6)=12, sigma(7)=8, sigma(8)=15, sigma(9)=13, sigma(10)=18
        # Pairs with same sigma:
        # sigma(6)=12, sigma(11)=12. a=6, b=11. |6/11 - 1.5| = |0.545 - 1.5| = 0.954
        # sigma(10)=18, sigma(14)=24? No.
        # Let's just return a known valid pair if we run out of time.
        # a=1, b=1 is invalid (a!=b not required? "a,b positive integers". Usually distinct? 
        # "Approximate a ratio". If a=b, ratio is 1. |1 - 1.5| = 0.5.
        # But sigma(1)=1, sigma(2)=3.
        # Let's check small numbers for a valid pair.
        # sigma(1)=1
        # sigma(2)=3
        # sigma(3)=4
        # sigma(4)=7
        # sigma(5)=6
        # sigma(6)=12
        # sigma(7)=8
        # sigma(8)=15
        # sigma(9)=13
        # sigma(10)=18
        # sigma(11)=12 -> Pair (6, 11) or (11, 6).
        # Ratio 11/6 = 1.833. |1.833 - 1.5| = 0.333.
        # Ratio 6/11 = 0.545. |0.545 - 1.5| = 0.954.
        # So [11, 6] is better.
        return [11, 6]

    # 2. Group numbers by sigma value
    fibers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        fibers[sigma[n]].append(n)
        
    if time.time() > deadline - 5:
        return [11, 6]

    # 3. Search for best pair
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5)
    
    # To find the pair with smallest |a/b - 1.5|, we want a/b to be close to 1.5.
    # This means a is close to 1.5 * b.
    # We iterate through fibers. For each fiber (group of numbers with same sigma),
    # we look for pairs (a, b) in the group such that a ≈ 1.5 b.
    
    # Optimization:
    # For a given fiber, sort the numbers.
    # For each b in the fiber, we want a in the fiber such that a is close to 1.5*b.
    # We can use binary search to find the closest a.
    
    # To speed up, we can skip fibers that are too small?
    # Or we can limit the search to fibers that contain numbers in a range that allows a/b ≈ 1.5.
    # If max(fiber) / min(fiber) < 1.5 * (1 - epsilon) or > 1.5 * (1 + epsilon), 
    # it might still contain a pair, but the ratio of any two elements is bounded.
    # Actually, if we have a and b in the same fiber, a/b can be anything.
    # But we specifically want a/b ≈ 1.5.
    # So we only care about pairs where a is roughly 1.5 times b.
    # This implies b should be at least 1/1.5 of a.
    
    # Let's iterate through the fibers.
    # For each fiber, if len < 2, skip.
    # Sort the fiber.
    # For each b in the fiber:
    #   target = 1.5 * b
    #   Find a in fiber closest to target.
    #   Check error.
    
    # To make this faster, we can pre-sort the fibers?
    # The number of fibers is large, but most are small.
    
    # Let's refine the search:
    # We want to minimize |a/b - 1.5| = |2a - 3b| / (2b).
    # Since a, b are integers, |2a - 3b| is an integer >= 1 (if not 0).
    # So the error is at least 1/(2b).
    # To get a very small error, we need a large b.
    # So we should prioritize fibers that contain large numbers.
    
    # Let's sort fibers by their maximum element, descending?
    # Or just iterate and keep the best.
    # The current approach iterates all fibers.
    
    # Optimization: 
    # If we find a pair with error E, we only need to consider pairs where 1/(2b) < E, i.e., b > 1/(2E).
    # Initially E ≈ 0.333. 1/(2*0.333) ≈ 1.5. So b > 1.5 is always true for b>=2.
    # As E decreases, we can skip small b.
    
    # Let's implement the search.
    
    # To avoid checking every pair in large fibers (though most are small), 
    # we use binary search.
    
    import bisect
    
    for s_val, nums in fibers.items():
        if time.time() > deadline - 2:
            break
            
        if len(nums) < 2:
            continue
            
        nums.sort()
        
        # For each b in nums, find a closest to 1.5 * b
        for b in nums:
            if time.time() > deadline - 1:
                break
                
            # Pruning: if 1/(2*b) >= best_err, then even the best possible integer difference (1) 
            # gives an error >= best_err. So we can skip this b?
            # Wait, error = |2a - 3b| / (2b).
            # Min non-zero |2a - 3b| is 1 (if 2a-3b is odd) or 2 (if even).
            # Actually 2a - 3b can be any integer.
            # So min error for this b is 1/(2b).
            if 1.0 / (2.0 * b) >= best_err:
                continue
                
            target_a = 1.5 * b
            pos = bisect.bisect_left(nums, target_a)
            
            candidates = []
            if pos > 0:
                candidates.append(nums[pos-1])
            if pos < len(nums):
                candidates.append(nums[pos])
            
            for a in candidates:
                if a == b:
                    continue
                    
                # Calculate error exactly?
                # err = abs(a/b - 1.5)
                # To avoid float precision issues in comparison, we can compare fractions.
                # |a/b - 3/2| = |2a - 3b| / (2b)
                # We want to minimize this.
                # Current best is best_err.
                # Compare |2a - 3b| / (2b) < best_err
                # |2a - 3b| < 2b * best_err
                
                num = abs(2*a - 3*b)
                den = 2*b
                
                if num == 0:
                    # Invalid
                    continue
                
                # Compare num/den < best_err
                # num < den * best_err
                if num < den * best_err:
                    best_err = num / den
                    best_a = a
                    best_b = b

    return [int(best_a), int(best_b)]
