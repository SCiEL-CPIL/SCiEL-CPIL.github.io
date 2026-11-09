import time
import sys
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 123457
    
    # List of small primes to test as the maximum prime factor P
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Function to generate a boolean array of p-smooth numbers <= limit
    # A number is p-smooth if its largest prime factor is <= p
    def generate_smooth_array(primes_list, limit):
        smooth = bytearray(limit + 1)
        smooth[1] = 1
        
        # Start with {1}, and for each prime, multiply all existing smooth numbers
        # by powers of that prime to generate new smooth numbers
        smooth_set = {1}
        for p in primes_list:
            # For each existing smooth number, generate p^k * x for k >= 1
            # Use a queue to handle newly added numbers in this prime's pass
            queue = deque()
            for x in list(smooth_set):
                val = x * p
                while val <= limit:
                    if val not in smooth_set:
                        smooth_set.add(val)
                        smooth[val] = 1
                        queue.append(val)
                    val *= p
            # Process newly added numbers (they can also be multiplied by p)
            while queue:
                x = queue.popleft()
                val = x * p
                while val <= limit:
                    if val not in smooth_set:
                        smooth_set.add(val)
                        smooth[val] = 1
                        queue.append(val)
                    val *= p
        
        return smooth

    for idx_p, p in enumerate(primes):
        if time.time() - start_time > DEADLINE:
            break
            
        # Generate all numbers whose prime factors are a subset of primes <= p
        current_primes = primes[:idx_p + 1]
        
        smooth_arr = generate_smooth_array(current_primes, N)
        
        # Check if there is a pair a, b where both are smooth and a + b = N
        # Linear scan for efficiency
        for a in range(1, N):
            if smooth_arr[a] and smooth_arr[N - a]:
                return [a, N - a]
        
        # If no pair found with max prime factor p, try next prime
    
    # Fallback: Sieve LPF and scan
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
            if best_max_pf == 2:
                break
                
    return [best_a, best_b]
