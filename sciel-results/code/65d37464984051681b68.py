"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 123457
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def generate_smooth_numbers(N, primes_subset):
        """Generate all numbers <= N with prime factors in primes_subset."""
        smooth = set([1])
        queue = [1]
        idx = 0
        while idx < len(queue):
            current = queue[idx]
            idx += 1
            for p in primes_subset:
                nxt = current * p
                if nxt <= N and nxt not in smooth:
                    smooth.add(nxt)
                    queue.append(nxt)
        return smooth

    for i in range(len(primes)):
        if time.time() > DEADLINE:
            break
            
        current_primes = primes[:i+1]
        smooth_numbers = generate_smooth_numbers(N, current_primes)
        
        # Check if any x in smooth_numbers has N-x also in smooth_numbers
        for x in smooth_numbers:
            y = N - x
            if 1 <= y <= N and y in smooth_numbers:
                return [x, y]
        
    # Fallback
    return [1, N-1]

# EVOLVE-BLOCK-END
