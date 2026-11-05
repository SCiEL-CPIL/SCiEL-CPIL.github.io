"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    N = 512
    
    # Find the number M <= 512 with the maximum number of divisors.
    # Any set of divisors of M is a valid subset because lcm(a,b) divides M,
    # and thus lcm(a,b) <= M <= 512.
    
    max_divs = 0
    best_m = 1
    best_divs_list = [1]
    
    # Precompute divisors for all numbers up to 512
    # Or just iterate and count.
    # Given N=512, simple iteration is sufficient.
    
    for m in range(1, N + 1):
        divs = []
        for i in range(1, int(m**0.5) + 1):
            if m % i == 0:
                divs.append(i)
                if i != m // i:
                    divs.append(m // i)
        
        if len(divs) > max_divs:
            max_divs = len(divs)
            best_m = m
            best_divs_list = divs
            
        # Check time
        if time.time() - start_time > deadline:
            break

    return best_divs_list
# EVOLVE-BLOCK-END
