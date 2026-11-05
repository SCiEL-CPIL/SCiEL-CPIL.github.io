import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # Safety margin
    upper = 1000001  # n+1 can be 1000001
    
    # Sieve for Largest Prime Factor
    # Initialize with 0
    lpf = [0] * (upper + 1)
    
    # Sieve of Eratosthenes variant to find Largest Prime Factor
    for i in range(2, upper + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, upper + 1, i):
                lpf[j] = i
                
    # Find best n in [10000, 1000000]
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    # Linear scan to find the minimum of max(LPF(n), LPF(n+1))
    for n in range(10001, 1000001):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
            
    return best_n
# EVOLVE-BLOCK-END
