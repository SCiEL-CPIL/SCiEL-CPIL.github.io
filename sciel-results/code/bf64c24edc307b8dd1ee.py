import time
import array

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 10**7
    
    # Use unsigned long long (8 bytes) for phi values
    # phi(n) < n <= 10^7, so 32 bits is enough, but 64 is safe and standard for Q
    phi = array.array('Q', range(N + 1))
    
    # Sieve of Eratosthenes for phi
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    # Find the longest run of distinct phi values
    best_k = 1
    best_s = 1
    
    current_k = 1
    seen = set()
    seen.add(phi[1])
    
    for s in range(2, N + 1):
        val = phi[s]
        if val in seen:
            # Run broke
            if current_k > best_k:
                best_k = current_k
                best_s = s - current_k
            # Reset
            seen = {val}
            current_k = 1
        else:
            seen.add(val)
            current_k += 1
            
    # Check last run
    if current_k > best_k:
        best_k = current_k
        best_s = N - current_k + 1
        
    return {'start': best_s, 'length': best_k}
