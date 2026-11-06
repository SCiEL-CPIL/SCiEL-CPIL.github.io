import time
from math import isqrt

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0

    N_LIMIT = 10000
    
    # Sieve of Eratosthenes
    is_prime = [True] * (N_LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N_LIMIT) + 1):
        if is_prime[i]:
            for j in range(i*i, N_LIMIT + 1, i):
                is_prime[j] = False
                
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n_primes = len(primes)
    
    # DP tables
    # DP[j][k] = max length of sequence ending with primes[j], primes[k]
    # Prev[j][k] = index i such that sequence ended with primes[i], primes[j]
    # We only need j < k.
    # To save memory and time, we can use lists of lists.
    DP = [[0] * n_primes for _ in range(n_primes)]
    Prev = [[-1] * n_primes for _ in range(n_primes)]
    
    max_len = 2
    best_j = 0
    best_k = 1
    
    # Iterate over the second-to-last prime index j
    for j in range(n_primes - 1):
        # For a fixed j, we will compute DP[j][k] for all k > j
        # We maintain current_max and ptr for the inner loop
        current_max = 0
        ptr = j - 1
        
        # k goes from j+1 to n_primes-1
        for k in range(j + 1, n_primes):
            # Check time
            if time.time() > deadline:
                # Return best found so far
                break
            
            target = 2 * primes[j] - primes[k]
            
            # Add new valid i's to current_max
            # Valid i must satisfy primes[i] >= target
            # As k increases, target decreases, so more i's become valid
            # ptr moves from j-1 downwards
            while ptr >= 0 and primes[ptr] >= target:
                if DP[ptr][j] > current_max:
                    current_max = DP[ptr][j]
                    # We need to track which ptr gave this max
                    # But current_max is just a value.
                    # We need to store the best predecessor for DP[j][k]
                    # Actually, we can just store the best i in a variable best_i_for_j
                    # But current_max updates as we go.
                    # We need to know WHICH i gave the max for the CURRENT k.
                    # Wait, the max is over ALL valid i's.
                    # So we need to store the index i that achieves current_max.
                    pass
                ptr -= 1
        
        # The above logic is slightly flawed for tracking Prev.
        # We need to know which i gave the max for each k.
        # Let's restructure.
        
    # Revised Logic for DP loop
    # We need to be careful with Prev tracking.
    
    DP = [[0] * n_primes for _ in range(n_primes)]
    Prev = [[-1] * n_primes for _ in range(n_primes)]
    
    max_len = 2
    best_end_j = 0
    best_end_k = 1
    
    for j in range(n_primes - 1):
        current_max_val = 0
        current_max_idx = -1
        ptr = j - 1
        
        for k in range(j + 1, n_primes):
            if time.time() > deadline:
                # Finish current j loop or break out?
                # Breaking out of inner loop, then outer loop should also break
                # For simplicity, just return best found
                pass 
            
            target = 2 * primes[j] - primes[k]
            
            # Update current_max with new valid i's
            while ptr >= 0 and primes[ptr] >= target:
                if DP[ptr][j] > current_max_val:
                    current_max_val = DP[ptr][j]
                    current_max_idx = ptr
                ptr -= 1
            
            if current_max_val > 0:
                DP[j][k] = current_max_val + 1
                Prev[j][k] = current_max_idx
            else:
                DP[j][k] = 2
                Prev[j][k] = -1
            
            if DP[j][k] > max_len:
                max_len = DP[j][k]
                best_end_j = j
                best_end_k = k
                
            if time.time() > deadline:
                # We can break early if we are confident, but let's just let it run
                # The loop is fast.
                pass

    # Backtrack to get the sequence
    seq = []
    if max_len == 2:
        # Find any pair
        # best_end_j, best_end_k should be set
        seq = [primes[best_end_j], primes[best_end_k]]
    else:
        j = best_end_j
        k = best_end_k
        seq.append(primes[k])
        seq.append(primes[j])
        
        while Prev[j][k] != -1:
            i = Prev[j][k]
            seq.append(primes[i])
            # The previous pair was (i, j)
            # We need to find the predecessor of (i, j)
            # Which is Prev[i][j]
            # So we swap j, k to i, j
            k = j
            j = i
            
        # seq is in reverse order (last to first)
        seq.reverse()
        
    return seq

# Run construct to verify
# print(construct())
