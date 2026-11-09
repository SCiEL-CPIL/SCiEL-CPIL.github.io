import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s buffer
    
    N = 1000000
    # 1. Compute Smallest Prime Factors (SPF)
    spf = list(range(N+1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i: # i is prime
            for j in range(i*i, N+1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute Powerful Part P(n) for all n using DP
    P = [1] * (N+1)
    # P[1] = 1 is already set
    for i in range(2, N+1):
        p = spf[i]
        k = i // p
        if k % p == 0:
            # Check if k is divisible by p^2 (i.e., k/p is divisible by p)
            if (k // p) % p == 0:
                P[i] = P[k] * p
            else:
                P[i] = P[k] * p * p
        else:
            P[i] = P[k]
            
        # Deadline check (optional, likely not needed, but good practice)
        if i % 100000 == 0:
            if time.time() > deadline:
                break
        
    # 3. Find max P(n)*P(n+1)
    max_val = 0
    best_n = 10000
    for n in range(10000, N):
        score = P[n] * P[n+1]
        if score > max_val:
            max_val = score
            best_n = n
            
    return best_n
