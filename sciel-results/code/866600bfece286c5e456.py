import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # 10s buffer for 90s limit
    
    # We need to check n up to 1000000, so n+1 up to 1000001
    LIMIT = 1000001
    
    # 1. Compute Smallest Prime Factors (SPF)
    spf = list(range(LIMIT + 1))
    for i in range(2, int(LIMIT**0.5) + 1):
        if spf[i] == i: # i is prime
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute Powerful Part P(n) for all n
    P = [1] * (LIMIT + 1)
    # P[1] = 1 by definition (empty product)
    # For n > 1, P(n) is product of p^e where e >= 2
    
    for i in range(2, LIMIT + 1):
        temp = i
        p_part = 1
        while temp > 1:
            p = spf[temp]
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp >= 2:
                p_part *= (p ** exp)
        P[i] = p_part
        
        # Deadline check
        if i % 50000 == 0:
            if time.time() > deadline:
                break

    # 3. Find max P(n)*P(n+1)
    max_val = 0
    best_n = 10000
    # n ranges from 10000 to 1000000
    for n in range(10000, 1000001):
        # Ensure we don't access out of bounds if deadline broke early
        if n + 1 > LIMIT:
            break
        score = P[n] * P[n+1]
        if score > max_val:
            max_val = score
            best_n = n
            
    return best_n
