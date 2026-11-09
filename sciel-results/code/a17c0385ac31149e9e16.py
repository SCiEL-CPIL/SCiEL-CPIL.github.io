import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 60s
    
    N = 1000000
    MAX_VAL = N + 1  # Need P up to N+1
    
    # 1. Compute Smallest Prime Factor (SPF)
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
                
    # 2. Compute Powerful Part for all numbers
    P = [1] * (MAX_VAL + 1)
    for i in range(2, MAX_VAL + 1):
        x = i
        res = 1
        while x > 1:
            p = spf[x]
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            if count >= 2:
                res *= (p ** count)
        P[i] = res
        
        # Time check (rarely triggered)
        if i % 100000 == 0 and time.time() > deadline:
            # If we run out of time, we can't guarantee correctness for higher numbers
            # But we can return a valid candidate from the computed part
            # However, the search for max is in the next loop.
            # If we break here, P is incomplete.
            # Given N=1e6, this loop is fast (~1-2s).
            pass

    # 3. Find max product
    best_n = 10000
    best_score = 0
    for n in range(10000, N + 1):
        score = P[n] * P[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
            
        if time.time() > deadline:
            break

    return best_n
