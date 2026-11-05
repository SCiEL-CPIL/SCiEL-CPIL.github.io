import time

def construct():
    start_time = time.time()
    LIMIT = 90  # seconds (evaluator kills at 90s)
    MARGIN = 5  # seconds safety margin
    
    N_MAX = 1000001  # We need P up to 1000001 to evaluate n=1000000
    START = 10000
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    # Using a list of ints for speed; this is the standard fast SPF sieve
    spf = list(range(N_MAX + 1))
    for i in range(2, int(N_MAX**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N_MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute Powerful Part P[k]
    # P[i] is the product of p^e for all prime powers p^e || i where e >= 2
    # Recurrence:
    # Let p = spf[i], x = i // p
    # If x % p != 0: i has p^1, so P[i] = P[x]
    # If x % p == 0:
    #   Let y = x // p
    #   If y % p != 0: i has p^2, so P[i] = P[x] * p^2 (since P[x] has no p factor)
    #   If y % p == 0: i has p^3+, so P[i] = P[x] * p (since P[x] already has p^(e-1) where e-1>=2)
    #     Wait, if i = p^e * m, e>=3. x = p^(e-1) * m.
    #     If e-1 >= 2, P[x] = P[m] * p^(e-1).
    #     P[i] should be P[m] * p^e = P[x] * p.
    #     If e-1 == 1 (i.e., e=2), x = p*m. P[x] = P[m].
    #     P[i] = P[m] * p^2 = P[x] * p^2.
    
    P = [1] * (N_MAX + 1)
    P[1] = 1
    
    for i in range(2, N_MAX + 1):
        p = spf[i]
        x = i // p
        if x % p == 0:
            y = x // p
            if y % p == 0:
                # i has p^3 or higher
                P[i] = P[x] * p
            else:
                # i has exactly p^2
                P[i] = P[x] * p * p
        else:
            # i has exactly p^1
            P[i] = P[x]
            
    # Check time
    if time.time() - start_time > LIMIT - MARGIN:
        return 10000

    # 3. Find n that maximizes P[n] * P[n+1]
    # We want to maximize the product of the powerful parts of consecutive integers.
    # P values can be large, but Python handles big integers natively.
    # However, for speed, we can try to use numpy if the values fit in int64, 
    # but P values can exceed 2^63 easily (e.g., 2^20 * 3^10 * ...).
    # Let's check the magnitude.
    # Max P[i] for i <= 10^6.
    # 10^6 = 2^6 * 5^6. P[10^6] = 2^6 * 5^6 = 10^6.
    # 999983 is prime. P=1.
    # What about 2^19 = 524288. P = 524288.
    # 2^19 * 3 = 1572864 > 10^6.
    # 2^10 * 3^6 = 1024 * 729 = 746496. P = 746496.
    # 2^6 * 3^6 * 5^2 = 64 * 729 * 25 = 1,166,400 > 10^6.
    # 2^5 * 3^6 * 5^2 = 32 * 729 * 25 = 583,200. P = 583,200.
    # 2^4 * 3^5 * 5^2 * 7 = 16 * 243 * 25 * 7 = 680,400. P = 16*243*25*7 = 680,400? No, 7^1 is not powerful. P = 16*243*25 = 97,200.
    # The maximum powerful part for n <= 10^6 is likely around 10^6.
    # Product of two such parts can be up to 10^12, which fits in int64 (9.2 * 10^18).
    # So we can safely use numpy int64 for the product calculation.
    
    try:
        import numpy as np
        # Convert P to numpy array
        # We need indices START to N_MAX (which is 1000000) for n, and START+1 to N_MAX+1 for n+1.
        # P has length N_MAX+1 = 1000002. Indices 0..1000001.
        # n ranges from 10000 to 1000000.
        # P[n] is P[10000..1000000]
        # P[n+1] is P[10001..1000001]
        
        # Note: P[1000001] is computed.
        
        P_arr = np.array(P, dtype=np.int64)
        
        # Slice the relevant parts
        # n is from START to 1000000 inclusive.
        # Number of elements: 1000000 - 10000 + 1 = 990001
        left = P_arr[START:1000001]
        right = P_arr[START+1:1000002]
        
        products = left * right
        
        max_idx = np.argmax(products)
        best_n = int(START + max_idx)
        
    except ImportError:
        # Fallback to pure Python if numpy is not available
        best_n = START
        best_score = -1
        for n in range(START, 1000001):
            score = P[n] * P[n + 1]
            if score > best_score:
                best_score = score
                best_n = n

    return best_n
