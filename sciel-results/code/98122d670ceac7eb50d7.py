import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # We will try different K_max values
    # Start with a smaller one to get a quick result
    K_max = 10**6
    
    while K_max <= 10**7 and time.time() < deadline - 10:
        # Compute sigma up to K_max
        # Using SPF
        spf = list(range(K_max + 1))
        for i in range(2, int(math.isqrt(K_max)) + 1):
            if spf[i] == i:
                for j in range(i*i, K_max + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        sigma = [0] * (K_max + 1)
        sigma[1] = 1
        for i in range(2, K_max + 1):
            # Factorize i using spf
            n = i
            prod = 1
            while n > 1:
                p = spf[n]
                exp = 0
                while n % p == 0:
                    n //= p
                    exp += 1
                # sigma(p^exp) = (p^(exp+1)-1)/(p-1)
                term = 0
                pow_p = 1
                for _ in range(exp + 1):
                    term += pow_p
                    pow_p *= p
                prod *= term
            sigma[i] = prod
            
        # Find best M
        counts = {}
        solutions = {}
        for k in range(1, K_max + 1):
            M = k * sigma[k]
            if M > 10**18:
                continue
            if M not in counts:
                counts[M] = 0
                solutions[M] = []
            counts[M] += 1
            solutions[M].append(k)
            
        # Update best
        for M, cnt in counts.items():
            if cnt > best_count:
                best_count = cnt
                best_M = M
                best_k = solutions[M]
                
        # Increase K_max
        K_max *= 2
        
    # Final check and return
    return {'M': int(best_M), 'k': [int(k) for k in best_k]}
