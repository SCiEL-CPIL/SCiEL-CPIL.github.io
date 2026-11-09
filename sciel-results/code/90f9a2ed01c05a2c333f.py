import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s buffer
    
    N = 1000000
    
    # 1. Linear sieve to compute SPF and track prime powers
    spf = [0] * (N + 1)
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            spf[i] = i
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            spf[i * p] = p
            if i % p == 0:
                break
    
    # 2. Compute Powerful Part P(n) for all n using dynamic approach
    # For each n, we can use the relationship:
    # If n = p^k * m where p doesn't divide m, then:
    # - If k >= 2: P(n) = p^k * P(m)
    # - If k == 1: P(n) = P(m)
    
    P = [1] * (N + 1)
    # We need to track for each number: the smallest prime factor and its exponent
    # Let's compute: spf_exp[i] = exponent of spf[i] in i
    # And: spf_free[i] = i with all factors of spf[i] removed
    
    spf_exp = [0] * (N + 1)
    spf_free = [1] * (N + 1)
    
    for i in range(2, N + 1):
        p = spf[i]
        exp = 0
        temp = i
        while temp % p == 0:
            temp //= p
            exp += 1
        spf_exp[i] = exp
        spf_free[i] = temp
        
        if exp >= 2:
            P[i] = (p ** exp) * P[temp]
        else:
            P[i] = P[temp]
        
        if time.time() > deadline:
            break
    
    # 3. Find max P(n)*P(n+1)
    max_val = 0
    best_n = 10000
    for n in range(10000, N):
        score = P[n] * P[n + 1]
        if score > max_val:
            max_val = score
            best_n = n
            
    return best_n
