import math

def construct():
    N_MAX = 1000000
    START = 10000
    
    # 1. Sieve of Smallest Prime Factors (SPF)
    spf = list(range(N_MAX + 1))
    for i in range(2, int(math.isqrt(N_MAX)) + 1):
        if spf[i] == i:
            for j in range(i * i, N_MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Identify Powerful Numbers
    # A number m is powerful if for all p|m, p^2|m
    powerful_set = set()
    for m in range(1, N_MAX + 1):
        temp = m
        is_pow = True
        while temp > 1:
            p = spf[temp]
            if temp % (p * p) != 0:
                is_pow = False
                break
            while temp % p == 0:
                temp //= p
        if is_pow:
            powerful_set.add(m)
            
    powerful_list = sorted(list(powerful_set), reverse=True)
    
    # 3. Helper to compute Powerful Part P(x)
    def get_powerful_part(x):
        if x == 1:
            return 1
        temp = x
        pp = 1
        while temp > 1:
            p = spf[temp]
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp >= 2:
                pp *= (p ** exp)
        return pp

    # 4. Helper to check if x is squarefree
    def is_squarefree(x):
        if x == 1:
            return True
        temp = x
        while temp > 1:
            p = spf[temp]
            if temp % (p * p) == 0:
                return False
            while temp % p == 0:
                temp //= p
        return True

    # 5. Search
    best_val = 0
    best_n = START
    
    for A in powerful_list:
        if A * N_MAX <= best_val:
            break
            
        k_min = (START + A - 1) // A
        k_max = N_MAX // A
        
        if k_min > k_max:
            continue
            
        for k in range(k_min, k_max + 1):
            if math.gcd(k, A) != 1:
                continue
            if not is_squarefree(k):
                continue
                
            n = A * k
            m = n + 1
            B = get_powerful_part(m)
            
            val = A * B
            if val > best_val:
                best_val = val
                best_n = n
                
    return best_n
