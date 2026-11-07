import math
from collections import defaultdict

def construct():
    primes = [2, 3, 5, 7, 11, 13]
    
    # Precompute V(p, e) = p^e * sigma(p^e)
    V_terms = []
    for p in primes:
        e = 1
        pe = p
        sigma_pe = 1
        while True:
            sigma_pe = sigma_pe * p + 1
            val = pe * sigma_pe
            if val > 10**18:
                break
            V_terms.append((p, e, val))
            pe *= p
            e += 1
            
    best_M = 1
    best_k_list = [1]
    
    base_primes = primes
    max_exp = 10
    
    def solve_for_k(M, sqrt_M):
        p_to_es = defaultdict(list)
        for p, e, val in V_terms:
            if M % val == 0:
                p_to_es[p].append((e, val))
                
        res = []
        primes_in_k = list(p_to_es.keys())
        
        def dfs_k(idx, remaining_M, current_k):
            if idx == len(primes_in_k):
                if remaining_M == 1:
                    if current_k <= sqrt_M:
                        res.append(current_k)
                return
                
            p = primes_in_k[idx]
            # Option: e=0
            dfs_k(idx + 1, remaining_M, current_k)
            
            # Option: e in p_to_es[p]
            for e, val in p_to_es[p]:
                if remaining_M % val == 0:
                    new_k = current_k * (p ** e)
                    if new_k <= sqrt_M:
                        dfs_k(idx + 1, remaining_M // val, new_k)
        
        dfs_k(0, M, 1)
        return res

    def generate_M(idx, current_M):
        nonlocal best_M, best_k_list
        
        if idx == len(base_primes):
            M = current_M
            if M > 10**18:
                return
            
            sqrt_M = math.isqrt(M)
            ks = solve_for_k(M, sqrt_M)
            if len(ks) > len(best_k_list):
                best_M = M
                best_k_list = ks
            return
        
        p = base_primes[idx]
        # Try exponents 0 to max_exp
        temp_M = current_M
        for e in range(max_exp + 1):
            if temp_M > 10**18:
                break
            generate_M(idx + 1, temp_M)
            if temp_M > 10**18 // p:
                break
            temp_M *= p

    generate_M(0, 1)
    
    return {'M': int(best_M), 'k': [int(k) for k in best_k_list]}
