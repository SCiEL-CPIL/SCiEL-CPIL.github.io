import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin before 60s kill
    
    MAX_VAL = 10210
    MIN_VAL = 211
    N = 210
    MAX_SIZE = 100
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Map primes to bit indices
    primes = [i for i in range(2, MAX_VAL + 1) if spf[i] == i]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 3. Compute bitmask V(n) for squarefree kernel
    V = [0] * (MAX_VAL + 1)
    for n in range(2, MAX_VAL + 1):
        x = n
        mask = 0
        while x > 1:
            p = spf[x]
            exp = 0
            while x % p == 0:
                x //= p
                exp += 1
            if exp % 2 == 1:
                mask ^= (1 << prime_to_idx[p])
        V[n] = mask
    
    target_mask = V[N]  # kernel of 210
    
    # 4. Gaussian elimination with RREF to find minimal max M
    # basis: dict pivot_bit -> (mask, rep_mask)
    # rep_mask bit i corresponds to number MIN_VAL + i
    basis = {}
    
    def find_solution(target_m, max_index):
        """Try to reduce target_m using basis. Returns rep_mask if successful."""
        current_rep = 0
        current_mask = target_m
        for p in sorted(basis.keys(), reverse=True):
            if current_mask & (1 << p):
                b_mask, b_rep = basis[p]
                current_mask ^= b_mask
                current_rep ^= b_rep
        if current_mask == 0:
            return current_rep
        return None
    
    for M in range(MIN_VAL, MAX_VAL + 1):
        if time.time() > deadline:
            # Return best known valid solution
            # 840 = 210*4, 210*840 = 210^2*4 = (420)^2
            return [840]
        
        mask = V[M]
        rep = (1 << (M - MIN_VAL))
        
        # Reduce mask using existing basis
        current_mask = mask
        current_rep = rep
        
        for p in sorted(basis.keys(), reverse=True):
            if current_mask & (1 << p):
                b_mask, b_rep = basis[p]
                current_mask ^= b_mask
                current_rep ^= b_rep
        
        if current_mask == 0:
            # Dependent, no new pivot
            # Check if target is solvable
            sol_rep = find_solution(target_mask, M - MIN_VAL)
            if sol_rep is not None:
                # Extract indices
                count = sol_rep.bit_count()
                if 0 < count <= MAX_SIZE:
                    selected = []
                    for i in range(M - MIN_VAL + 1):
                        if (sol_rep >> i) & 1:
                            selected.append(MIN_VAL + i)
                    # Validate
                    prod = 1
                    for x in selected:
                        prod *= x
                    if isqrt(prod * N) ** 2 == prod * N:
                        return selected
            continue
        
        # New pivot found
        pivot = current_mask.bit_length() - 1
        
        # Add to basis
        basis[pivot] = (current_mask, current_rep)
        
        # Eliminate this new pivot from all existing basis vectors to maintain RREF
        for p in list(basis.keys()):
            if p != pivot:
                if basis[p][0] & (1 << pivot):
                    b_mask, b_rep = basis[p]
                    basis[p] = (b_mask ^ current_mask, b_rep ^ current_rep)
        
        # Check if target is now solvable
        sol_rep = find_solution(target_mask, M - MIN_VAL)
        if sol_rep is not None:
            count = sol_rep.bit_count()
            if 0 < count <= MAX_SIZE:
                selected = []
                for i in range(M - MIN_VAL + 1):
                    if (sol_rep >> i) & 1:
                        selected.append(MIN_VAL + i)
                # Validate
                prod = 1
                for x in selected:
                    prod *= x
                if isqrt(prod * N) ** 2 == prod * N:
                    return selected
    
    # Fallback
    return [840]
