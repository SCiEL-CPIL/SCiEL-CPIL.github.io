import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 210
    MAX_VAL = 10210
    target = 210
    
    def calc_kernel(n):
        res = 1
        x = n
        p = 2
        while p * p <= x:
            if x % p == 0:
                exp = 0
                while x % p == 0:
                    x //= p
                    exp += 1
                if exp % 2 == 1:
                    res *= p
            p += 1
        if x > 1:
            res *= x
        return res

    # Fast path: Check for small subsets (1, 2, 3 elements)
    kernel_to_val = {}
    pair_to_vals = {}
    values = []
    kernels = []
    
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            return [840]

        k_M = calc_kernel(M)
        values.append(M)
        kernels.append(k_M)
        
        # Check 1-element
        if k_M == target:
            return [M]
        
        # Check 2-element
        needed_2 = target ^ k_M
        if needed_2 in kernel_to_val:
            other_val = kernel_to_val[needed_2]
            return [M, other_val]
        
        # Check 3-element
        needed_3 = target ^ k_M
        if needed_3 in pair_to_vals:
            v1, v2 = pair_to_vals[needed_3]
            return [M, v1, v2]
        
        # Update kernel_to_val
        if k_M not in kernel_to_val:
            kernel_to_val[k_M] = M
            
        # Update pair_to_vals
        for i in range(len(values) - 1):
            v = values[i]
            k_v = kernels[i]
            xor_val = k_M ^ k_v
            if xor_val not in pair_to_vals:
                pair_to_vals[xor_val] = (M, v)
                
        # If M is getting large and we haven't found a solution, 
        # we might need the linear basis fallback.
        # But for now, we continue.
        
        # Heuristic: If M > 1000, switch to linear basis to be safe?
        # Or just keep going. The loop is O(M^2) due to pair updates.
        # If M=1000, 10^6 ops. Fast.
        # If M=5000, 25*10^6 ops. Still fast in Python? Maybe 1-2s.
        # If M=10000, 10^8 ops. Might be 10-20s.
        # Given deadline 50s, it should be fine.

    # Fallback: Linear Basis Method (from current program)
    # This ensures we find a solution even if it requires > 3 elements
    # and was missed by the small subset search (unlikely for small M).
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(MAX_VAL)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    D = len(primes)
    
    # Target vector for 210
    t_vec = 0
    n = N
    for i, p in enumerate(primes):
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            if exp % 2 == 1:
                t_vec |= (1 << i)
                
    def get_vec(x):
        v = 0
        for i, p in enumerate(primes):
            if p * p > x:
                break
            if x % p == 0:
                exp = 0
                while x % p == 0:
                    x //= p
                    exp += 1
                if exp % 2 == 1:
                    v |= (1 << i)
        if x > 1:
            if x in prime_to_idx:
                v |= (1 << prime_to_idx[x])
        return v

    basis = [0] * (D + 1)
    origin = [set() for _ in range(D + 1)]
    null_space = []
    
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            return [840]
            
        v = get_vec(M)
        
        temp = v
        temp_origin = {M}
        
        for i in range(D - 1, -1, -1):
            if (temp >> i) & 1:
                if basis[i] == 0:
                    basis[i] = temp
                    origin[i] = temp_origin
                    break
                else:
                    temp ^= basis[i]
                    temp_origin ^= origin[i]
        
        if temp == 0:
            null_space.append(temp_origin)
        
        temp_t = t_vec
        indices = set()
        for i in range(D - 1, -1, -1):
            if (temp_t >> i) & 1:
                if basis[i] != 0:
                    temp_t ^= basis[i]
                    indices ^= origin[i]
                else:
                    break
        
        if temp_t == 0:
            if len(indices) > 100:
                changed = True
                while changed and len(indices) > 100:
                    changed = False
                    for ns in null_space:
                        if ns.issubset(indices):
                            indices -= ns
                            changed = True
                            break
                        else:
                            inter = len(ns & indices)
                            if 2 * inter > len(ns):
                                indices ^= ns
                                changed = True
                                break
            if len(indices) <= 100 and len(indices) >= 1:
                return list(indices)

    return [840]
