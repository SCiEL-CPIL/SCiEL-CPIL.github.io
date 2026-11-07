import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    base = 210
    max_value = 10210
    max_size = 100
    
    # Sieve primes up to max_value
    sieve = [True] * (max_value + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(max_value**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, max_value + 1, i):
                sieve[j] = False
    primes = [i for i in range(max_value + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Number of uint64 words needed
    num_words = (num_primes + 63) // 64
    
    # Precompute square-free bitsets as numpy arrays
    sqf_np = [None] * (max_value + 1)
    for n in range(211, max_value + 1):
        temp = n
        bits = 0
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                bits |= (1 << prime_to_idx[p])
        if temp > 1:
            bits |= (1 << prime_to_idx[temp])
        
        # Convert int bits to numpy array
        arr = np.zeros(num_words, dtype=np.uint64)
        for w in range(num_words):
            mask = (bits >> (w * 64)) & 0xFFFFFFFFFFFFFFFF
            arr[w] = mask
        sqf_np[n] = arr
    
    # Target: square-free part of 210 = 2*3*5*7
    target_bits_int = 0
    for p in [2, 3, 5, 7]:
        target_bits_int |= (1 << prime_to_idx[p])
    
    target_np = np.zeros(num_words, dtype=np.uint64)
    for w in range(num_words):
        target_np[w] = (target_bits_int >> (w * 64)) & 0xFFFFFFFFFFFFFFFF
    
    # Maintain reduced target
    t_curr = target_np.copy()
    t_used = set() # Pivots of basis vectors used to reduce target
    
    # Basis: dict mapping pivot_bit -> numpy_array
    basis_vec = {}
    basis_combo = {} # pivot_bit -> int mask of original indices
    
    # Helper to find highest set bit in numpy array
    def get_highest_bit(vec):
        for i in range(len(vec)-1, -1, -1):
            val = vec[i]
            if val != 0:
                return i * 64 + (val.bit_length() - 1)
        return -1
    
    # Process numbers in increasing order
    for M in range(211, max_value + 1):
        if time.time() > deadline:
            break
        
        v = sqf_np[M].copy()
        idx = M - 211  # 0-based index
        v_combo = 1 << idx
        
        # Reduce v by existing basis
        while True:
            hb = get_highest_bit(v)
            if hb == -1:
                break # v is zero, dependent
            if hb in basis_vec:
                v ^= basis_vec[hb]
                v_combo ^= basis_combo[hb]
            else:
                # New pivot
                basis_vec[hb] = v
                basis_combo[hb] = v_combo
                
                # Update reduced target if it has this bit set
                word_idx = hb // 64
                bit_idx = hb % 64
                if (t_curr[word_idx] >> bit_idx) & 1:
                    t_curr ^= v
                    # Toggle hb in t_used
                    if hb in t_used:
                        t_used.remove(hb)
                    else:
                        t_used.add(hb)
                break
        
        # Check if target is now in span (t_curr is zero)
        if not t_curr.any():
            # Target is in span!
            sol_mask = 0
            for hb in t_used:
                sol_mask ^= basis_combo[hb]
            
            sol_count = bin(sol_mask).count('1')
            
            if 1 <= sol_count <= max_size:
                result = []
                for i in range(M - 210):
                    if (sol_mask >> i) & 1:
                        result.append(211 + i)
                
                # Verify
                prod = base
                for x in result:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    return list(result)
                else:
                    continue
    
    # Fallback
    return [840]
