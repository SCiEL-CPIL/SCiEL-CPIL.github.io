# EVOLVE-BLOCK-START
import time
from math import isqrt
import random

def _sieve_sfp(limit):
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    sfp = [0] * (limit + 1)
    sfp[1] = 1
    for x in range(2, limit + 1):
        v = x
        r = 1
        while v > 1:
            p = spf[v]
            cnt = 0
            while v % p == 0:
                v //= p
                cnt ^= 1
            if cnt:
                r *= p
        sfp[x] = r
    return sfp

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    t0 = time.time()
    deadline = t0 + 80.0
    LO = 210
    MAXV = 10210
    
    sfp = _sieve_sfp(MAXV)
    
    # Primes and index mapping
    is_prime = [True] * (MAXV + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, MAXV + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, MAXV + 1, i):
                is_prime[j] = False
    prime_index = {p: i for i, p in enumerate(primes)}
    
    def get_vector(x):
        v = 0
        sf = sfp[x]
        p = 2
        while p * p <= sf:
            if sf % p == 0:
                idx = prime_index[p]
                v ^= (1 << idx)
            p += 1 if p == 2 else 2
        if sf > 1:
            idx = prime_index[sf]
            v ^= (1 << idx)
        return v
        
    target_vec = get_vector(30)
    
    vecs = [0] * (MAXV + 1)
    for x in range(211, MAXV + 1):
        vecs[x] = get_vector(x)
        
    for M in range(310, MAXV + 1):
        if time.time() > deadline:
            break
            
        K = M - 310
        sum_C = 0
        for x in range(211, M + 1):
            sum_C ^= vecs[x]
        T = sum_C ^ target_vec
        
        # Build map for current range
        vec_to_indices = {}
        for x in range(211, M + 1):
            vec_to_indices.setdefault(vecs[x], []).append(x)
            
        found_U = None
        
        if K == 0:
            if T == 0:
                found_U = []
                
        elif K == 1:
            if T in vec_to_indices:
                found_U = [vec_to_indices[T][0]]
                
        elif K == 2:
            # Find x, y distinct such that v[x]^v[y] = T
            # v[y] = T ^ v[x]
            for x in range(211, M + 1):
                needed = T ^ vecs[x]
                if needed in vec_to_indices:
                    for y in vec_to_indices[needed]:
                        if y != x:
                            found_U = [x, y]
                            break
                if found_U:
                    break
                    
        elif K == 3:
            # Find x, y, z distinct such that v[x]^v[y]^v[z] = T
            # v[z] = T ^ v[x] ^ v[y]
            for x in range(211, M + 1):
                for y in range(x + 1, M + 1):
                    needed = T ^ vecs[x] ^ vecs[y]
                    if needed in vec_to_indices:
                        for z in vec_to_indices[needed]:
                            if z != x and z != y:
                                found_U = [x, y, z]
                                break
                    if found_U:
                        break
                if found_U:
                    break
                    
        elif K == 4:
            # Find x, y, z, w distinct such that v[x]^v[y]^v[z]^v[w] = T
            # v[z]^v[w] = T ^ v[x] ^ v[y]
            # Precompute all pair XORs?
            # Or iterate pairs (x,y) and (z,w)?
            # Let's iterate x, y and look for a pair (z,w) with XOR = T^v[x]^v[y]
            # This is O(N^2 * N^2) = O(N^4) which is 10^8 for N=100. Too slow.
            # Better: Store all pairs (z,w) and their XOR in a dict.
            pair_map = {}
            for z in range(211, M + 1):
                for w in range(z + 1, M + 1):
                    xor_zw = vecs[z] ^ vecs[w]
                    if xor_zw not in pair_map:
                        pair_map[xor_zw] = (z, w)
            
            for x in range(211, M + 1):
                for y in range(x + 1, M + 1):
                    needed = T ^ vecs[x] ^ vecs[y]
                    if needed in pair_map:
                        z, w = pair_map[needed]
                        if z != x and z != y and w != x and w != y:
                            found_U = [x, y, z, w]
                            break
                if found_U:
                    break
                    
        else:
            # K > 4
            # Random sampling
            candidates = list(range(211, M + 1))
            for _ in range(200):
                if len(candidates) < K:
                    break
                subset = random.sample(candidates, K - 1)
                xor_subset = 0
                for x in subset:
                    xor_subset ^= vecs[x]
                needed = T ^ xor_subset
                subset_set = set(subset)
                for x in candidates:
                    if x not in subset_set and vecs[x] == needed:
                        found_U = subset + [x]
                        break
                if found_U:
                    break
        
        if found_U:
            U_set = set(found_U)
            selected = [x for x in range(211, M + 1) if x not in U_set]
            prod = 210
            for x in selected:
                prod *= x
            if _is_square(prod):
                return [int(v) for v in selected]
                
    return [840]
# EVOLVE-BLOCK-END
