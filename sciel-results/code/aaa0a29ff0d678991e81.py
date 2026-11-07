import time

def construct():
    start_time = time.time()
    
    # Primes up to 10210
    # We only need primes up to 240 for the search, but let's be general
    # Actually, for the hardcoded solution 224, 240, we just need to verify it.
    # For the search, we look up to 239.
    
    # Simple sieve for primes up to 10210
    limit = 10210
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Map prime to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def get_kernel_vec(n):
        """Returns an integer representing the squarefree kernel as a bit vector."""
        x = n
        vec = 0
        for i, p in enumerate(primes):
            if p * p > x:
                break
            if x % p == 0:
                count = 0
                while x % p == 0:
                    x //= p
                    count += 1
                if count % 2 == 1:
                    vec |= (1 << i)
        if x > 1:
            # x is a prime
            if x in prime_to_idx:
                vec |= (1 << prime_to_idx[x])
            else:
                # This shouldn't happen if limit is correct
                pass
        return vec

    target_vec = get_kernel_vec(210)
    
    # Search for solution with max < 240
    # Numbers: 211 to 239
    candidates = list(range(211, 240))
    vecs = [get_kernel_vec(x) for x in candidates]
    
    # Check size 1
    for i in range(len(candidates)):
        if vecs[i] == target_vec:
            return [candidates[i]]
            
    # Check size 2
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if (vecs[i] ^ vecs[j]) == target_vec:
                return [candidates[i], candidates[j]]
                
    # Check size 3
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            for k in range(j + 1, len(candidates)):
                if (vecs[i] ^ vecs[j] ^ vecs[k]) == target_vec:
                    return [candidates[i], candidates[j], candidates[k]]
                    
    # Check size 4
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            for k in range(j + 1, len(candidates)):
                for l in range(k + 1, len(candidates)):
                    if (vecs[i] ^ vecs[j] ^ vecs[k] ^ vecs[l]) == target_vec:
                        return [candidates[i], candidates[j], candidates[k], candidates[l]]
                        
    # Check size 5
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            for k in range(j + 1, len(candidates)):
                for l in range(k + 1, len(candidates)):
                    for m in range(l + 1, len(candidates)):
                        if (vecs[i] ^ vecs[j] ^ vecs[k] ^ vecs[l] ^ vecs[m]) == target_vec:
                            return [candidates[i], candidates[j], candidates[k], candidates[l], candidates[m]]
                            
    # If no small solution found, return the known best solution
    # 224 and 240
    # Verify 224, 240
    # sf(224) = 14, sf(240) = 15. 14 * 15 = 210. Correct.
    return [224, 240]
