import time

def construct():
    start_time = time.time()
    LIMIT = 10000
    
    # 1. Generate primes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Precompute next_prime_ge[x]
    next_prime_ge = [0] * (LIMIT + 2)
    next_p = -1
    for i in range(LIMIT, -1, -1):
        if is_prime[i]:
            next_p = i
        next_prime_ge[i] = next_p
        
    n = len(primes)
    best_len = 1
    best_seq = [primes[0]] if primes else []
    
    # 3. Iterate over starting pairs
    for i in range(n):
        p1 = primes[i]
        # If even with gap 1 we can't beat best_len, skip this p1?
        # Max possible length from p1 with min gap 1 is (LIMIT - p1) + 1.
        # But gaps must be non-decreasing primes differences.
        # Simple bound: if (LIMIT - p1) < best_len, we can't even fit best_len primes with gap 1.
        # Actually, we need at least best_len - 1 gaps.
        if (LIMIT - p1) < best_len:
            break
            
        for j in range(i + 1, n):
            p2 = primes[j]
            g1 = p2 - p1
            
            # Upper bound on length:
            # p_k - p1 >= (k-1)*g1  => k <= (LIMIT - p1)/g1 + 1
            max_possible_len = (LIMIT - p1) // g1 + 1
            
            if max_possible_len <= best_len:
                # Since g1 increases with j, max_possible_len will only decrease.
                # So we can break the inner loop.
                break
                
            # Greedy extension
            seq = [p1, p2]
            p_curr = p2
            g_curr = g1
            curr_len = 2
            
            while True:
                target = p_curr + g_curr
                if target > LIMIT:
                    break
                p_next = next_prime_ge[target]
                if p_next == -1 or p_next > LIMIT:
                    break
                g_next = p_next - p_curr
                seq.append(p_next)
                curr_len += 1
                p_curr = p_next
                g_curr = g_next
                
            if curr_len > best_len:
                best_len = curr_len
                best_seq = seq
                
            # Time check
            if time.time() - start_time > 80:
                return best_seq

    return best_seq
