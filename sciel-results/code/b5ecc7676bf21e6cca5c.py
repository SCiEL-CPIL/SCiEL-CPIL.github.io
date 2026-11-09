import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer for safety
    
    # We need n up to 1000000, so n+1 up to 1000001
    N = 1000001
    
    # 1. Compute primes up to sqrt(N) is enough for the sieve?
    # No, we need primes up to sqrt(N) to find p^2 <= N.
    # Primes p where p^2 <= N => p <= sqrt(N).
    # sqrt(1000001) ~ 1000.
    # So we only need primes up to 1000.
    
    limit_sqrt = int(N**0.5) + 1
    is_prime = [True] * (limit_sqrt + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit_sqrt**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit_sqrt + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit_sqrt + 1) if is_prime[i]]
    
    # 2. Compute Powerful Part P(n) for all n
    P = [1] * (N + 1)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        # For each power p^e with e >= 2
        pe = p2
        while pe <= N:
            pe_next = pe * p
            if pe_next > N:
                # pe is the highest power <= N
                # All multiples of pe are not divisible by pe_next (which is > N)
                # So we multiply all multiples of pe by pe
                # Wait, if pe_next > N, then no m <= N is divisible by pe_next.
                # So all multiples of pe get the factor pe.
                for m in range(pe, N + 1, pe):
                    P[m] *= pe
                break
            
            # For numbers divisible by pe but not by pe_next, multiply P by pe
            # Equivalent to: m = k * pe, where k is not divisible by p
            limit_k = N // pe
            for k in range(1, limit_k + 1):
                if k % p != 0:
                    P[k * pe] *= pe
            
            pe = pe_next
    
    # 3. Find max P(n)*P(n+1) with deadline guard
    max_val = 0
    best_n = 10000
    # n ranges from 10000 to 1000000
    for n in range(10000, N): # N is 1000001, so range ends at 1000000
        score = P[n] * P[n + 1]
        if score > max_val:
            max_val = score
            best_n = n
        
        # Deadline check
        if n % 100000 == 0 and time.time() > deadline:
            break
    
    return best_n
