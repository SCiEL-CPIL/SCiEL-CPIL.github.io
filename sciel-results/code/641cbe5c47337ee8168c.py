import math

def construct():
    P_MAX = 1000000
    Q_MAX = 10000000
    
    # Sieve of Eratosthenes for primes up to P_MAX
    is_prime = [True] * (P_MAX + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(P_MAX)) + 1):
        if is_prime[i]:
            for j in range(i*i, P_MAX + 1, i):
                is_prime[j] = False
    
    # List of primes up to P_MAX
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # is_bad[p] will be True if p is known to be bad (has a prime factor q > p for some p^2+i)
    # We only care about p <= P_MAX
    is_bad = [False] * (P_MAX + 1)
    
    # Helper: Modular Square Root
    # Returns list of roots r such that r^2 == n mod m
    def mod_sqrt(n, m):
        if m == 2:
            return [n % 2]
        
        # Check if n is a quadratic residue
        # Legendre symbol
        if pow(n, (m-1)//2, m) != 1 and n % m != 0:
            return []
        if n % m == 0:
            return [0]
            
        # Tonelli-Shanks
        def legendre_symbol(a, p):
            ls = pow(a, (p-1)//2, p)
            return -1 if ls == p-1 else ls

        def tonelli_shanks(n, p):
            if p % 4 == 3:
                x = pow(n, (p+1)//4, p)
                return [x, p-x]
            
            # Find Q and S such that p-1 = Q * 2^S
            Q = p - 1
            S = 0
            while Q % 2 == 0:
                S += 1
                Q //= 2
                
            # Find a non-residue z
            z = 2
            while legendre_symbol(z, p) != -1:
                z += 1
                
            c = pow(z, Q, p)
            x = pow(n, (Q+1)//2, p)
            t = pow(n, Q, p)
            M = S
            
            while t != 1:
                # Find the smallest i such that t^(2^i) == 1
                i = 0
                t2 = t
                while t2 != 1:
                    t2 = (t2 * t2) % p
                    i += 1
                    if i == M:
                        return [] # Should not happen
                b = pow(c, 1 << (M - i - 1), p)
                x = (x * b) % p
                c = (b * b) % p
                t = (t * c) % p
                M = i
                
            return [x, p-x]

        return tonelli_shanks(n, m)

    # Sieve bad primes
    # We iterate q up to Q_MAX. 
    # For each prime q, we check which primes p < q are bad due to q.
    # p is bad if p^2 + i is divisible by q for some i in 1..200.
    # This means p^2 == -i mod q.
    
    # Optimization: Precompute -i mod q? No, q changes.
    # We can just loop i.
    
    # To speed up, we can skip q if q > P_MAX + 200? 
    # No, q can be larger than p.
    # But if q > P_MAX, we only mark p if p is in our prime list.
    
    # We need to iterate primes q.
    # Generate primes up to Q_MAX? 
    # P_MAX is 10^6. Q_MAX is 10^7.
    # Sieve up to Q_MAX is feasible.
    
    is_prime_q = [True] * (Q_MAX + 1)
    is_prime_q[0] = is_prime_q[1] = False
    for i in range(2, int(math.isqrt(Q_MAX)) + 1):
        if is_prime_q[i]:
            for j in range(i*i, Q_MAX + 1, i):
                is_prime_q[j] = False
    
    primes_q = [i for i, val in enumerate(is_prime_q) if val]
    
    for q in primes_q:
        # For each i, check if -i is QR mod q
        for i in range(1, 201):
            n = (-i) % q
            roots = mod_sqrt(n, q)
            for r in roots:
                if r < q and r <= P_MAX and is_prime[r]:
                    is_bad[r] = True

    # Now find the best k among non-bad primes
    best_k = 0
    best_p = 0
    
    # We also need to verify the candidates because the sieve only covered q <= Q_MAX.
    # If a prime p is not marked bad, it might still be bad due to a prime factor q > Q_MAX.
    # So we must factor p^2+i for the candidates.
    
    # To factor efficiently, we can use trial division by small primes first, then Miller-Rabin/Pollard if needed.
    # But p^2+i <= (10^6)^2 + 200 = 10^12.
    # Trial division up to sqrt(10^12) = 10^6 is feasible if we have the prime list.
    
    def largest_prime_factor(n):
        # n <= 10^12
        # Trial division by primes up to sqrt(n)
        # sqrt(n) <= 10^6
        # We have primes up to 10^6.
        temp_n = n
        max_f = 1
        for p in primes:
            if p * p > temp_n:
                break
            if temp_n % p == 0:
                max_f = p
                while temp_n % p == 0:
                    temp_n //= p
        if temp_n > 1:
            max_f = max(max_f, temp_n)
        return max_f

    for p in primes:
        if is_bad[p]:
            continue
            
        k = 0
        for i in range(1, 201):
            n = p * p + i
            # Check if all prime factors <= p
            # Equivalent to: largest prime factor <= p
            lpf = largest_prime_factor(n)
            if lpf <= p:
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
