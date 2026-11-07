    import bisect
    from collections import defaultdict
    from array import array

    def construct():
        N_A = 10_000_000
        # We search b up to N_B. Since a and b are symmetric in the constraint sigma(a)=sigma(b)
        # but the ratio is a/b, we fix b range.
        # If we find a pair (a,b), we can also consider (b,a) if it improves ratio?
        # |b/a - 1.5| vs |a/b - 1.5|.
        # If a/b = r, b/a = 1/r.
        # We want r close to 1.5.
        # If r is close to 1.5, 1/r is close to 0.66.
        # |1.5 - 1.5| = 0.
        # |0.66 - 1.5| = 0.84.
        # So we want a/b close to 1.5, i.e., a > b.
        # So we should search for a > 1.5 b? No, a close to 1.5 b.
        # So a is roughly 1.5 b.
        # If we set N_B = 10^7, a can be up to 1.5 * 10^7, which is > 10^7.
        # So we must restrict b such that 1.5 b <= 10^7 => b <= 6.66 * 10^6.
        # Also, could a be smaller? If a/b = 1.5, a=1.5b.
        # What if a/b = 2/3? |2/3 - 1.5| = |0.66 - 1.5| = 0.83.
        # What if a/b = 1.4? |1.4 - 1.5| = 0.1.
        # So we want a slightly larger than b.
        # Max a is 10^7.
        # So b must be such that there exists a <= 10^7 with a/b ~ 1.5.
        # So b <= 10^7 / 1.5 = 6,666,666.
        
        N_B = 6_666_666
        
        # Linear Sieve for sigma
        sigma = array('I', [0]) * (N_A + 1)
        p_power = array('I', [0]) * (N_A + 1)
        sigma_p_power = array('I', [0]) * (N_A + 1)
        is_prime = bytearray([1]) * (N_A + 1) # 1 is True
        is_prime[0] = 0
        is_prime[1] = 0
        primes = []
        
        sigma[1] = 1
        p_power[1] = 1
        sigma_p_power[1] = 1
        
        for i in range(2, N_A + 1):
            if is_prime[i]:
                primes.append(i)
                p_power[i] = i
                sigma_p_power[i] = 1 + i
                sigma[i] = 1 + i
            
            for p in primes:
                prod = i * p
                if prod > N_A:
                    break
                is_prime[prod] = 0
                
                if i % p == 0:
                    # p is smallest prime factor of i
                    p_power[prod] = p_power[i] * p
                    sigma_p_power[prod] = sigma_p_power[i] + p_power[prod]
                    sigma[prod] = (sigma[i] // sigma_p_power[i]) * sigma_p_power[prod]
                    break
                else:
                    p_power[prod] = p
                    sigma_p_power[prod] = 1 + p
                    sigma[prod] = sigma[i] * (1 + p)

        # Group by sigma
        groups = defaultdict(lambda: array('I'))
        for n in range(1, N_A + 1):
            s = sigma[n]
            groups[s].append(n)
            
        # Sort the arrays for binary search
        for s in groups:
            groups[s].sort()
            
        best_a = 31
        best_b = 21
        best_err = abs(31/21 - 1.5)
        
        # Iterate b
        # To optimize, we can skip b where sigma[b] has very few candidates?
        # Just iterate all.
        for b in range(1, N_B + 1):
            S = sigma[b]
            L = groups[S]
            if len(L) == 0:
                continue
            
            target = 1.5 * b
            # bisect works on array?
            # bisect.bisect_left(L, target)
            idx = bisect.bisect_left(L, target)
            
            candidates = []
            if idx < len(L):
                candidates.append(L[idx])
            if idx > 0:
                candidates.append(L[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                if 2 * a == 3 * b:
                    continue
                # Calculate error exactly?
                # err = abs(a/b - 1.5) = abs(2a - 3b) / (2b)
                # To avoid float issues, compare (2a - 3b)^2 / (4b^2)
                # Or just use float, it's fine for comparison if we store best_err as float.
                # But for final score, we need exact rational?
                # The task says "score is the binary64 serialization of that rational".
                # So we just need to find the pair that minimizes the float value.
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
        return [best_a, best_b]
    