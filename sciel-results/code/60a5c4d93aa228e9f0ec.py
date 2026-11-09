def construct():
    N = 123457
    
    # Helper to generate k-smooth numbers up to N
    def generate_smooth(primes, limit):
        smooth_set = set()
        # Recursive generation
        def gen(idx, current):
            if current > limit:
                return
            smooth_set.add(current)
            if idx < len(primes):
                p = primes[idx]
                # Multiply by p^k
                val = current
                while val <= limit:
                    gen(idx, val)
                    val *= p
                gen(idx + 1, current)
        
        gen(0, 1)
        return smooth_set

    # Check 7-smooth (primes 2, 3, 5, 7)
    # If found, max PF <= 7
    smooth7_set = generate_smooth([2, 3, 5, 7], N)
    for a in smooth7_set:
        if a > N: continue
        b = N - a
        if b > 0 and b in smooth7_set:
            return [a, b]

    # Check 11-smooth (primes 2, 3, 5, 7, 11)
    # If found, max PF <= 11
    smooth11_set = generate_smooth([2, 3, 5, 7, 11], N)
    for a in smooth11_set:
        if a > N: continue
        b = N - a
        if b > 0 and b in smooth11_set:
            return [a, b]

    # Fallback: compute largest prime factor and find optimal decomposition
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i

    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1

    for a in range(1, N):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        
        current_max = pa if pa > pb else pb
        
        if current_max < best_max_pf:
            best_max_pf = current_max
            best_a = a
            best_b = b
            
            if best_max_pf <= 11:
                break

    return [best_a, best_b]
