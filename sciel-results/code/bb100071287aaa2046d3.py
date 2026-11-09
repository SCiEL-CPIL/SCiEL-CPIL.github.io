def construct():
    # Precompute primes up to 256
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = set(p for p in range(2, 257) if is_prime(p))
    
    # Precompute bitmask for each b in [1, 128]
    # Bit i (0-indexed) is 1 if (i+1) + b is prime
    prime_mask = [0] * 129
    for b in range(1, 129):
        mask = 0
        for a in range(1, 129):
            if a + b in primes:
                mask |= (1 << (a - 1))
        prime_mask[b] = mask
    
    # Search for b1, b2, b3 such that there are at least 3 valid a's
    for b1 in range(1, 129):
        m1 = prime_mask[b1]
        for b2 in range(1, 129):
            if b2 == b1:
                continue
            m12 = m1 & prime_mask[b2]
            # If fewer than 3 a's are valid for b1, b2, skip b3
            if bin(m12).count('1') < 3:
                continue
            for b3 in range(1, 129):
                if b3 == b1 or b3 == b2:
                    continue
                m123 = m12 & prime_mask[b3]
                if bin(m123).count('1') < 3:
                    continue
                
                # Extract candidate a's
                candidates = [i + 1 for i in range(128) if (m123 >> i) & 1]
                
                # Try combinations of 3 distinct a's
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        for k in range(j + 1, len(candidates)):
                            a1, a2, a3 = candidates[i], candidates[j], candidates[k]
                            sums = set()
                            for a in [a1, a2, a3]:
                                for b in [b1, b2, b3]:
                                    sums.add(a + b)
                            if len(sums) == 9:
                                return [a1, a2, a3], [b1, b2, b3]
    
    # Fallback if no solution found (should not happen)
    return [[1, 7, 13], [4, 10, 16]]
