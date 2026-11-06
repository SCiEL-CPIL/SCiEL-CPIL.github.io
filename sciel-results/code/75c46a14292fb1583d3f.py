import time

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457
    
    # Generate all y-smooth numbers up to TARGET for y=11, 13, 17, 19, 23
    # For each smoothness level, find a pair (a, b) with a+b=TARGET, both y-smooth
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = n
        return lp
    
    def is_smooth(n, max_prime):
        """Check if n is max_prime-smooth (all prime factors <= max_prime)"""
        if n <= 1:
            return True
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if p > max_prime:
                break
            while n % p == 0:
                n //= p
        return n == 1
    
    def generate_smooth_numbers(max_prime, limit):
        """Generate all max_prime-smooth numbers <= limit"""
        primes = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if p <= max_prime:
                primes.append(p)
            else:
                break
        
        smooth = set([1])
        for p in primes:
            new_smooth = set()
            for s in smooth:
                val = s
                while val <= limit:
                    new_smooth.add(val)
                    val *= p
            smooth |= new_smooth
        return [s for s in smooth if s <= limit]
    
    best = None
    best_val = float('inf')
    
    # Try increasing smoothness levels
    for max_p in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643]:
        if time.time() > deadline:
            break
        
        smooth_nums = generate_smooth_numbers(max_p, TARGET)
        smooth_set = set(smooth_nums)
        
        found = False
        for a in smooth_nums:
            if time.time() > deadline:
                break
            b = TARGET - a
            if b < 1:
                continue
            if b in smooth_set:
                # Both are max_p-smooth, so max(P(a),P(b)) <= max_p
                val = max(largest_prime_factor(a), largest_prime_factor(b))
                if val < best_val:
                    best_val = val
                    best = [a, b]
                    found = True
                    # Since we're going in order of increasing max_p, first found is optimal for this max_p
                    # But we should still check if we can do better with smaller actual P
                    break
        
        if found:
            # We found a pair with max prime factor <= max_p
            # Try to see if we can do better with a smaller max_p? No, we're going in order.
            # But maybe there's a pair with smaller actual P within this smoothness level.
            # Let's continue searching a bit more within this level if time permits
            # Actually, since we want to minimize the actual max(P(a),P(b)), not just the bound,
            # let's collect all valid pairs and pick the best
            pass
        
        # If we found something and time is running out, break
        if best is not None and time.time() > deadline - 5:
            break
    
    if best is None:
        # Fallback: use the seed
        best = [1, 123456]
    
    return best
