import time
import math

def _factorize(n):
    """Factorize n into (prime, exponent) pairs."""
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1 if d == 2 else 2
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _score(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    if n < 2:
        return 0
    factors = _factorize(n)
    
    # Generate all divisors
    divisors = [1]
    for p, e in factors:
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    divisors.sort()
    
    # Count adjacent coprime pairs
    score = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            score += 1
    return score

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill
    
    # Primes list (enough for up to ~15 distinct primes)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    LIMIT = 10**18
    
    def make_n(exponents, num_primes):
        """Build n from exponent list applied to first num_primes primes."""
        n = 1
        for i in range(num_primes):
            if exponents[i] > 0:
                n *= primes[i] ** exponents[i]
            if n > LIMIT:
                return -1  # overflow signal
        return n
    
    def score_n(n):
        return _score(n)
    
    best_n = 2
    best_score = 0
    
    # === Strategy 1: Structured DFS over monotone exponent space ===
    # We search over exponent vectors (e1 >= e2 >= ... >= ek >= 1)
    # where e1 applies to prime 2, e2 to prime 3, etc.
    # This is the key structural insight: monotone exponents cover all
    # "balanced" factorizations efficiently.
    
    def dfs_exponents(idx, max_exp, current_exp, current_n, num_primes):
        """
        DFS over exponent assignments for primes[0..num_primes-1].
        idx: current prime index
        max_exp: maximum allowed exponent (monotonicity constraint)
        current_exp: exponent assigned to previous prime
        current_n: running product
        num_primes: total number of primes to assign
        """
        nonlocal best_n, best_score
        
        if time.time() > deadline - 10:
            return
        
        if idx == num_primes:
            # All primes assigned, evaluate
            if current_n >= 2 and current_n <= LIMIT:
                s = _score(current_n)
                if s > best_score:
                    best_score = s
                    best_n = current_n
            return
        
        # Try exponents from max_exp down to 1
        # (we must assign at least exponent 1 to keep the prime)
        for e in range(min(max_exp, current_exp), 0, -1):
            new_n = current_n * (primes[idx] ** e)
            if new_n > LIMIT:
                # Since exponents only decrease, smaller e might still work
                # But if e=1 overflows, we must stop (all smaller also overflow? No, smaller is better)
                # Actually if primes[idx]^1 * current_n > LIMIT, then no exponent works
                # But wait, we're going DOWN, so e=1 is the minimum. If e=1 overflows, break.
                if e == 1:
                    break
                continue
            
            # Pruning: if even with minimum exponents (1) for remaining primes,
            # we can't stay under limit, skip
            # But this is complex; just proceed and let the recursion handle it
            
            dfs_exponents(idx + 1, e, e, new_n, num_primes)
    
    # Try different numbers of distinct primes
    # Max distinct primes: product of first k primes <= 10^18
    # 2*3*5*...*47 (15 primes) = 614889782588491410 ~ 6.1e17, fits
    # 2*3*5*...*53 (16 primes) ~ 3.2e19, too big
    # So max 15 distinct primes for squarefree, fewer with exponents
    
    # Start with smaller k (more room for exponents) and work up
    for num_primes in range(2, 16):
        if time.time() > deadline - 10:
            break
        
        # Maximum exponent for prime 2: 2^e * 3 * 5 * ... <= 10^18
        # For num_primes distinct primes with min exponent 1:
        # base = product of first num_primes primes
        base = 1
        for i in range(num_primes):
            base *= primes[i]
        if base > LIMIT:
            continue
        
        # Max exponent for prime 2: 2^(e-1) * base <= 10^18
        # 2^(e-1) <= 10^18 / base
        max_e1 = 1
        temp = base
        while temp * 2 <= LIMIT:
            temp *= 2
            max_e1 += 1
        # max_e1 is the max exponent for prime 2 (since base includes 2^1)
        # Actually base has 2^1, so 2^(e1-1) * (base/2) * 3 * 5... 
        # Let me recompute: n = 2^e1 * 3^e2 * ... 
        # base = 2*3*5*...
        # n = base * 2^(e1-1) * 3^(e2-1) * ...
        # So 2^(e1-1) <= 10^18 / base
        max_e1 = 1
        val = 1
        while val * 2 <= LIMIT // base:
            val *= 2
            max_e1 += 1
        # max_e1 is max for e1
        
        if max_e1 < 1:
            continue
        
        # DFS with max exponent for first prime
        dfs_exponents(0, max_e1, max_e1, 1, num_primes)
    
    # === Strategy 2: Local hill-climbing on best candidates ===
    # Take the best candidate and try adjusting individual exponents
    
    if time.time() < deadline - 15:
        # Get factors of best_n
        if best_n >= 2:
            factors = _factorize(best_n)
            # Try increasing/decreasing each exponent by 1
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
                factors_copy = list(factors)
                for i in range(len(factors_copy)):
                    p, e = factors_copy[i]
                    # Try e+1
                    if e + 1 <= 60:  # 2^60 is way over, but for large p it's fine
                        new_n = best_n * p
                        if new_n <= LIMIT:
                            s = _score(new_n)
                            if s > best_score:
                                best_score = s
                                best_n = new_n
                                factors = _factorize(best_n)
                                improved = True
                                break
                    # Try e-1 (if e > 1)
                    if e > 1:
                        new_n = best_n // p
                        if new_n >= 2:
                            s = _score(new_n)
                            if s > best_score:
                                best_score = s
                                best_n = new_n
                                factors = _factorize(best_n)
                                improved = True
                                break
    
    # === Strategy 3: Targeted high-exponent candidates ===
    # Based on insights: 2^a * 3^b * 5 * 7 * ... with a+b optimized
    
    if time.time() < deadline - 10:
        # For 14 distinct primes (2..43), try various (e2, e3) pairs
        p14 = 1
        for i in range(14):
            p14 *= primes[i]
        
        if p14 <= LIMIT:
            # n = 2^(a) * 3^(b) * 5 * 7 * ... * 43
            # = p14 * 2^(a-1) * 3^(b-1)
            for a in range(1, 20):
                if time.time() > deadline - 10:
                    break
                for b in range(1, 20):
                    if time.time() > deadline - 10:
                        break
                    mult = (2 ** (a-1)) * (3 ** (b-1))
                    if mult > LIMIT // p14:
                        break
                    n = p14 * mult
                    if n <= LIMIT and n >= 2:
                        s = _score(n)
                        if s > best_score:
                            best_score = s
                            best_n = n
        
        # For 13 distinct primes (2..41)
        p13 = p14 // 43
        if p13 <= LIMIT:
            for a in range(1, 30):
                if time.time() > deadline - 10:
                    break
                for b in range(1, 30):
                    if time.time() > deadline - 10:
                        break
                    mult = (2 ** (a-1)) * (3 ** (b-1))
                    if mult > LIMIT // p13:
                        break
                    n = p13 * mult
                    if n <= LIMIT and n >= 2:
                        s = _score(n)
                        if s > best_score:
                            best_score = s
                            best_n = n
        
        # For 12 distinct primes (2..37), allow higher exponents on 5 too
        p12 = p13 // 41
        if p12 <= LIMIT:
            for a in range(1, 25):
                if time.time() > deadline - 10:
                    break
                for b in range(1, 25):
                    if time.time() > deadline - 10:
                        break
                    for c in range(1, 15):
                        if time.time() > deadline - 10:
                            break
                        mult = (2 ** (a-1)) * (3 ** (b-1)) * (5 ** (c-1))
                        if mult > LIMIT // p12:
                            break
                        n = p12 * mult
                        if n <= LIMIT and n >= 2:
                            s = _score(n)
                            if s > best_score:
                                best_score = s
                                best_n = n
    
    # === Strategy 4: Squarefree with 15 primes ===
    if time.time() < deadline - 5:
        n_sq = 1
        for i in range(15):
            n_sq *= primes[i]
        if n_sq <= LIMIT:
            s = _score(n_sq)
            if s > best_score:
                best_score = s
                best_n = n_sq
    
    return best_n
