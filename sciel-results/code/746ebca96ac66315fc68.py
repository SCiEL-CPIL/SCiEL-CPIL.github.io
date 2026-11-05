import time
import bisect

def construct():
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85
    
    # Linear sieve to get primes and smallest prime factor
    spf = list(range(TARGET + 1))
    primes = []
    for i in range(2, TARGET + 1):
        if spf[i] == i:
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > TARGET:
                break
            spf[i * p] = p
    
    # Binary search on the threshold K (a prime)
    # We want the smallest prime K such that there exist a, b > 0 with a+b=TARGET
    # and both a and b are K-smooth (all prime factors <= K)
    
    # Helper: check if a number is K-smooth (all prime factors <= K)
    def is_k_smooth(n, K):
        while n > 1:
            p = spf[n]
            if p > K:
                return False
            while n % p == 0:
                n //= p
        return True
    
    # Helper: generate all K-smooth numbers <= TARGET
    def generate_smooths(K):
        # Get primes <= K
        idx = bisect.bisect_right(primes, K)
        relevant_primes = primes[:idx]
        
        smooths = set([1])
        # Use BFS/DFS to generate all K-smooth numbers
        stack = [1]
        while stack:
            x = stack.pop()
            for p in relevant_primes:
                y = x * p
                if y > TARGET:
                    break
                if y not in smooths:
                    smooths.add(y)
                    stack.append(y)
        return smooths
    
    # Binary search for the smallest K
    lo = 0
    hi = len(primes) - 1
    best_K = primes[-1]
    best_pair = None
    
    while lo <= hi:
        if time.time() - start_time > DEADLINE:
            break
        mid = (lo + hi) // 2
        K = primes[mid]
        
        # Generate K-smooth numbers
        smooths = generate_smooths(K)
        
        # Check if there's a pair a+b=TARGET with both in smooths
        found_pair = None
        for a in smooths:
            b = TARGET - a
            if b in smooths and a > 0 and b > 0:
                found_pair = [a, b]
                break
        
        if found_pair:
            best_K = K
            best_pair = found_pair
            hi = mid - 1  # Try smaller K
        else:
            lo = mid + 1  # Need larger K
    
    if best_pair is not None:
        return best_pair
    
    # Fallback: if binary search didn't find a pair, do a full scan
    best_max_p = TARGET
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET // 2 + 1):
        if time.time() - start_time > DEADLINE:
            break
        b = TARGET - a
        pa = spf[a] if a > 1 else 1
        pb = spf[b] if b > 1 else 1
        # Need to get the LARGEST prime factor, not smallest
        # Recompute LPF properly
        def lpf(n):
            if n <= 1:
                return 1
            result = 1
            while n > 1:
                p = spf[n]
                result = p
                while n % p == 0:
                    n //= p
            return result
        
        pa = lpf(a)
        pb = lpf(b)
        current_max = max(pa, pb)
        if current_max < best_max_p:
            best_max_p = current_max
            best_a = a
            best_b = b
    
    return [best_a, best_b]
