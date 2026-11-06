import math
import time
import random

# Deterministic Miller-Rabin bases for n < 2^64
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = 128
        g = 1
        r = 1
        q = 1
        x = y
        ys = y
        
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        
        if g != n:
            return g

def factorize(n):
    factors = []
    stack = [n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if is_prime(x):
            factors.append(x)
        else:
            d = pollard_brent(x)
            stack.append(d)
            stack.append(x // d)
    return factors

def largest_prime_factor(n):
    if n <= 1:
        return 1
    factors = factorize(n)
    return max(factors)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # We want to maximize k.
    # Let's fix a target K_max to search for, or search for the max possible.
    # The prompt says k <= 200.
    # Let's try to find a p with k=200 first? That might be too hard.
    # The seed found k=2 (score 2.33 implies seed k was ~1.8? No, score is value/seed.
    # If seed is 1.0, value is 1. If current score is 2.33, current k is likely 2 or 3?
    # Wait, "score 2.3333" for current program.
    # If seed program returns k=1, score 1.0.
    # Current program returns k=2 or 3?
    # Let's assume the "seed" value is the k value from a baseline.
    # If baseline k=1, score 1.0.
    # Current k=2, score 2.0? Or maybe the seed is different.
    # Regardless, we want to maximize k.
    
    # Strategy:
    # 1. Choose a sieve limit for q.
    # 2. For a fixed target K (e.g., 20, 50, 100, 200), sieve primes p up to P_max.
    # 3. For each surviving candidate p, verify the run length.
    
    # How many candidates survive?
    # For each q, roughly 2/q of primes are eliminated (if -i is QR for some i).
    # Actually, for a fixed q, the number of i in [1, K] such that -i is a QR mod q is roughly K/2 (half are QRs).
    # The union of these residues covers at most K residues mod q.
    # So roughly K/q primes are eliminated by each q.
    # Total survivors ~ P_max * prod_{q <= Q} (1 - K/q).
    # This is roughly P_max / (K log K)^{pi(Q)}? No.
    # prod (1 - 1/q) ~ 1/log Q.
    # So survivors ~ P_max / (log Q)^{something}.
    
    # Let's try K=20 first. It's a good balance.
    # If we find a p with k=20, score is at least 20.
    # Current score 2.33 suggests current k is small.
    
    # Let's dynamically increase K.
    # Start with K=10.
    # If we find a p with k >= K, we can try higher K.
    # But the sieve depends on K.
    
    # Let's pick a reasonable K to target, say K=30.
    # And P_max = 10^6.
    
    K_target = 30
    P_max = 10**6
    
    # Sieve primes up to Q_max
    # Q_max should be large enough to filter well, but not so large that the sieve is slow.
    # Q_max = 1000? 10000?
    # Let's use Q_max = 1000.
    Q_max = 1000
    
    # Generate primes up to Q_max
    is_prime_q = [True] * (Q_max + 1)
    is_prime_q[0] = is_prime_q[1] = False
    for i in range(2, int(Q_max ** 0.5) + 1):
        if is_prime_q[i]:
            for j in range(i * i, Q_max + 1, i):
                is_prime_q[j] = False
    small_primes = [i for i, prime in enumerate(is_prime_q) if prime]
    
    # For each q in small_primes, find residues r such that r^2 + i == 0 mod q for some i in [1, K_target]
    # i.e., r^2 == -i mod q.
    # Precompute for each q the set of forbidden residues.
    
    forbidden_residues = {}
    for q in small_primes:
        # i ranges from 1 to K_target
        # -i mod q ranges from -1 to -K_target mod q
        # We need to check which of these are quadratic residues.
        # If -i is a QR mod q, then p^2 == -i is solvable, so p is forbidden.
        forb = set()
        for i in range(1, K_target + 1):
            val = (-i) % q
            # Check if val is a QR mod q
            # Euler's criterion: val^{(q-1)/2} == 1 mod q
            # Or just compute sqrt?
            # For small q, precompute squares is fast.
            # squares = {r^2 % q for r in range(q)}
            # But we can do this more efficiently.
            # Actually, for each q, we can just compute the set of squares once.
            pass
        
        # Better: compute set of squares for q once.
        squares = set()
        for r in range(q):
            squares.add((r * r) % q)
        
        for i in range(1, K_target + 1):
            val = (-i) % q
            if val in squares:
                # Then p^2 == val is solvable.
                # We need to find the roots of p^2 == val mod q.
                # If q is prime, roots are +/- sqrt(val).
                # We can find one root by brute force for small q.
                for r in range(q):
                    if (r * r) % q == val:
                        forb.add(r)
                        forb.add((q - r) % q)
                        break
                
        forbidden_residues[q] = forb
    
    # Now, sieve primes p up to P_max.
    # A prime p is a candidate if for all q in small_primes, p % q is NOT in forbidden_residues[q].
    
    # We can use a simple sieve for p.
    # is_prime_p = [True] * (P_max + 1)
    # Mark composites.
    # Then for each prime p, check conditions.
    
    # This might be slow if we check each p against all q.
    # 78498 primes * 168 q's = 13 million checks. Fast.
    
    # Let's generate primes up to P_max first.
    is_prime_p = [True] * (P_max + 1)
    is_prime_p[0] = is_prime_p[1] = False
    for i in range(2, int(P_max ** 0.5) + 1):
        if is_prime_p[i]:
            for j in range(i * i, P_max + 1, i):
                is_prime_p[j] = False
    
    candidates = []
    for p in range(2, P_max + 1):
        if not is_prime_p[p]:
            continue
        
        # Check against all q
        valid = True
        for q in small_primes:
            if p % q in forbidden_residues[q]:
                valid = False
                break
        
        if valid:
            candidates.append(p)
            
    if time.time() > deadline:
        # Fallback
        return {'p': 7, 'k': 1}
        
    # Now verify candidates.
    # We want to find the max k.
    # Sort candidates? Maybe larger p are better? Or random?
    # Let's just iterate.
    
    best_p = 0
    best_k = 0
    
    # To save time, we can check candidates in random order or sorted.
    # Let's check all of them, but break if time runs out.
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        p2 = p * p
        k = 0
        # Check run length
        # We know from sieve that p^2 + i for i in [1, K_target] likely doesn't have small prime factors q.
        # But it might have large prime factors > p.
        # We need to factorize p^2 + i and check max factor <= p.
        
        # Optimization: if k > best_k, we only need to check up to best_k first?
        # No, we want the max k.
        # But if we find a non-smooth number at j, we stop.
        
        # Let's check j=1 to 200.
        # We can stop early if j > best_k and we just want to confirm it's not better?
        # No, we want the actual k.
        
        # However, if we find k > best_k, we update.
        
        # Fast check:
        # If p^2 + j is prime and > p, it fails.
        
        for j in range(1, 201):
            n = p2 + j
            # Check if n is p-smooth
            # If n is prime and n > p, fail.
            if n > p and is_prime(n):
                break
            
            # Factorize
            # Optimization: if we already know n has a factor > p from previous steps? No.
            # We can trial divide by primes up to p? No, p is large.
            # Pollard-Brent is good.
            
            # We can check small factors first?
            # The sieve already removed small q factors.
            # So n has no prime factors q <= Q_max (except possibly if q | p, but p is prime and q < p, so gcd(p,q)=1).
            # Wait, if q | p^2 + j, then p^2 == -j mod q.
            # We sieved out p such that p^2 == -j mod q for j in [1, K_target].
            # So for j <= K_target, n is not divisible by any q <= Q_max.
            # This means the smallest prime factor of n is > Q_max.
            # This makes factorization faster? Maybe.
            
            # Let's just factorize.
            factors = factorize(n)
            if max(factors) > p:
                break
            
            k = j
            if k > best_k:
                best_k = k
                best_p = p
                
            if k == 200:
                break
                
    if best_k == 0:
        # Fallback
        # Try small primes directly
        for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]:
            p2 = p * p
            k = 0
            while k < 200:
                n = p2 + k + 1
                if n > p and is_prime(n):
                    break
                factors = factorize(n)
                if max(factors) > p:
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
            if best_k > 5:
                break

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
