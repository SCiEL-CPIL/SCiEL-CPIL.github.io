import math
import time
from functools import lru_cache

def powerful_part(n):
    """Compute the powerful part of n: product of p^e for e >= 2."""
    if n <= 1:
        return 1
    result = 1
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            if e >= 2:
                result *= p ** e
        p += 1 if p == 2 else 2
    if m > 1:
        # m is prime with exponent 1, so it doesn't contribute
        pass
    return result

def is_powerful(n):
    """Check if n is powerful (every prime factor has exponent >= 2)."""
    if n <= 1:
        return True
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            if e < 2:
                return False
        p += 1 if p == 2 else 2
    if m > 1:
        return False  # remaining prime has exponent 1
    return True

def generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set([1])
    # p^2 for primes p
    p = 2
    while p * p <= limit:
        # Check if p is prime
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            p2 = p * p
            # Multiply by other powerful numbers
            for pw in list(powerful):
                val = p2 * pw
                if val <= limit:
                    powerful.add(val)
        p += 1
    return sorted(powerful)

def modinv(a, m):
    """Modular inverse of a mod m using extended Euclidean."""
    if m == 1:
        return 0
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        return None  # No inverse
    return x % m

def extended_gcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = extended_gcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def crt_pair(a, b):
    """Solve x ≡ 0 (mod a), x ≡ -1 (mod b). Return smallest positive solution or None."""
    if math.gcd(a, b) != 1:
        return None
    # x = a * t, need a*t ≡ -1 (mod b) => t ≡ -a^{-1} (mod b)
    inv = modinv(a % b, b)
    if inv is None:
        return None
    t = (-inv) % b
    x = a * t
    if x <= 0:
        x += a * b
    return x

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    best_n = 10000
    best_score = powerful_part(10000) * powerful_part(10001)
    
    # Generate powerful numbers up to a reasonable limit
    # For n <= 1,000,000, A and B can be up to ~1,000,000
    # But A*B should be manageable for CRT solutions in range
    max_pw = 500000
    powerful_nums = generate_powerful_numbers(max_pw)
    
    # Filter to those > 1 for meaningful powerful parts
    powerful_nums = [p for p in powerful_nums if p > 1]
    
    # Sort by value descending to try larger ones first
    powerful_nums.sort(reverse=True)
    
    # Try pairs of powerful numbers
    # We want A and B such that the CRT solution n is in [10000, 1000000]
    # n = a * t where t ≡ -a^{-1} (mod b)
    # So n ≈ a * (something mod b), roughly a*b scale
    
    # Strategy: iterate over pairs, compute CRT, check range, compute score
    n_pairs = len(powerful_nums)
    
    # To be efficient, limit the search
    # Try all pairs where a*b <= some threshold to ensure solutions exist in range
    # Actually, for any coprime a, b, there's a unique solution mod a*b
    # The solution is in [1, a*b], so we need a*b >= 10000 for a chance,
    # and the solution must be <= 1000000
    
    # Let's try a two-level loop with time checking
    best_score = 0
    best_n = 10000
    
    # Precompute powerful parts for the powerful numbers themselves
    # (they are their own powerful parts since they're powerful)
    
    # Iterate: for each a in powerful_nums, for each b in powerful_nums with gcd(a,b)=1
    # This is O(n^2) which could be large. Let's be smarter.
    
    # Key insight: the powerful part of n includes at least a (if a|n and a is powerful),
    # but n may have additional prime factors. However, if we construct n = a * t where t is small,
    # the powerful part of n is at least a (times any p^e in t with e>=2).
    
    # For a good lower bound, score >= a * b when n = a*t and n+1 = b*s with t, s squarefree-ish.
    
    # Let's do a bounded search
    # Try pairs where a and b are both large
    # Limit to top K powerful numbers for speed
    
    K = min(n_pairs, 2000)  # Top 2000 powerful numbers
    top_powerful = powerful_nums[:K]
    
    for i, a in enumerate(top_powerful):
        if time.time() > deadline:
            break
        for j, b in enumerate(top_powerful):
            if time.time() > deadline:
                break
            if math.gcd(a, b) != 1:
                continue
            # Solve CRT: x ≡ 0 (mod a), x ≡ -1 (mod b)
            x = crt_pair(a, b)
            if x is None:
                continue
            # x is in [1, a*b]. Check if in range
            if x < 10000:
                continue
            if x > 1000000:
                continue
            
            # Compute actual powerful parts
            pp_n = powerful_part(x)
            pp_np1 = powerful_part(x + 1)
            score = pp_n * pp_np1
            
            if score > best_score:
                best_score = score
                best_n = x
    
    # Also try: n = a * t where t is chosen to make n+1 divisible by b
    # This is the same as CRT above.
    
    # Additional: try n where n itself is powerful and n+1 has large powerful part
    # Or vice versa. The CRT approach should cover this.
    
    return best_n
