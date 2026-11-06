import math
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Generate powerful numbers up to L
    L = 10**12
    powerful = set()
    # k^3 <= L => k <= 10000
    max_k = int(L ** (1/3)) + 1
    
    for k in range(1, max_k):
        k3 = k ** 3
        if k3 > L:
            break
        max_s = int(math.isqrt(L // k3))
        for s in range(1, max_s + 1):
            n = s * s * k3
            powerful.add(n)
            
    powerful_list = list(powerful)
    random.shuffle(powerful_list)
    
    # 2. Sample sums to find candidates
    # Number of samples based on time
    # Each sample is cheap. Let's do 5 million samples.
    num_samples = 5_000_000
    sum_counter = Counter()
    
    # Precompute some random indices? No, just random.choice
    # To speed up, use random.randint on the list
    for _ in range(num_samples):
        if time.time() > deadline - 30:
            break
        a = random.choice(powerful_list)
        b = random.choice(powerful_list)
        if a > b:
            a, b = b, a
        sum_counter[a + b] += 1
        
    # Get top candidates
    top_candidates = [n for n, _ in sum_counter.most_common(50)]
    
    # 3. Precompute primes for is_powerful check
    # We need primes up to 10^6 for checking numbers up to 10^18
    # Sieve of Eratosthenes
    limit_primes = 10**6
    is_prime = [True] * (limit_primes + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit_primes**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_primes + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def is_powerful(n):
        if n <= 1:
            return True
        m = n
        for p in primes:
            if p * p * p > m:
                break
            if m % p == 0:
                e = 0
                while m % p == 0:
                    m //= p
                    e += 1
                if e == 1:
                    return False
        # After the loop, m is 1 or a prime power p^e where p > 10^6
        # Since n <= 10^18, p^e <= 10^18.
        # If e >= 2, it's powerful. If e == 1, it's not.
        # But we only know m is not divisible by any prime <= 10^6.
        # So m is either 1, or a prime, or a product of primes > 10^6.
        # If m is a product of two primes > 10^6, their product is > 10^12.
        # If m is a square of a prime > 10^6, m > 10^12.
        # Actually, if m > 1:
        # Check if m is a perfect square.
        r = int(math.isqrt(m))
        if r * r == m:
            return True
        # If m is not a square, could it be powerful?
        # Only if m = 1.
        # If m is a prime, exponent is 1 -> Not powerful.
        # If m is p*q (p,q > 10^6), exponents are 1 -> Not powerful.
        # So, if m > 1 and not a square, it's not powerful.
        return False

    best_n = None
    best_pairs = []
    best_count = 0
    
    # 4. Exact count for top candidates
    # We need to find all pairs (a,b) in powerful such that a+b=N.
    # a <= b.
    # We iterate a in powerful. If a > N/2, break (since list is unsorted, we can't break easily, but we can filter).
    # Better: sort powerful_list? It's 2.6M elements. Sorting takes time but is doable.
    # Or just iterate and check.
    
    # To speed up exact count, we can just iterate through the sorted list of powerful numbers <= N/2.
    # Let's sort it once.
    powerful_sorted = sorted(powerful_list)
    
    for N in top_candidates:
        if time.time() > deadline - 10:
            break
            
        pairs = []
        # Binary search for upper bound of N/2
        # Or just iterate
        for a in powerful_sorted:
            if a > N // 2:
                break
            b = N - a
            if b < a:
                continue
            if b in powerful:
                # b is in powerful set, so it's powerful
                pairs.append([a, b])
            else:
                # b might be powerful but > L
                if b > L:
                    if is_powerful(b):
                        pairs.append([a, b])
        
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_n = N
            best_pairs = pairs
            
    if best_n is None:
        # Fallback to a simple known small N
        # N=2: 1+1. Pairs: [[1,1]]
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_n, 'pairs': best_pairs}
