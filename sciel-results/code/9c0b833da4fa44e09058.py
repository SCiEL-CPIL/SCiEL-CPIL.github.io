import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    n = 12
    max_val = 10000
    max_sum = 2 * max_val
    
    # Precompute smallest prime factor (SPF) table up to 2*max_val = 20000
    spf = list(range(max_sum + 1))
    for i in range(2, int(max_sum**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_sum + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def prime_factors(x):
        """Return set of prime factors of x using SPF table."""
        factors = set()
        while x > 1:
            p = spf[x]
            factors.add(p)
            while x % p == 0:
                x //= p
        return factors
    
    def score_set(arr):
        """Count distinct prime factors across all pairwise sums."""
        all_primes = set()
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                x = s
                while x > 1:
                    p = spf[x]
                    all_primes.add(p)
                    while x % p == 0:
                        x //= p
        return len(all_primes)
    
    def score_with_primes(arr):
        """Return (score, prime_set) for a given array."""
        all_primes = set()
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                x = s
                while x > 1:
                    p = spf[x]
                    all_primes.add(p)
                    while x % p == 0:
                        x //= p
        return len(all_primes), all_primes
    
    # Key insight: If all numbers are multiples of M, then all pairwise sums are multiples of M.
    # So the prime factors of all sums are a subset of the prime factors of M (plus any additional primes from the quotients).
    # Best strategy: choose M with few prime factors, then pick 12 distinct multiples of M.
    
    # Strategy 1: All multiples of 6 -> sums are multiples of 6 -> primes subset of {2,3}
    # But the quotients (sum/6) may introduce additional primes.
    # If we pick numbers such that sum/6 is always a power of 2 or 3, we minimize primes.
    
    # Better: pick numbers that are all multiples of a highly composite number with few prime factors.
    # Let's try: all numbers are multiples of 2^a * 3^b, and we want sums to also be of that form.
    # If all numbers are multiples of 2^a * 3^b, then sums are multiples of 2^a * 3^b.
    # The quotient (sum / (2^a*3^b)) needs to have few prime factors.
    
    # Simplest: all multiples of 6. Sums are multiples of 6. Quotients are integers.
    # If we pick numbers 6*a_i, sums are 6*(a_i + a_j). We need a_i + a_j to have few prime factors.
    # If a_i are powers of 2: a_i + a_j has at most 2 prime factors (for distinct powers, sum is odd * power of 2).
    # Actually, 2^i + 2^j = 2^min(i,j) * (1 + 2^|i-j|), and 1+2^k is odd.
    # So primes of sum = {2, 3} union primes of (1+2^k).
    
    # Let's try: numbers = 6 * 2^i for i=0..11. All <= 10000? 6*2^11 = 12288 > 10000.
    # So max i such that 6*2^i <= 10000: 2^i <= 1666, i <= 10. Only 11 values.
    
    # Alternative: mix of powers. Or use a different base.
    
    # Let's try a structured approach:
    # Pick a base B with few prime factors. All numbers = B * k_i.
    # Sums = B * (k_i + k_j). Primes = primes(B) union primes(k_i + k_j).
    
    # Best B: 6 (primes {2,3}). Then we need k_i + k_j to have few new primes.
    # If k_i are powers of 2: k_i + k_j = 2^min * (1 + 2^diff). 
    # 1+2^diff for diff=1..11: 3,5,9,17,33,65,129,257,513,1025,2049
    # Primes: 3,5,9=3^2,17,33=3*11,65=5*13,129=3*43,257,513=3^3*19,1025=5^2*41,2049=3*683
    # Union: {2,3,5,11,13,17,19,41,43,683} - that's 10 primes. Not great.
    
    # What if k_i are all 1? Can't, need distinct.
    
    # What if we use B=30 (primes {2,3,5})? Then sums are multiples of 30.
    # k_i + k_j needs to have few primes. If k_i are powers of 2, similar issue.
    
    # Alternative: make all k_i equal? No, need distinct numbers.
    
    # Let's try: all numbers are multiples of 6, and we use local search to minimize.
    
    # Initializations
    initializations = []
    
    # Multiples of 6
    init1 = [6 * i for i in range(1, n+1)]
    initializations.append(init1)
    
    # Multiples of 30
    init2 = [30 * i for i in range(1, n+1)]
    initializations.append(init2)
    
    # Even numbers
    init3 = [2 * i for i in range(1, n+1)]
    initializations.append(init3)
    
    # Multiples of 12
    init4 = [12 * i for i in range(1, n+1)]
    initializations.append(init4)
    
    # Powers of 2 times 6 (as many as fit)
    init5 = []
    v = 6
    while len(init5) < n and v <= max_val:
        init5.append(v)
        v *= 2
    if len(init5) >= n:
        initializations.append(init5[:n])
    
    # Mix: small multiples of 6
    random.seed(42)
    for _ in range(30):
        candidates = random.sample(range(6, max_val+1, 6), n)
        initializations.append(sorted(candidates))
    
    for _ in range(20):
        candidates = random.sample(range(2, max_val+1, 2), n)
        initializations.append(sorted(candidates))
    
    # Small numbers
    for _ in range(20):
        candidates = random.sample(range(1, 200), n)
        initializations.append(sorted(candidates))
    
    best_score = float('inf')
    best_arr = None
    
    for init in initializations:
        if time.time() > deadline:
            break
        s = score_set(init)
        if s < best_score:
            best_score = s
            best_arr = list(init)
    
    if best_arr is None:
        return list(range(1, n+1))
    
    # Local search with simulated annealing style
    arr = list(best_arr)
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 3000
    temperature = 1.0
    cooling = 0.999
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Try random replacements
        improved = False
        for _ in range(300):
            if time.time() > deadline:
                break
            i = random.randint(0, n-1)
            # Bias towards small values and multiples of 6
            if random.random() < 0.5:
                new_val = random.randint(1, 300)
            else:
                new_val = random.randint(6, max_val)
                if new_val % 6 != 0:
                    new_val = 6 * (new_val // 6 + 1)
            
            if new_val in arr and new_val != arr[i]:
                continue
            
            old_val = arr[i]
            arr[i] = new_val
            
            new_score = score_set(arr)
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                improved = True
                temperature = max(0.1, temperature * cooling)
            else:
                delta = new_score - current_score
                # Accept worse with probability exp(-delta/T)
                if delta > 0 and random.random() < pow(2.718281828, -delta / temperature):
                    current_score = new_score
                    no_improve_count = 0
                else:
                    arr[i] = old_val
                    no_improve_count += 1
        
        if not improved:
            temperature = max(0.1, temperature * cooling)
    
    # Final targeted local search
    for _ in range(5):
        if time.time() > deadline:
            break
        for i in range(n):
            if time.time() > deadline:
                break
            for trial in range(50):
                if time.time() > deadline:
                    break
                # Try multiples of 6 near current value
                base = arr[i]
                offset = random.choice([-6, -3, -2, -1, 1, 2, 3, 6])
                new_val = base + offset
                if new_val < 1 or new_val > max_val:
                    continue
                if new_val in arr and new_val != arr[i]:
                    continue
                old_val = arr[i]
                arr[i] = new_val
                new_score = score_set(arr)
                if new_score < current_score:
                    current_score = new_score
                else:
                    arr[i] = old_val
    
    return sorted(arr)
