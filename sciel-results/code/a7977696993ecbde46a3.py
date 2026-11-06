import random

def construct():
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL  # 20000
    
    # Precompute smallest prime factor for all numbers up to MAX_SUM
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        """Return set of distinct prime factors using SPF."""
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    def objective(nums):
        prime_set = set()
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                s = nums[i] + nums[j]
                prime_set |= get_prime_factors(s)
        return len(prime_set)
    
    random.seed(42)
    
    # Try structured initializations
    best = None
    best_score = float('inf')
    
    # Initialization 1: random samples
    for seed in range(50):
        random.seed(seed)
        candidate = random.sample(range(1, MAX_VAL+1), 12)
        score = objective(candidate)
        if score < best_score:
            best_score = score
            best = candidate[:]
    
    # Initialization 2: multiples of small numbers
    for mod in [2, 3, 4, 5, 6]:
        multiples = [i for i in range(mod, MAX_VAL+1, mod)][:100]
        if len(multiples) >= 12:
            for seed in range(10):
                random.seed(seed + 1000)
                candidate = random.sample(multiples, 12)
                score = objective(candidate)
                if score < best_score:
                    best_score = score
                    best = candidate[:]
    
    # Local search with restarts
    for restart in range(5):
        for iteration in range(20000):
            candidate = best[:]
            idx = random.randint(0, 11)
            new_val = random.randint(1, MAX_VAL)
            if new_val in candidate:
                continue
            candidate[idx] = new_val
            score = objective(candidate)
            if score < best_score:
                best_score = score
                best = candidate[:]
                if best_score <= 3:
                    return best
        
        # Perturb: replace 3 random elements
        for _ in range(3):
            idx = random.randint(0, 11)
            best[idx] = random.randint(1, MAX_VAL)
            # Ensure distinctness
            while len(set(best)) < 12:
                best[random.randint(0, 11)] = random.randint(1, MAX_VAL)
        best_score = objective(best)
    
    return best
