import time
import random
import sys

# Precompute Smallest Prime Factor (SPF) up to 20000
MAX_VAL = 20000
spf = list(range(MAX_VAL + 1))
for i in range(2, int(MAX_VAL**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_VAL + 1, i):
            if spf[j] == j:
                spf[j] = i

def factorize(n):
    """Return set of prime factors of n."""
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def get_prime_union(arr):
    """Return the union of prime factors of all pairwise sums in arr."""
    primes = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            s = ai + arr[j]
            primes.update(factorize(s))
    return primes

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 12
    max_val = 10000
    
    # Initial candidates
    inits = []
    # 1. Powers of 2
    p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    inits.append(p2)
    # 2. Small integers
    inits.append(list(range(1, 13)))
    # 3. Odd integers
    inits.append(list(range(1, 25, 2)))
    # 4. Primes
    inits.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
    # 5. Random
    random.seed(42)
    for _ in range(10):
        inits.append(random.sample(range(1, max_val + 1), n))
        
    best_set = None
    best_primes = set()
    best_cost = float('inf')
    
    # Evaluate initializations
    for init in inits:
        primes = get_prime_union(init)
        cost = len(primes)
        if cost < best_cost:
            best_cost = cost
            best_set = init.copy()
            best_primes = primes
    
    # Local Search
    # We will try to replace one element at a time
    # To speed up, we can try to generate candidates that are "good"
    
    # Helper to check if a candidate is valid (distinct, <= max_val)
    def is_valid_candidate(c, current_set, idx_to_skip):
        if c > max_val:
            return False
        if c == current_set[idx_to_skip]:
            return True # We are replacing it, so it's "valid" to consider, but we check distinctness against others
        for j in range(n):
            if j != idx_to_skip:
                if c == current_set[j]:
                    return False
        return True

    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        current_set = best_set.copy()
        current_primes = best_primes.copy()
        current_cost = best_cost
        
        # Try to improve by swapping one element
        improved = False
        for _ in range(20): # Try 20 random swaps per outer loop iteration
            if time.time() > deadline:
                break
            
            idx = random.randint(0, n - 1)
            # Generate candidate
            # Bias: Try numbers close to existing ones or random
            if random.random() < 0.5:
                # Random
                c = random.randint(1, max_val)
            else:
                # Perturb existing
                base = random.choice(current_set)
                c = base + random.randint(-100, 100)
                if c < 1: c = 1
                if c > max_val: c = max_val
            
            if not is_valid_candidate(c, current_set, idx):
                continue
                
            new_set = current_set.copy()
            new_set[idx] = c
            
            new_primes = get_prime_union(new_set)
            new_cost = len(new_primes)
            
            if new_cost < current_cost:
                best_set = new_set
                best_primes = new_primes
                best_cost = new_cost
                current_cost = new_cost
                current_primes = new_primes
                improved = True
                no_improve_count = 0
            elif new_cost == current_cost and random.random() < 0.05:
                # Accept to escape local minima
                best_set = new_set
                best_primes = new_primes
                # best_cost remains same
                current_set = new_set
                current_primes = new_primes
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        if not improved and no_improve_count > max_no_improve:
            # Restart with random set
            random_set = random.sample(range(1, max_val + 1), n)
            primes = get_prime_union(random_set)
            if len(primes) < best_cost:
                best_set = random_set
                best_primes = primes
                best_cost = len(primes)
            no_improve_count = 0

    return best_set

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(res)
    print(len(get_prime_union(res)))
