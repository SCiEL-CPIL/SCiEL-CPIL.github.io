import random

def construct():
    limit = 40
    candidates = list(range(2, limit + 1))
    
    def is_valid(selected):
        prods = {1}
        for x in selected:
            new_prods = {p * x for p in prods}
            if new_prods & prods:
                return False
            prods |= new_prods
        return True

    best_set = []
    
    # Try a few deterministic orders first
    # 1. Increasing
    # 2. Decreasing
    # 3. Primes first
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    orders = [
        candidates,
        sorted(candidates, reverse=True),
        primes + composites,
        composites + primes
    ]
    
    for order in orders:
        current = []
        for x in order:
            new_current = current + [x]
            if is_valid(new_current):
                current = new_current
        if len(current) > len(best_set):
            best_set = current[:]

    # Randomized restarts
    import time
    start_time = time.time()
    while time.time() - start_time < 50: # Leave 10s buffer
        random.shuffle(candidates)
        current = []
        for x in candidates:
            new_current = current + [x]
            if is_valid(new_current):
                current = new_current
        if len(current) > len(best_set):
            best_set = current[:]
            # If we found a very large set, we might stop early?
            # Max possible is likely 12-14.
            if len(best_set) >= 15:
                break

    return best_set
