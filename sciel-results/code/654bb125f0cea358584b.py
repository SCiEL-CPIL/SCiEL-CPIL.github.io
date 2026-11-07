def construct():
    import time
    import random
    
    deadline = time.time() + 85
    
    candidates = list(range(2, 41))
    
    def try_add(products, x):
        # Returns new products if x can be added, else None
        new = set()
        for p in products:
            val = p * x
            if val in products or val in new:
                return None
            new.add(val)
        return products | new
    
    best = []
    
    # Greedy with different orderings
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if trial % 3 == 0:
            order = candidates[:]
            random.shuffle(order)
        elif trial % 3 == 1:
            order = sorted(candidates, key=lambda x: x)  # ascending
            random.shuffle(order)
        else:
            # Prioritize primes and prime powers
            from math import isqrt
            def is_prime(n):
                if n < 2: return False
                for d in range(2, isqrt(n)+1):
                    if n % d == 0:
                        return False
                return True
            def prime_power_score(n):
                # Higher score for prime powers
                score = 0
                for p in [2,3,5,7,11,13,17,19,23,29,31,37]:
                    if n % p == 0:
                        count = 0
                        while n % p == 0:
                            n //= p
                            count += 1
                        if count >= 2:
                            score += 10
                return -score + random.random()
            order = sorted(candidates, key=prime_power_score)
        
        products = {1}
        selected = []
        
        for x in order:
            new_products = try_add(products, x)
            if new_products is not None:
                products = new_products
                selected.append(x)
        
        if len(selected) > len(best):
            best = selected[:]
    
    return best
