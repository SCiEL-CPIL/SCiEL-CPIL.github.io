import time

def construct():
    start_time = time.time()
    deadline = 55.0
    
    candidates = list(range(2, 41))
    
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    candidates.sort(key=lambda x: (prime_factor_count(x), x))
    
    best_set = []
    
    def can_add(x, products):
        for p in products:
            if p * x in products:
                return False
        return True

    def backtrack(idx, current_set, current_products):
        nonlocal best_set
        
        if time.time() > deadline:
            return
            
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            
        if len(current_set) + (len(candidates) - idx) <= len(best_set):
            return
            
        if idx == len(candidates):
            return
            
        x = candidates[idx]
        
        # Try including x
        if can_add(x, current_products):
            new_products = set(current_products)
            for p in current_products:
                new_products.add(p * x)
                
            current_set.append(x)
            backtrack(idx + 1, current_set, new_products)
            current_set.pop()
            
        # Try excluding x
        backtrack(idx + 1, current_set, current_products)

    backtrack(0, [], {1})
    
    return list(best_set)
