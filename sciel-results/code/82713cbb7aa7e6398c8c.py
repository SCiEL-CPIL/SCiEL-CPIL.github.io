import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    candidates = list(range(2, 41))
    
    def get_encoded(n):
        enc = 0
        mult = 1
        for p in primes:
            count = 0
            while n % p == 0:
                count += 1
                n //= p
            enc += count * mult
            mult *= 100
        return enc
    
    encoded_candidates = [(n, get_encoded(n)) for n in candidates]
    
    best_set = []
    best_size = 0
    
    # Run greedy with random restarts
    while time.time() < deadline:
        random.shuffle(encoded_candidates)
        current_set = []
        current_sums = {0}
        
        for n, enc in encoded_candidates:
            if time.time() > deadline:
                break
            
            # Check for collision
            # new_sums = {s + enc for s in current_sums}
            # If any s + enc is in current_sums, collision.
            
            # Optimization: 
            # Instead of creating the full set, we can iterate?
            # No, set comprehension is faster in C.
            
            new_sums = {s + enc for s in current_sums}
            
            # Check intersection
            if not current_sums.isdisjoint(new_sums):
                continue
            else:
                current_sums |= new_sums
                current_set.append(n)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
            # If we hit the theoretical max or a very high number, stop
            # Max possible is likely <= 18 or 19.
            if best_size >= 18:
                break

    return best_set
