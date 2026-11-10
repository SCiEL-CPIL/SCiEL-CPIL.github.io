import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_size = 0
    
    # Helper to check validity and get sums
    def is_valid_and_get_sums(s):
        sums = {Fraction(0, 1)}
        for a in s:
            add_val = Fraction(1, a)
            shifted = {x + add_val for x in sums}
            if not shifted.isdisjoint(sums):
                return False, None
            sums = sums.union(shifted)
        return True, sums

    # Try a few deterministic orders first
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
        sorted(range(1, 23), key=lambda x: random.random()) # This is random, so do it in loop
    ]
    
    # Actually, just run the loop.
    # We can also try to extend the best set found.
    
    while time.time() < deadline:
        candidates = list(range(1, 23))
        random.shuffle(candidates)
        
        current_set = []
        current_sums = {Fraction(0, 1)}
        
        for a in candidates:
            add_val = Fraction(1, a)
            # Check if adding 'a' causes collision
            # Collision if (current_sums + add_val) intersects current_sums
            # i.e., exists x, y in current_sums such that x + add_val == y
            # i.e., y - x == add_val
            
            # Efficient check:
            # shifted = {x + add_val for x in current_sums}
            # if shifted.isdisjoint(current_sums):
            
            # Optimization:
            # If len(current_sums) is large, the set comprehension is the bottleneck.
            # But we have to do it.
            
            shifted = {x + add_val for x in current_sums}
            
            if shifted.isdisjoint(current_sums):
                current_set.append(a)
                current_sums = current_sums.union(shifted)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = sorted(current_set)
            
            # If we found a large set, we might break early if we think it's max?
            # But we don't know the max.
            # However, if best_size == 22, break.
            if best_size == 22:
                break

    return best_set
