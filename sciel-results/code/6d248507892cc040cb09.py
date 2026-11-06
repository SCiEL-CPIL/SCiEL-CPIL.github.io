import random
import time
from math import gcd
from functools import reduce

def construct():
    # Calculate LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    M = reduce(lcm, range(1, 23))
    weights = [M // (i + 1) for i in range(22)]
    
    # Seed solution
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    while time.time() < deadline:
        # Randomly shuffle the order of considering elements
        order = list(range(22))
        random.shuffle(order)
        
        current_set = []
        sums = {0}
        
        # Greedy construction
        for idx in order:
            w = weights[idx]
            
            # Check if adding w causes collision
            # New sums are sums U (sums + w)
            # Collision if (sums + w) intersects sums
            # Equivalent to len(sums U (sums + w)) == 2 * len(sums)
            
            # Optimization: If len(sums) is very large, this step is the bottleneck.
            # But we stop at deadline.
            
            # We can break early if we have already exceeded best_size 
            # and we are just trying to add more? No, we need to complete the set 
            # to know its final size, but we can stop if we are sure we won't beat best?
            # Hard to know. Just run.
            
            new_sums = set(sums)
            # Add shifted sums
            for s in sums:
                new_sums.add(s + w)
                
            if len(new_sums) == 2 * len(sums):
                current_set.append(idx + 1)
                sums = new_sums
            # else: skip
            
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
            # If we found a very large set, we can exit early if we hit a theoretical max?
            # Max possible is 22. Unlikely.
            # But if best_size >= 15, it's a huge improvement.
            if best_size >= 15:
                break

    return best_set
