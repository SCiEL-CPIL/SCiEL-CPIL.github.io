import time
import random

def construct():
        deadline = time.time() + 55
        # Start with the known valid set of size 35
        best_set = set(range(66, 101))
        best_size = len(best_set)
        
        def is_valid(s):
            # Bitset DP for subset sum up to 200
            # reachable[i] is 1 if sum i is reachable
            reachable = 1  # bit 0 set
            for x in s:
                if x > 200:
                    continue
                reachable |= (reachable << x)
            return (reachable >> 200) & 1 == 0

        # Local search to try to increase size
        # We try to add elements from 1 to 65
        # If adding x fails, we try swapping x with y in best_set
        
        elements_to_try = list(range(1, 66))
        random.shuffle(elements_to_try)
        
        for x in elements_to_try:
            if time.time() > deadline:
                break
            if x in best_set:
                continue
                
            new_set = best_set | {x}
            if is_valid(new_set):
                best_set = new_set
                best_size += 1
            else:
                # Try swapping x with some y in best_set
                # We want to keep size constant, but maybe enable future additions?
                # Or just see if we can find a better configuration of size 35?
                # The goal is to MAXIMIZE size.
                # If we can't add, we can't increase size.
                # So we stop trying to add.
                pass
        
        return list(best_set)
