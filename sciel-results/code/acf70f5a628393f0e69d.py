"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 24
    elements = list(range(1, n + 1))
    
    best = list(range(13, 25))  # {13,...,24}, size 12
    
    def backtrack(idx, current, achievable):
        nonlocal best
        if time.time() > deadline:
            return
        
        if len(current) > len(best):
            best = list(current)
        
        if idx == n:
            return
        
        # Option: skip element idx
        backtrack(idx + 1, current, achievable)
        
        # Option: include element idx
        a = elements[idx]
        recip_a = Fraction(1, a)
        
        # Check if 1/a is already an achievable sum
        if recip_a in achievable:
            return
        
        # Check conflicts: for each b in current, is (1/b - 1/a) a subset sum of others?
        valid = True
        for b in current:
            diff = Fraction(1, b) - recip_a
            if diff <= 0:
                continue
            # Check subset sum of {1/x : x in current, x != b}
            sums = set()
            for x in current:
                if x == b:
                    continue
                rx = Fraction(1, x)
                new_sums = {rx}
                for s in sums:
                    ns = s + rx
                    new_sums.add(ns)
                    if ns == diff:
                        valid = False
                        break
                if not valid:
                    break
                sums = sums | new_sums
            if diff in sums:
                valid = False
                break
        
        if not valid:
            return
        
        # Update achievable sums
        new_achievable = set(achievable)
        new_achievable.add(recip_a)
        for s in list(achievable):
            new_achievable.add(s + recip_a)
        
        backtrack(idx + 1, current + [a], new_achievable)
    
    backtrack(0, [], set())
    
    return sorted(best)
# EVOLVE-BLOCK-END
