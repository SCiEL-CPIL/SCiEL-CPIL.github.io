from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Candidates are 1 to 24
    # We process in decreasing order of value (increasing index)
    # This means we try to include 24, 23, ..., 1
    
    S = []
    sums = {Fraction(0, 1)}
    
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
            
        target = Fraction(1, x)
        
        # Check if target is already in sums
        if target in sums:
            continue
            
        # If not, we can add x
        # Update sums with new combinations including x
        # new_sums = sums U {s + target for s in sums}
        
        # Optimization: only keep sums <= 1
        # Since future targets are 1/y with y < x, so 1/y > 1/x.
        # Actually, future targets are LARGER.
        # So we need sums that can potentially equal a LARGER value?
        # No, we check if target (which is getting LARGER as x decreases) is in sums.
        # The sums are formed by 1/y where y > x.
        # So sums are small.
        # We need to know if a specific small number is in the set.
        # We should keep all sums, but prune those > 1?
        # Max target is 1/1 = 1.
        # So any sum > 1 can never be a target.
        # So we can prune sums > 1.
        
        new_sums = set(sums)
        for s in sums:
            ns = s + target
            if ns <= 1:
                new_sums.add(ns)
        
        sums = new_sums
        S.append(x)
        
    return S
