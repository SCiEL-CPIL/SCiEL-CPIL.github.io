"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    p = 1009
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # For each element e, we need to check if inv[e] is in the same interval
    # and e != inv[e] (diagonal is fine as long as e^2 % p != 1, i.e., e != 1 and e != p-1)
    # Actually, diagonal: x=y means x*x % p != 1, so x != 1 and x != 1008
    
    # An interval [s, e] is valid iff:
    # 1. For all x in [s,e], x != 1 and x != p-1 (diagonal constraint)
    # 2. For all x in [s,e], inv[x] is NOT in [s,e] (unless inv[x] == x, but that only happens for x=1 or x=p-1 which are excluded)
    #    Wait, if inv[x] == x, then x*x % p == 1, which is the diagonal case. So for x != 1, p-1, inv[x] != x.
    #    So condition 2 is: for all x in [s,e], inv[x] < s or inv[x] > e
    
    # Key insight: if we sort the pairs (x, inv[x]) and note that inv[inv[x]] = x,
    # we can think of this as a constraint satisfaction problem.
    # 
    # For a fixed start s, what's the maximum valid end e?
    # As e increases, the constraint "inv[x] > e for all x in [s,e]" becomes harder to satisfy.
    # Specifically, for each x in [s,e], we need inv[x] < s or inv[x] > e.
    # Since inv[x] != x for x not in {1, p-1}, and we exclude 1 and p-1,
    # the constraint is: for all x in [s,e] \ {1, p-1}, inv[x] < s or inv[x] > e.
    
    # Let's precompute for each x (not 1 or p-1) whether inv[x] < x or inv[x] > x.
    # If inv[x] < x, then for x to be in a valid interval starting at s, we need s > inv[x] (so inv[x] < s).
    # If inv[x] > x, then for x to be in a valid interval ending at e, we need e < inv[x] (so inv[x] > e).
    
    # For a given start s, the maximum valid end e is:
    # e_max = min(inv[x] - 1 for x in [s, e] if inv[x] > x and x not in {1, p-1})
    # But this is circular. Let me think differently.
    
    # For a fixed s, let's find the maximum e such that [s,e] is valid.
    # Start with e = s, and extend while valid.
    # But we can be smarter: for each x >= s, x not in {1,p-1}, if inv[x] > x, then e must be < inv[x].
    # So e_max(s) = min(inv[x] - 1 for x in [s, 1008] if x not in {1,p-1} and inv[x] > x)
    # But we also need to check that for all x in [s, e_max], inv[x] < s or inv[x] > e_max.
    # The condition inv[x] > e_max is guaranteed by construction (e_max < inv[x] for all such x).
    # The condition inv[x] < s needs to be checked for x in [s, e_max] with inv[x] < x.
    
    # So for a fixed s:
    # 1. Compute e_max = min(inv[x] - 1 for x in [s, 1008] if x not in {1,p-1} and inv[x] > x), or 1008 if no such x.
    # 2. Check that for all x in [s, e_max] with x not in {1,p-1} and inv[x] < x, we have inv[x] < s.
    # 3. If valid, the interval [s, e_max] has length e_max - s + 1.
    
    # But wait, we also need s itself to not be 1 or p-1, and similarly for all elements.
    # Since s >= 1 and we're looking at intervals in [1, 1008], and we exclude 1 and 1008 from consideration
    # as elements that can be in the interval, we need s >= 2 and e <= 1007? No, that's not right.
    # The interval [s,e] must not contain 1 or 1008. So if s=1, the interval is invalid.
    # Similarly if e=1008, the interval is invalid.
    
    # Let me reconsider: the interval [s,e] is valid iff:
    # - s >= 2 and e <= 1007 (to exclude 1 and 1008) -- NO, this is wrong.
    #   Actually, 1 and 1008 just can't be IN the interval. So s can be 1 only if... no, if s=1, then 1 is in the interval, which is invalid.
    #   So we need s >= 2? No, s can be 1 only if 1 is not in [s,e], but s=1 means 1 is in [s,e]. So s must be >= 2? 
    #   Wait, the interval is [s,e] inclusive. If s=1, then 1 is in the interval. Since 1*1 % 1009 = 1, this is invalid.
    #   Similarly, if e=1008, then 1008 is in the interval, and 1008*1008 % 1009 = (-1)^2 % 1009 = 1, invalid.
    #   So we need s >= 2 and e <= 1007? No, we need 1 not in [s,e] and 1008 not in [s,e].
    #   So s > 1 or e < 1... but s <= e, so if s=1, 1 is in the interval. So s must be >= 2? 
    #   Actually no: if s=1, the interval contains 1. So for validity, we need s >= 2 (to exclude 1) is wrong.
    #   We need 1 not in [s,e], which means e < 1 or s > 1. Since e >= s >= 1, we need s > 1, i.e., s >= 2.
    #   Similarly, 1008 not in [s,e] means s > 1008 or e < 1008. Since s <= 1008, we need e < 1008, i.e., e <= 1007.
    
    # So valid intervals must have s >= 2 and e <= 1007.
    
    # Let me precompute:
    # For each x in [2, 1007], compute inv[x] mod p.
    # Note that inv[x] is in [1, 1008]. If inv[x] is in [2, 1007], it's a potential conflict.
    # If inv[x] is 1 or 1008, then since 1 and 1008 are not in our interval, there's no conflict from that side.
    
    # For a fixed s (2 <= s <= 1007):
    # e_max = min(inv[x] - 1 for x in [s, 1007] if inv[x] > x and 2 <= inv[x] <= 1007)
    # If no such x, e_max = 1007.
    # Then check: for all x in [s, min(e_max, 1007)] with inv[x] < x and 2 <= inv[x] <= 1007, we need inv[x] < s.
    # If this holds, [s, e_max] is valid with length e_max - s + 1.
    
    # But we should also consider that e_max might be < s, in which case no valid interval starts at s.
    
    best_len = 0
    best_start = 2
    
    # Precompute inv for all x
    for s in range(2, 1008):
        if time.time() > deadline:
            break
        
        # Find e_max: the minimum inv[x]-1 for x in [s, 1007] with inv[x] > x and inv[x] in [2, 1007]
        e_max = 1007
        for x in range(s, 1008):
            if inv[x] > x and 2 <= inv[x] <= 1007:
                if inv[x] - 1 < e_max:
                    e_max = inv[x] - 1
                if e_max < s:
                    break
        
        if e_max < s:
            continue
        
        # Check the other direction: for x in [s, e_max] with inv[x] < x and inv[x] in [2, 1007], need inv[x] < s
        valid = True
        for x in range(s, e_max + 1):
            if inv[x] < x and 2 <= inv[x] <= 1007:
                if inv[x] >= s:
                    valid = False
                    break
        
        if valid:
            length = e_max - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    if best_len == 0:
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
