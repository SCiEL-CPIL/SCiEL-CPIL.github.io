# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    MOD = 1009
    N = 1008

    # Compute modular inverse for each x in 1..1008
    # inv[x] = x^(MOD-2) % MOD
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Identify self-inverse elements: x where inv[x] == x
    # For prime p, these are 1 and p-1 (1008).
    # The interval cannot contain 1 or 1008.
    # So valid range for endpoints is [2, 1007].
    
    # Map each value v in [2, 1007] to its pair partner.
    # If v has a partner u != v, we add an obstruction.
    # We specifically need to know, for a given upper bound e, 
    # what is the maximum lower bound 'a' that forces the start 's' to be > a.
    
    # Let's create a list `obstructions` where obstructions[e] contains the 'a' values
    # such that (a, e) is a forbidden pair with a < e.
    # Since we are sweeping e downwards, we just need the max 'a' for each 'e'.
    # Actually, we can just store the max 'a' associated with each 'e'.
    
    max_a_for_e = [0] * (MOD + 1)
    
    for x in range(2, 1008):  # 2 to 1007 inclusive
        ix = inv[x]
        if ix == x:
            continue
        a = min(x, ix)
        b = max(x, ix)
        
        # Check if both are in valid range [2, 1007]
        if a >= 2 and b <= 1007:
            # Update max_a_for_e[b]
            if a > max_a_for_e[b]:
                max_a_for_e[b] = a

    # Now sweep e from 1007 down to 2
    # e_max starts at 1007, e_min is 2
    current_max_a = 0
    best_start = 2
    best_len = 0
    
    # We iterate e from 1007 down to 2.
    # At each step, we add the obstruction from e itself.
    # current_max_a holds the max 'a' for all pairs (a, b) where b >= current_e?
    # No, let's be precise.
    # If the interval is [s, e], it is valid if for all x in [s, e], inv[x] is NOT in [s, e].
    # This implies that if x is in [s, e] and inv[x] = y, then y < s or y > e.
    # Since we are looking for contiguous intervals, and we fix e,
    # any x in [s, e] whose inverse y is also in [s, e] is bad.
    # If y > x, then y is in [x+1, e]. So if y <= e, we must have x >= s and y >= s.
    # Wait, the condition is: for any pair (a, b) with a < b, we cannot have both a and b in [s, e].
    # If b <= e, then to avoid having b in [s, e], we must have s > b? No, if b is in [s, e], then a must NOT be in [s, e].
    # So if b <= e, and a < b, then either a < s or a > e. Since a < b <= e, a cannot be > e.
    # So we must have a < s.
    # Therefore, for a fixed e, s must be greater than ALL 'a' such that (a, b) is a pair and b <= e.
    # So s_min(e) = max(2, max{ a | (a, b) is pair, b <= e } + 1).
    
    # We sweep e from 1007 down to 2.
    # We maintain a running maximum of 'a' for all pairs (a, b) where b >= current_e?
    # No, the condition is b <= e.
    # So if we start at e=1007 and go down to e=2:
    # At e=1007, we consider all pairs with b <= 1007.
    # At e=1006, we consider all pairs with b <= 1006.
    # This means as e decreases, the set of pairs { (a,b) | b <= e } SHRINKS.
    # So the max 'a' might decrease. This is hard to update incrementally downwards.
    
    # Alternative: Sweep e from 2 UP to 1007.
    # As e increases, the set of pairs { (a,b) | b <= e } GROWS.
    # We can maintain `current_max_a` which is the max 'a' for all pairs seen so far (b <= e).
    # Then s_min = max(2, current_max_a + 1).
    # Length = e - s_min + 1.
    
    current_max_a = 0
    best_start = 2
    best_len = 0
    
    for e in range(2, 1008):
        # Add obstruction from pairs with b == e
        if max_a_for_e[e] > current_max_a:
            current_max_a = max_a_for_e[e]
            
        s_min = max(2, current_max_a + 1)
        
        if s_min <= e:
            length = e - s_min + 1
            if length > best_len:
                best_len = length
                best_start = s_min
        else:
            # No valid interval ending at e? Or length 0.
            # Since s_min > e, length would be negative or 0.
            pass

    # If no interval found, return a trivial one (should not happen for N=1008)
    if best_len == 0:
        return [2, 1]
        
    return [best_start, best_len]

# EVOLVE-BLOCK-END
