"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    p = 1009
    # Precompute modular inverses for 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Identify self-inverse residues (x*x % p == 1)
    # For prime p, x^2 = 1 mod p => x = 1 or x = p-1 = 1008
    self_inverse = set()
    for x in range(1, p):
        if (x * x) % p == 1:
            self_inverse.add(x)
    
    # Build a list of "obstructions": pairs (a, b) where a != b and a*b % p == 1
    # For an interval to be valid, it must not contain both a and b for any such pair,
    # and must not contain any self-inverse element.
    
    # An interval [s, e] is invalid if:
    # 1. It contains 1 or 1008 (self-inverse)
    # 2. There exists a pair (a, b) with a*b % p == 1, a != b, and both a, b in [s, e]
    
    # For a given start s, the interval can extend to e as long as:
    # - No self-inverse in [s, e]
    # - No pair (a, b) with both in [s, e]
    
    # Efficient approach: for each start, find the minimum "blocking" end.
    # A pair (a, b) with a < b blocks any interval [s, e] where s <= a and b <= e.
    # So for start s, the maximum valid end is:
    #   min over all pairs (a,b) with a >= s of (b - 1)  [if b > a]
    #   min over all self-inverse x >= s of (x - 1)
    #   but also need to consider pairs where a < s <= b: these don't block since a not in interval
    #   Wait, if a < s, then a is not in [s, e], so the pair doesn't block.
    #   Only pairs where both a and b are in [s, e] block.
    #   So for start s, a pair (a, b) with a < b blocks if s <= a and b <= e.
    #   The blocking condition for end e is: exists pair (a,b) with s <= a < b <= e.
    #   So the maximum e for start s is: min over all pairs (a,b) with a >= s of (b-1), 
    #   and also min over self-inverse x >= s of (x-1).
    #   But wait, we also need e >= s, and the interval [s, e] must not have any self-inverse.
    
    # Let's precompute for each position s, the earliest "conflict end" that blocks extension.
    # conflict[s] = minimum b-1 over all pairs (a,b) with a >= s, plus minimum x-1 over self-inverse x >= s.
    # Then the maximum valid interval starting at s is [s, min(conflict[s], p-1)] if s is not self-inverse
    # and conflict[s] >= s.
    
    # Actually, let's think more carefully.
    # For start s, the interval [s, e] is valid iff:
    #   - s is not self-inverse
    #   - No self-inverse in [s, e]
    #   - No pair (a,b) with s <= a < b <= e where a*b % p == 1
    
    # So the maximum e for start s is:
    #   e_max = min(
    #       min {b-1 : exists a >= s, b > a, a*b%p==1},
    #       min {x-1 : x >= s, x self-inverse},
    #       p-1
    #   )
    #   But we also need e_max >= s for the interval to be nonempty.
    
    # Let's precompute:
    # For each pair (a,b) with a < b and a*b%p==1, it blocks all starts s <= a with ends e >= b.
    # So for a given start s, the blocking ends from pairs are: min {b : a >= s, b > a, a*b%p==1} - 1
    # And from self-inverse: min {x : x >= s, x in self_inverse} - 1
    
    # Let's compute this efficiently.
    # Create a list of all pairs (a, b) with a < b and a*b % p == 1.
    # For each a in 1..p-1, b = inv[a]. If b > a, add (a, b).
    
    pairs = []
    for a in range(1, p):
        b = inv[a]
        if b > a:
            pairs.append((a, b))
    
    # Sort pairs by a
    pairs.sort()
    
    # For each start s, we need: min b over pairs with a >= s
    # And min x over self-inverse with x >= s
    
    # Let's build an array min_b_from[s] = min {b : (a,b) in pairs, a >= s}
    # and min_self_from[s] = min {x : x in self_inverse, x >= s}
    
    min_b_from = [p] * (p + 1)  # p+1 size, index 1..p
    min_self_from = [p] * (p + 1)
    
    # Process from right to left
    # For min_b_from: for each s from p down to 1, min_b_from[s] = min(b for pairs with a >= s)
    # We can do this by iterating s from p down to 1, and maintaining a running min.
    
    # Create an array where at index a, we have the min b for pairs starting at a
    min_b_at = [p] * (p + 1)
    for a, b in pairs:
        if b < min_b_at[a]:
            min_b_at[a] = b
    
    # Now compute min_b_from[s] = min(min_b_at[s], min_b_from[s+1])
    for s in range(p, 0, -1):
        min_b_from[s] = min_b_at[s]
        if s < p:
            min_b_from[s] = min(min_b_from[s], min_b_from[s + 1])
    
    # Similarly for self-inverse
    min_self_at = [p] * (p + 1)
    for x in self_inverse:
        min_self_at[x] = x
    
    for s in range(p, 0, -1):
        min_self_from[s] = min_self_at[s]
        if s < p:
            min_self_from[s] = min(min_self_from[s], min_self_from[s + 1])
    
    # Now for each start s, the max valid end is:
    #   e_max = min(min_b_from[s], min_self_from[s]) - 1
    #   But we need e_max >= s for a valid interval.
    #   Also, s itself must not be self-inverse.
    
    best_start = 1
    best_length = 0
    
    for s in range(1, p):
        if time.time() > deadline:
            break
        
        if s in self_inverse:
            continue
        
        # The blocking end from pairs and self-inverse
        block_end = min(min_b_from[s], min_self_from[s])
        e_max = block_end - 1
        
        if e_max < s:
            continue
        
        length = e_max - s + 1
        if length > best_length:
            best_length = length
            best_start = s
    
    if best_length == 0:
        # Fallback: should not happen, but return something valid
        for x in range(1, p):
            if x not in self_inverse:
                return [x, 1]
        return [1, 1]
    
    return [best_start, best_length]

# EVOLVE-BLOCK-END
