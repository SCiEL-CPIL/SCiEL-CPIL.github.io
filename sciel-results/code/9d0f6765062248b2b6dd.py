"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    p = 1009
    n = p - 1  # 1008
    
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
    
    # Build list of inverse pairs (a, b) with a < b and a*b % p == 1
    pairs = []
    for a in range(1, p):
        b = inv[a]
        if b > a:
            pairs.append((a, b))
    
    # For each start s, the maximum valid end e is determined by:
    # 1. No self-inverse in [s, e]
    # 2. No pair (a, b) with s <= a < b <= e
    #
    # The interval [s, e] is invalid if there exists a pair (a, b) with
    # s <= min(a,b) and max(a,b) <= e.
    # So for start s, the blocking end is min over all pairs (a,b) with
    # min(a,b) >= s of (max(a,b) - 1), and min over self-inverse x >= s of (x-1).
    #
    # Let's precompute efficiently:
    # For each position i in [1, n], define:
    #   conflict_at[i] = the maximum value M such that there exists a pair
    #   (a, b) with min(a,b) = i and max(a,b) = M, OR i is self-inverse (M = i).
    #   Actually, we need: for start s, e_max = min over all i >= s of (conflict_end[i] - 1)
    #   where conflict_end[i] is the max element of any pair whose min element is i,
    #   or i itself if i is self-inverse.
    
    # Let's build an array: for each i in [1, n], what is the "blocking max" for pairs starting at i?
    # pair_max_at[i] = max(b) for all pairs (a,b) with a = i (and a < b)
    # If i is self-inverse, it blocks at i itself.
    
    pair_max_at = [0] * (p + 1)  # index 1..p
    for a, b in pairs:
        # a < b, so the pair blocks when interval reaches b
        if b > pair_max_at[a]:
            pair_max_at[a] = b
    
    # For self-inverse, the element blocks the interval if it's included
    # So for start s, if s is self-inverse, no valid interval starts there.
    # If s < self_inverse_element, the interval can't extend past self_inverse_element - 1.
    
    # Now, for start s, the maximum valid end is:
    #   e_max = min(
    #       min { pair_max_at[i] : i >= s, pair_max_at[i] > 0 } - 1,
    #       min { x : x >= s, x in self_inverse } - 1,
    #       n
    #   )
    # But we need to be careful: pair_max_at[i] is the max of the pair where min is i.
    # The pair (i, pair_max_at[i]) is in [s, e] iff s <= i and pair_max_at[i] <= e.
    # So e must be < pair_max_at[i] for all i >= s where pair_max_at[i] > 0.
    
    # Precompute suffix minimums
    # suffix_pair_min[s] = min { pair_max_at[i] : i >= s, pair_max_at[i] > 0 }
    # suffix_self_min[s] = min { x : x >= s, x in self_inverse }
    
    suffix_pair_min = [p + 1] * (p + 2)  # p+1 means "no blocking pair"
    suffix_self_min = [p + 1] * (p + 2)
    
    # Initialize
    for i in range(1, p):
        if pair_max_at[i] > 0:
            suffix_pair_min[i] = pair_max_at[i]
        if i in self_inverse:
            suffix_self_min[i] = i
    
    # Compute suffix minimums from right to left
    for s in range(p - 1, 0, -1):
        if suffix_pair_min[s + 1] < suffix_pair_min[s]:
            suffix_pair_min[s] = suffix_pair_min[s + 1]
        if suffix_self_min[s + 1] < suffix_self_min[s]:
            suffix_self_min[s] = suffix_self_min[s + 1]
    
    best_start = 2
    best_length = 0
    
    for s in range(1, p):
        if time.time() > deadline:
            break
        
        # s must not be self-inverse
        if s in self_inverse:
            continue
        
        # The blocking end from pairs: suffix_pair_min[s] - 1
        # The blocking end from self-inverse: suffix_self_min[s] - 1
        block_end = min(suffix_pair_min[s], suffix_self_min[s])
        e_max = block_end - 1
        
        # Also cap at n
        if e_max > n:
            e_max = n
        
        if e_max < s:
            continue
        
        length = e_max - s + 1
        if length > best_length:
            best_length = length
            best_start = s
    
    if best_length == 0:
        # Fallback: find any valid single element
        for x in range(1, p):
            if x not in self_inverse:
                return [x, 1]
        return [1, 1]
    
    return [best_start, best_length]

# EVOLVE-BLOCK-END
