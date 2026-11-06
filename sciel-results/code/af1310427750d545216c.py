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
    
    best_start = 1
    best_length = 0
    
    # For each starting position, find the maximum valid interval length
    for start in range(1, p):  # 1 to 1008
        if time.time() > deadline:
            break
        
        # Check if start itself is self-inverse
        if start in self_inverse:
            continue
        
        # Extend the interval from start
        # We need to track which elements are in the current interval
        # An interval [start, end] is valid if:
        #   - No element in it is self-inverse
        #   - For no element x in it is inv[x] also in it (and inv[x] != x, since self-inverse already excluded)
        
        # Greedy extension: keep track of a set of elements in the interval
        # But for efficiency, we can just check incrementally
        # When we add element e to [start, e-1], we need to check:
        #   - e is not self-inverse
        #   - inv[e] is not in [start, e-1] (i.e., start <= inv[e] <= e-1)
        
        max_end = start
        valid = True
        for end in range(start, p):  # end from start to 1008
            if end in self_inverse:
                break
            # Check if inv[end] is in [start, end-1]
            ie = inv[end]
            if start <= ie <= end - 1:
                break
            # Also need to check: for any previously added element x in [start, end-1],
            # is inv[x] == end? That's equivalent to: is x == inv[end]?
            # Which is the same check: is inv[end] in [start, end-1]?
            # So the above check is sufficient!
            max_end = end
        
        length = max_end - start + 1
        if length > best_length:
            best_length = length
            best_start = start
    
    if best_length == 0:
        # Fallback: should not happen, but return something valid
        # Find any valid single element
        for x in range(1, p):
            if x not in self_inverse:
                return [x, 1]
        return [1, 1]
    
    return [best_start, best_length]

# EVOLVE-BLOCK-END
