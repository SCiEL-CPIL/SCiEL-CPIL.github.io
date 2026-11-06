"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START

def construct():
    p = 1009
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Precompute modular inverses for all values 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x such that x*x % p == 1
    # These are x=1 and x=p-1=1008
    self_inverse = set()
    for x in range(1, p):
        if (x * x) % p == 1:
            self_inverse.add(x)
    
    # For each starting position, find the maximum valid interval length
    # An interval [start, start+length-1] is valid if:
    # 1. No element x in the interval has x in self_inverse
    # 2. No pair x,y in the interval has x*y % p == 1 (i.e., inv[x] is in the interval)
    
    best_start = 2
    best_length = 1
    
    # Brute force: for each start, extend as far as possible
    for start in range(1, p):  # start from 1 to 1008
        if start in self_inverse:
            continue
        # Try to extend the interval
        present = set()
        length = 0
        valid = True
        for x in range(start, p):  # x from start to 1008
            if x in self_inverse:
                valid = False
                break
            # Check if inv[x] is already in present
            if inv[x] in present:
                valid = False
                break
            # Also need to check: for all y in present, x*y % p != 1
            # This is equivalent to inv[x] not in present (already checked)
            present.add(x)
            length += 1
        
        if valid and length > best_length:
            best_length = length
            best_start = start
        
        # Time check
        if time.time() > deadline - 5:
            break
    
    # Verify the result
    if best_length < 1:
        # Fallback: find any valid single element
        for x in range(1, p):
            if x not in self_inverse:
                best_start = x
                best_length = 1
                break
    
    return [best_start, best_length]

# EVOLVE-BLOCK-END
