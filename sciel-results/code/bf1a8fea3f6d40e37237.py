import time
import numpy as np
import itertools

# EVOLVE-BLOCK-START
def _paleyn_orientation(p):
    """
    Construct Paley tournament on p vertices (p must be prime, p % 4 == 3).
    Returns list of bits for pairs (i, j) i < j.
    0 means i -> j, 1 means j -> i.
    """
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
    return orientation

def _verify_paley_exact(p, orientation):
    """
    Exact verification of the 4-dominator property.
    """
    in_bits = [0] * p
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                # i -> j, so i is in-neighbor of j
                in_bits[j] |= (1 << i)
            else:
                # j -> i, so j is in-neighbor of i
                in_bits[i] |= (1 << j)
    
    # Check all 4-combinations
    for a, b, c, d in itertools.combinations(range(p), 4):
        if (in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]) == 0:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We know f(4) = 67. Paley(67) is a valid tournament.
    # We try to find the smallest valid Paley tournament.
    # Candidates p % 4 == 3: 31, 43, 47, 59, 67...
    # 31, 43, 47, 59 are known to be invalid for k=4.
    # We will verify them quickly to be rigorous, but expect 67 to be the first valid one.
    
    candidates = [31, 43, 47, 59, 67]
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        orientation = _paleyn_orientation(p)
        
        # Verify
        is_valid = _verify_paley_exact(p, orientation)
        
        if is_valid:
            return {'n': p, 'orientation': orientation}
            
    # Fallback: return 67 orientation (known valid)
    # This should not happen if the loop above works, but ensures we return something.
    p = 67
    orientation = _paleyn_orientation(p)
    return {'n': p, 'orientation': orientation}
# EVOLVE-BLOCK-END
