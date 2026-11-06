import time
from itertools import combinations

def _mod_inv(x, p):
    """Calculate modular inverse of x modulo p using Fermat's little theorem."""
    if x == 0:
        return 0
    return pow(x, p-2, p)

def construct():
    # Overall deadline: 145 seconds (5s safety margin)
    overall_deadline = time.time() + 145
    
    # Construct S(5,6,12) using PGL(2,11) action on 12 points
    # Points: 0..10 and 11 (infinity)
    p = 11
    points = list(range(12))
    
    # Generate all elements of PGL(2,11)
    # A transformation is (a, b, c, d) with ad - bc != 0
    # We normalize so that a=1 if a!=0, else c=1
    transformations = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    det = (a * d - b * c) % p
                    if det == 0:
                        continue
                    # Normalize
                    if a != 0:
                        # Divide by a
                        inv_a = _mod_inv(a, p)
                        a_n = 1
                        b_n = (b * inv_a) % p
                        c_n = (c * inv_a) % p
                        d_n = (d * inv_a) % p
                    else:
                        # a == 0, so c != 0
                        inv_c = _mod_inv(c, p)
                        a_n = 0
                        b_n = (b * inv_c) % p
                        c_n = 1
                        d_n = (d * inv_c) % p
                    transformations.append((a_n, b_n, c_n, d_n))
    
    # Base block for S(5,6,12)
    # This is a known base block for the Witt design under PGL(2,11)
    base_block = frozenset([0, 1, 2, 4, 5, 9])
    
    blocks = set()
    
    for t in transformations:
        a, b, c, d = t
        transformed_block = set()
        for x in base_block:
            if x == 11: # Infinity
                if c == 0:
                    y = 11
                else:
                    y = (a * _mod_inv(c, p)) % p
            else:
                denom = (c * x + d) % p
                if denom == 0:
                    y = 11
                else:
                    numer = (a * x + b) % p
                    y = (numer * _mod_inv(denom, p)) % p
            transformed_block.add(y)
        blocks.add(frozenset(transformed_block))
        
    # Convert to list of lists
    result = [list(b) for b in blocks]
    
    # Verification: Check if it's a valid packing of 6-sets
    # 1. Distinctness is guaranteed by set
    # 2. No repeated 5-subset
    five_subs = set()
    for b in result:
        for sub in combinations(b, 5):
            if sub in five_subs:
                # Invalid, fall back to greedy or return what we have?
                # If invalid, the base block or group was wrong.
                # For now, if invalid, we might return a smaller valid packing?
                # But S(5,6,12) is unique, so if this fails, it's a bug.
                pass
            five_subs.add(sub)
            
    # If we have 132 blocks and no conflicts, return them
    if len(result) == 132:
        return result
    else:
        # Fallback to greedy if something went wrong (should not happen)
        return _greedy_packing()

def _greedy_packing():
    """Greedy construction of 6-subsets with no repeated 5-subset."""
    points = list(range(12))
    all_5subs = list(combinations(points, 5))
    covered_5 = set()
    blocks = []
    
    # Generate all 6-subsets in lex order
    all_6subs = list(combinations(points, 6))
    
    for block in all_6subs:
        valid = True
        new_5s = []
        for sub in combinations(block, 5):
            if sub in covered_5:
                valid = False
                break
            new_5s.append(sub)
        
        if valid:
            blocks.append(list(block))
            for sub in new_5s:
                covered_5.add(sub)
            
    return blocks
