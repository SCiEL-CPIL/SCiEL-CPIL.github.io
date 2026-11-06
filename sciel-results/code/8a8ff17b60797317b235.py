import time
from itertools import combinations

def construct():
    # Overall deadline: 145 seconds (5s safety margin before 150s kill)
    start_time = time.time()
    overall_deadline = start_time + 145

    # Construct S(5,6,12) using PGL(2,11) action on 12 points
    # Points: 0..10 and 11 (infinity)
    p = 11

    # Precompute modular inverses for 0..p-1
    mod_inv = [0] * p
    for i in range(1, p):
        mod_inv[i] = pow(i, p - 2, p)

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
                        inv_a = mod_inv[a]
                        a_n = 1
                        b_n = (b * inv_a) % p
                        c_n = (c * inv_a) % p
                        d_n = (d * inv_a) % p
                    else:
                        # a == 0, so c != 0
                        inv_c = mod_inv[c]
                        a_n = 0
                        b_n = (b * inv_c) % p
                        c_n = 1
                        d_n = (d * inv_c) % p
                    transformations.append((a_n, b_n, c_n, d_n))

    # Base block for S(5,6,12) under PGL(2,11)
    # This is a known base block for the Witt design
    base_block = [0, 1, 2, 4, 5, 9]

    blocks = set()

    for t in transformations:
        a, b, c, d = t
        transformed_block = set()
        for x in base_block:
            if c == 0:
                # x -> (a*x + b)/d
                if d == 0:
                    # Should not happen if det != 0 and c=0 implies ad!=0 so d!=0
                    y = 11
                else:
                    y = ((a * x + b) * mod_inv[d]) % p
            else:
                denom = (c * x + d) % p
                if denom == 0:
                    y = 11
                else:
                    numer = (a * x + b) % p
                    y = (numer * mod_inv[denom]) % p
            transformed_block.add(y)
        blocks.add(frozenset(transformed_block))

    # Check if we got the full design
    if len(blocks) == 132:
        result = [list(b) for b in blocks]
        
        # Verification: Check if it's a valid packing of 6-sets
        # 1. Distinctness is guaranteed by set
        # 2. No repeated 5-subset
        five_subs = set()
        valid = True
        for b in result:
            for sub in combinations(b, 5):
                if sub in five_subs:
                    valid = False
                    break
                five_subs.add(sub)
            if not valid:
                break

        if valid and len(result) == 132:
            return result

    # Fallback: If the group generation failed or produced an invalid packing,
    # try a different known construction or hardcode if possible.
    # For now, return the greedy packing as a safe fallback (score < 1.0 but valid)
    return _greedy_packing()

def _greedy_packing():
    """Greedy construction of 6-subsets with no repeated 5-subset."""
    points = list(range(12))
    all_5subs = set(combinations(points, 5))
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
