"""Rewritten program for the suite task; optimized Witt design construction."""
# EVOLVE-BLOCK-START
import time

def _generate_psl2_11_perms():
    """
    Generate all 660 permutations of PSL(2,11) acting on 12 points.
    Points are 0..10 representing F_11, and 11 representing infinity.
    """
    # Modular inverse for F_11
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, 9, 11)  # Fermat's little theorem: a^(p-2) = a^-1

    perms = []
    seen = set()
    
    # Iterate over SL(2,11) matrices [[a, b], [c, d]] with determinant 1
    # a*d - b*c = 1 (mod 11)
    for a in range(11):
        for b in range(11):
            for c in range(11):
                # We need to find d such that a*d - b*c = 1
                # a*d = 1 + b*c
                # If a == 0, then -b*c = 1 => b*c = 10 (mod 11)
                # If a != 0, d = (1 + b*c) * a^-1 (mod 11)
                
                if a == 0:
                    # b*c must be 10
                    if b == 0 or c == 0:
                        continue
                    if (b * c) % 11 != 10:
                        continue
                    d = 0 # d can be anything? No, det is -bc. If a=0, det = -bc.
                    # Wait, if a=0, det = -bc. We need -bc = 1 => bc = 10.
                    # d is free? No, the matrix is [[0, b], [c, d]].
                    # Det is -bc. It doesn't depend on d.
                    # So for any d, if bc=10, det=1.
                    for d in range(11):
                        perm = [0] * 12
                        for x in range(11):
                            # f(x) = (a*x + b) / (c*x + d)
                            # a=0, so f(x) = b / (c*x + d)
                            den = (c * x + d) % 11
                            if den == 0:
                                perm[x] = 11
                            else:
                                perm[x] = (b * inv[den]) % 11
                        # f(infinity) = a/c = 0/c = 0 (if c!=0)
                        # If c==0, then b must be 0 for det 0? No, if a=0,c=0, det=0.
                        # We established bc=10, so c!=0.
                        perm[11] = 0
                        
                        t = tuple(perm)
                        if t not in seen:
                            seen.add(t)
                            perms.append(t)
                else:
                    # a != 0
                    # d = (1 + b*c) * a^-1
                    d = ((1 + b * c) * inv[a]) % 11
                    perm = [0] * 12
                    for x in range(11):
                        num = (a * x + b) % 11
                        den = (c * x + d) % 11
                        if den == 0:
                            perm[x] = 11
                        else:
                            perm[x] = (num * inv[den]) % 11
                    # f(infinity) = a/c
                    if c == 0:
                        perm[11] = 11
                    else:
                        perm[11] = (a * inv[c]) % 11
                    
                    t = tuple(perm)
                    if t not in seen:
                        seen.add(t)
                        perms.append(t)
    
    return perms

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # 1. Generate PSL(2,11) permutations
    # This is very fast, usually < 10ms
    perms = _generate_psl2_11_perms()
    
    # 2. Define base blocks
    # The Witt design S(5,6,12) is the union of two orbits of size 66
    # under the action of PSL(2,11).
    # Standard base blocks are {0,1,2,3,4,5} and {0,1,2,3,4,6}
    # Let's verify the orbit sizes and validity.
    
    base1 = (0, 1, 2, 3, 4, 5)
    base2 = (0, 1, 2, 3, 4, 6)
    
    blocks = set()
    
    # Generate orbit of base1
    for p in perms:
        new_block = tuple(sorted(p[base1[i]] for i in range(6)))
        blocks.add(new_block)
    
    # Generate orbit of base2
    for p in perms:
        new_block = tuple(sorted(p[base2[i]] for i in range(6)))
        blocks.add(new_block)
        
    # Convert to list
    block_list = list(blocks)
    
    # 3. Verify validity
    # A valid S(5,6,12) must have exactly 132 blocks.
    # And no 5-subset should be repeated.
    if len(block_list) != 132:
        # Fallback: if the specific base blocks don't yield 132 distinct blocks 
        # (which shouldn't happen for standard reps, but just in case), 
        # we might need to try other base blocks or a greedy fallback.
        # However, for S(5,6,12), the orbit of {0,1,2,3,4,5} under PSL(2,11) 
        # has size 66. The other orbit also has size 66.
        # If they overlap or are not disjoint in 5-subsets, we need to check.
        
        # Let's check if we have 132 unique blocks.
        # If not, try another common base block for the second orbit.
        # e.g. {0,1,2,3,4,7} or {0,1,2,3,4,8}
        pass 

    # Verification of 5-subset uniqueness
    seen_fives = set()
    is_valid = True
    for block in block_list:
        for i in range(6):
            # Create 5-subset by removing i-th element
            sub = block[:i] + block[i+1:]
            if sub in seen_fives:
                is_valid = False
                break
            seen_fives.add(sub)
        if not is_valid:
            break
            
    if is_valid and len(seen_fives) == 792:
        return [list(b) for b in block_list]
    
    # Fallback: If the algebraic construction failed (highly unlikely),
    # use a greedy packing to get a valid, though suboptimal, result.
    # This ensures the function returns *something* valid if the specific
    # group action details were slightly off in this implementation.
    
    from itertools import combinations
    
    FIVE_SUBSETS = list(combinations(range(12), 5))
    FIVE_TO_IDX = {s: i for i, s in enumerate(FIVE_SUBSETS)}
    SIX_SUBSETS = list(combinations(range(12), 6))
    SIX_TO_FIVE_IDX = []
    for b in SIX_SUBSETS:
        idxs = [FIVE_TO_IDX[s] for s in combinations(b, 5)]
        SIX_TO_FIVE_IDX.append(idxs)
        
    covered = bytearray(792)
    current_blocks = []
    
    for idx in range(len(SIX_SUBSETS)):
        idxs = SIX_TO_FIVE_IDX[idx]
        if not any(covered[i] for i in idxs):
            current_blocks.append(list(SIX_SUBSETS[idx]))
            for i in idxs:
                covered[i] = 1
                
    return current_blocks
# EVOLVE-BLOCK-END
