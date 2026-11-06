"""Rewritten program for the suite task; optimized Witt design construction."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

# Pre-computed or efficiently generated PSL(2,11) elements
# We generate them on the fly but optimize the inner loop.

def _get_psl2_11_perms():
    """Generate all 660 permutations of PSL(2,11) on 12 points."""
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, 9, 11)
    
    perms = []
    seen = set()
    # Iterate over matrices with determinant 1
    for a in range(11):
        for b in range(11):
            for c in range(11):
                # Solve for d: ad - bc = 1 => ad = 1 + bc => d = (1+bc) * a^-1
                # If a == 0, then -bc = 1 => bc = -1.
                # Case 1: a != 0
                if a != 0:
                    a_inv = inv[a]
                    for d_candidate in range(11):
                        # We can just calculate d directly
                        pass # We will iterate d and check determinant to be safe and simple
                    for d in range(11):
                        if (a * d - b * c) % 11 == 1:
                            perm = [0] * 12
                            for x in range(11):
                                num = (a * x + b) % 11
                                den = (c * x + d) % 11
                                if den == 0:
                                    perm[x] = 11
                                else:
                                    perm[x] = (num * inv[den]) % 11
                            if c == 0:
                                perm[11] = 11
                            else:
                                perm[11] = (a * inv[c]) % 11
                            t = tuple(perm)
                            if t not in seen:
                                seen.add(t)
                                perms.append(t)
                else:
                    # a == 0 => -bc = 1 => bc = 10
                    # b and c must be non-zero
                    if b == 0 or c == 0:
                        continue
                    if (b * c) % 11 == 10:
                        for d in range(11):
                            # determinant is 1
                            perm = [0] * 12
                            for x in range(11):
                                num = b % 11
                                den = (c * x + d) % 11
                                if den == 0:
                                    perm[x] = 11
                                else:
                                    perm[x] = (num * inv[den]) % 11
                            perm[11] = (0 * inv[c]) % 11 if c != 0 else 11 # a=0, so 0/c = 0
                            # Wait, if a=0, x=infty -> 0/c = 0.
                            # If c=0, a=0 is invalid (det 0).
                            perm[11] = 0
                            t = tuple(perm)
                            if t not in seen:
                                seen.add(t)
                                perms.append(t)
    return perms

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # 1. Generate PSL(2,11) permutations
    perms = _get_psl2_11_perms()
    
    # 2. Define base blocks candidates. 
    # The block (0,1,2,3,4,5) is known to work for S(5,6,12) under PSL(2,11) if the orbit size is 132.
    base_candidates = [
        (0, 1, 2, 3, 4, 5),
        (0, 1, 3, 4, 5, 9),
        (0, 1, 2, 4, 5, 8),
    ]
    
    orbit = set()
    valid_base = None
    
    for base in base_candidates:
        orbit = set()
        for p in perms:
            # Apply permutation
            new_block = tuple(sorted(p[base[i]] for i in range(6)))
            orbit.add(new_block)
        
        if len(orbit) == 132:
            valid_base = base
            break
    
    if valid_base is not None:
        # Verify no 5-subset is repeated
        seen_fives = set()
        is_valid = True
        for block in orbit:
            for i in range(6):
                sub = block[:i] + block[i+1:]
                if sub in seen_fives:
                    is_valid = False
                    break
                seen_fives.add(sub)
            if not is_valid:
                break
        
        if is_valid and len(seen_fives) == 792:
            return [list(b) for b in orbit]
    
    # Fallback: Greedy packing if algebraic construction fails (should not happen)
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
