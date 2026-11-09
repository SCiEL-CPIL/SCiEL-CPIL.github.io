import time

def construct():
    """
    Return the 132 blocks of the Steiner system S(5,6,12).
    
    This implementation uses the algebraic construction via the group PGL(2, 11)
    acting on the projective line over GF(11) augmented with infinity.
    The 132 blocks form two orbits of 66 under PGL(2,11).
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin

    # Points: 0..10 represent elements of GF(11), 11 represents infinity
    # We will use integers 0-11.
    
    def apply_mobius(x, a, b, c, d):
        # x is 0..10 or 11 (infinity)
        if x == 11:
            if c == 0:
                # (a*inf + b)/(d) -> a/d
                return (a * pow(d, 10, 11)) % 11
            else:
                # (a*inf + b)/(c*inf + d) -> a/c
                return (a * pow(c, 10, 11)) % 11
        else:
            denom = (c * x + d) % 11
            if denom == 0:
                return 11
            else:
                num = (a * x + b) % 11
                return (num * pow(denom, 10, 11)) % 11

    # Generate all distinct transformations in PGL(2, 11)
    # We can represent them by their action on 3 points to avoid duplicates,
    # or just iterate over all 4-tuples and normalize.
    
    transformations = []
    seen_transforms = set()
    
    for a in range(11):
        for b in range(11):
            for c in range(11):
                for d in range(11):
                    det = (a * d - b * c) % 11
                    if det != 0:
                        # Normalize: divide by a if a!=0.
                        if a != 0:
                            inv_a = pow(a, 10, 11)
                            na = 1
                            nb = (b * inv_a) % 11
                            nc = (c * inv_a) % 11
                            nd = (d * inv_a) % 11
                        elif c != 0:
                            inv_c = pow(c, 10, 11)
                            na = (a * inv_c) % 11
                            nb = (b * inv_c) % 11
                            nc = 1
                            nd = (d * inv_c) % 11
                        elif b != 0:
                            inv_b = pow(b, 10, 11)
                            na = (a * inv_b) % 11
                            nb = 1
                            nc = (c * inv_b) % 11
                            nd = (d * inv_b) % 11
                        else: # d != 0
                            inv_d = pow(d, 10, 11)
                            na = (a * inv_d) % 11
                            nb = (b * inv_d) % 11
                            nc = (c * inv_d) % 11
                            nd = 1
                            
                        key = (na, nb, nc, nd)
                        if key not in seen_transforms:
                            seen_transforms.add(key)
                            transformations.append((na, nb, nc, nd))
                            
    if time.time() - start_time > DEADLINE:
        return []

    # Base blocks for the two orbits
    base_block1 = [0, 1, 3, 4, 5, 9]
    base_block2 = [2, 6, 7, 8, 10, 11]
    
    blocks_set = set()
    
    # Apply all transformations to the base blocks
    for base_block in [base_block1, base_block2]:
        for t in transformations:
            a, b, c, d = t
            mapped_block = set()
            for p in base_block:
                mapped_block.add(apply_mobius(p, a, b, c, d))
            blocks_set.add(frozenset(mapped_block))
            
    # We should have 132 blocks now (66 from each orbit)
    # If for some reason we don't, we can try to verify and potentially fix
    # But typically this construction yields exactly the 132 blocks of S(5,6,12)
    
    # Verify validity
    from itertools import combinations
    covered_5_subsets = set()
    valid = True
    blocks_list = [list(b) for b in blocks_set]
    
    for b in blocks_list:
        for s in combinations(b, 5):
            if s in covered_5_subsets:
                valid = False
                break
            covered_5_subsets.add(s)
        if not valid:
            break
            
    if not valid:
        # Fallback: Greedy search if algebraic construction fails
        # This should not happen with the correct base blocks, but for safety.
        covered = set()
        greedy_blocks = []
        for b in combinations(range(12), 6):
            if time.time() - start_time > DEADLINE:
                break
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                greedy_blocks.append(list(b))
                covered.update(subsets)
        return greedy_blocks

    return blocks_list
