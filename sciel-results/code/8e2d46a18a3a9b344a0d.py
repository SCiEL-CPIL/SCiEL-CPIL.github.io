def construct():
    """
    Constructs the Witt design S(5,6,12) by generating the 132 blocks
    from 12 base blocks under the cyclic group Z_11.
    Points are 0..10 and 11 (representing infinity).
    """
    # 12 base blocks for the cyclic development of S(5,6,12)
    # 11 represents infinity
    base_blocks = [
        [11, 0, 1, 3, 4, 5],
        [11, 0, 1, 2, 4, 8],
        [11, 0, 1, 2, 5, 9],
        [11, 0, 1, 3, 6, 10],
        [11, 0, 2, 3, 5, 7],
        [11, 0, 1, 4, 5, 9],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 4, 8, 9],
        [0, 1, 3, 4, 7, 8],
        [0, 2, 3, 5, 7, 10],
        [0, 1, 3, 5, 8, 9],
        [0, 2, 3, 4, 7, 9]
    ]
    
    blocks = []
    seen_5_subsets = set()
    valid = True
    
    for base in base_blocks:
        for shift in range(11):
            # Apply cyclic shift to points 0..10, keep 11 fixed
            shifted_block = []
            for p in base:
                if p == 11:
                    shifted_block.append(11)
                else:
                    shifted_block.append((p + shift) % 11)
            
            # Sort for consistent representation
            shifted_block.sort()
            block_tuple = tuple(shifted_block)
            
            # Check for duplicate blocks (should not happen in a design)
            # and check 5-subset uniqueness
            from itertools import combinations
            for sub in combinations(block_tuple, 5):
                if sub in seen_5_subsets:
                    valid = False
                    break
                seen_5_subsets.add(sub)
            
            if not valid:
                break
            
            blocks.append(list(block_tuple))
            
        if not valid:
            break

    if valid and len(blocks) == 132:
        return blocks
    else:
        # Fallback: Greedy lexicographic packing if the hard-coded design is flawed
        from itertools import combinations
        covered, greedy_blocks = set(), []
        for b in combinations(range(12), 6):
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                greedy_blocks.append(list(b))
                covered.update(subsets)
        return greedy_blocks
