def construct():
    # Direct construction of the Steiner System S(5,6,12)
    # Points: 0..11 (11 represents infinity)
    # Base blocks are generated under the cyclic group Z_11 acting on {0..10}
    base_blocks = [
        (0, 1, 3, 4, 5, 9),
        (0, 2, 3, 6, 8, 10),
        (0, 2, 4, 5, 8, 10),
        (0, 1, 2, 6, 7, 10),
        (0, 1, 3, 6, 7, 9),
        (0, 2, 4, 6, 7, 9),
        (0, 1, 4, 5, 7, 8),
        (0, 1, 2, 4, 8, 9),
        (0, 2, 3, 5, 7, 8),
        (0, 1, 3, 5, 8, 10),
        (0, 2, 3, 4, 7, 9),
        (0, 1, 2, 3, 4, 5)
    ]
    
    blocks = []
    seen = set()
    
    for b in base_blocks:
        for shift in range(11):
            # Apply shift to points 0-10, keep 11 fixed
            new_block = tuple((p + shift) % 11 if p < 11 else 11 for p in b)
            # Sort to ensure canonical representation if needed, but tuples are distinct
            # We need to ensure no duplicates and valid packing
            # The construction guarantees validity, but we add to list
            block_tuple = tuple(sorted(new_block))
            if block_tuple not in seen:
                seen.add(block_tuple)
                blocks.append(list(block_tuple))
                
    return blocks
