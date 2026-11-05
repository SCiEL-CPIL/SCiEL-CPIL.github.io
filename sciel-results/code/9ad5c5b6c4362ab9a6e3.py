<<<<<<< SEARCH
    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        # Iterate pairs
        # To speed up, we can randomize order or just iterate
        for i in range(n):
            if time.time() > deadline:
                return blocks, False
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return blocks, False
                
                b1 = blocks[i]
                b2 = blocks[j]
                
                # Determine U_need: triplets that will become uncovered if b1, b2 removed
                # A triplet T is in U_need if count[T] == (1 if T in b1 else 0) + (1 if T in b2 else 0)
                # AND count[T] > 0 (it must be covered currently)
                
                u_need = []
                # Check triplets in b1
                for triplet in itertools.combinations(b1, t):
                    tid = get_id(triplet)
                    c = count[tid]
                    # Subtract coverage from b1
                    c_rem = c - 1
                    # Subtract coverage from b2 if triplet in b2
                    if triplet in [tuple(sorted(x)) for x in itertools.combinations(b2, t)]:
                         # This check is slow. Better: precompute set of triplet IDs for b2?
                         pass
                
                # Optimization: Precompute set of triplet IDs for each block?
                # Or just check membership?
                # len(combinations(7,3)) = 35.
                # Checking membership in set is O(1).
                
                # Let's create a set of IDs for b1 and b2 on the fly for this pair check?
                # It's small.
                ids_b1 = set()
                for triplet in itertools.combinations(b1, t):
                    ids_b1.add(get_id(triplet))
                    
                ids_b2 = set()
                for triplet in itertools.combinations(b2, t):
                    ids_b2.add(get_id(triplet))
                
                # Union of IDs covered by b1 and b2
                # We need to check for each ID in union(b1, b2)
                # Actually, just iterate through ids_b1 and ids_b2
                check_ids = ids_b1 | ids_b2
                
                for tid in check_ids:
                    c = count[tid]
                    covered_by_b1 = 1 if tid in ids_b1 else 0
                    covered_by_b2 = 1 if tid in ids_b2 else 0
                    if c == covered_by_b1 + covered_by_b2:
                        # This triplet is ONLY covered by b1 and/or b2
                        # So it must be covered by the new block
                        u_need.append(all_triplets[tid])
                
                if not u_need:
                    # Both are redundant together?
                    # If u_need is empty, it means S \ {b1, b2} covers everything.
                    # So we can remove BOTH.
                    # Update counts
                    update_counts([b1, b2], -1)
                    # Remove from list
                    # Be careful with indices
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
                
                # Check if a single block can cover u_need
                # Union of points in u_need
                p_need = set()
                for triplet in u_need:
                    p_need.update(triplet)
                
                if len(p_need) <= k:
                    # We can form a block containing p_need
                    # Construct new block
                    new_block = list(p_need)
                    # Fill with random/other points to reach size k
                    while len(new_block) < k:
                        # Pick a point not in new_block
                        # Using random to vary
                        p = random.randint(0, v-1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    # Replace b1, b2 with new_block
                    # Update counts
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    # Update list
                    new_blocks = []
                    for idx, b in enumerate(blocks):
                        if idx != i and idx != j:
                            new_blocks.append(b)
                    new_blocks.append(new_block)
                    
                    return new_blocks, True

        return blocks, False
=======
    def get_block_ids(block):
        """Get set of triplet IDs covered by a block."""
        ids = set()
        for triplet in itertools.combinations(block, t):
            ids.add(get_id(triplet))
        return ids

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
        
        # Precompute IDs for all blocks
        block_ids = [get_block_ids(b) for b in blocks]
        
        # Iterate pairs
        for i in range(n):
            if time.time() > deadline:
                return blocks, False
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return blocks, False
                
                b1 = blocks[i]
                b2 = blocks[j]
                ids_b1 = block_ids[i]
                ids_b2 = block_ids[j]
                
                # Determine U_need: triplets that will become uncovered if b1, b2 removed
                u_need = []
                check_ids = ids_b1 | ids_b2
                
                for tid in check_ids:
                    c = count[tid]
                    covered_by_b1 = 1 if tid in ids_b1 else 0
                    covered_by_b2 = 1 if tid in ids_b2 else 0
                    if c == covered_by_b1 + covered_by_b2:
                        # This triplet is ONLY covered by b1 and/or b2
                        u_need.append(all_triplets[tid])
                
                if not u_need:
                    # Both are redundant together
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
                
                # Check if a single block can cover u_need
                p_need = set()
                for triplet in u_need:
                    p_need.update(triplet)
                
                if len(p_need) <= k:
                    # We can form a block containing p_need
                    new_block = list(p_need)
                    while len(new_block) < k:
                        p = random.randint(0, v-1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    # Replace b1, b2 with new_block
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = []
                    for idx, b in enumerate(blocks):
                        if idx != i and idx != j:
                            new_blocks.append(b)
                    new_blocks.append(new_block)
                    
                    return new_blocks, True

        return blocks, False

    def try_swap_1_for_1(blocks):
        """Try to replace one block with a better one that covers more uncovered triplets."""
        n = len(blocks)
        if n == 0:
            return blocks, False
        
        # For each block, try to find a replacement that covers at least as many
        # triplets while allowing removal of the original
        for i in range(n):
            if time.time() > deadline:
                return blocks, False
            
            b = blocks[i]
            ids_b = get_block_ids(b)
            
            # Find triplets that would become uncovered if we remove b
            u_need = []
            for tid in ids_b:
                if count[tid] == 1:
                    u_need.append(all_triplets[tid])
            
            if not u_need:
                # b is redundant, remove it
                update_counts([b], -1)
                return [blk for idx, blk in enumerate(blocks) if idx != i], True
            
            # Try to construct a new block that covers u_need
            p_need = set()
            for triplet in u_need:
                p_need.update(triplet)
            
            if len(p_need) > k:
                continue
            
            # Try several random completions
            for attempt in range(20):
                new_block = list(p_need)
                while len(new_block) < k:
                    p = random.randint(0, v-1)
                    if p not in new_block:
                        new_block.append(p)
                new_block.sort()
                
                # Check if new_block covers all of u_need
                ids_new = get_block_ids(new_block)
                if all(get_id(triplet) in ids_new for triplet in u_need):
                    # Replace b with new_block
                    update_counts([b], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = []
                    for idx, blk in enumerate(blocks):
                        if idx != i:
                            new_blocks.append(blk)
                    new_blocks.append(new_block)
                    
                    return new_blocks, True
        
        return blocks, False
>>>>>>> REPLACE

