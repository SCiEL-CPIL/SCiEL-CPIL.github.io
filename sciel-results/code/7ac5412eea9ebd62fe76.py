<<<<<<< SEARCH
def remove_redundant_blocks(blocks, v, t):
    # Convert to sets for faster intersection? Or just check coverage.
    # A block is redundant if every triple in it is covered by at least one other block.
    blocks_set = [set(b) for b in blocks]
    
    # To make it efficient, we can maintain a count of how many blocks cover each triple.
    # But blocks are large.
    # Simpler: Iterate through blocks. For each block, check if it can be removed.
    # To check if block B can be removed:
    # For each triple in B, is there another block in blocks (excluding B) that contains that triple?
    
    # This is O(N * 35 * N) which is too slow if N is 100. 100*35*100 = 350,000. This is fine.
    
    removed = True
    while removed:
        removed = False
        new_blocks = []
        for i, b in enumerate(blocks):
            # Check if b is redundant
            redundant = True
            for triple in itertools.combinations(b, t):
                # Check if any other block covers this triple
                covered_by_other = False
                for j, other_b in enumerate(blocks):
                    if i == j:
                        continue
                    if all(p in other_b for p in triple):
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            
            if redundant:
                removed = True
            else:
                new_blocks.append(b)
        
        blocks = new_blocks
        
    return blocks
=======
def remove_redundant_blocks(blocks, v, t):
    # Convert to sets for faster intersection? Or just check coverage.
    # A block is redundant if every triple in it is covered by at least one other block.
    blocks_set = [set(b) for b in blocks]
    
    # To make it efficient, we can maintain a count of how many blocks cover each triple.
    # But blocks are large.
    # Simpler: Iterate through blocks. For each block, check if it can be removed.
    # To check if block B can be removed:
    # For each triple in B, is there another block in blocks (excluding B) that contains that triple?
    
    # This is O(N * 35 * N) which is too slow if N is 100. 100*35*100 = 350,000. This is fine.
    
    removed = True
    while removed:
        removed = False
        new_blocks = []
        for i, b in enumerate(blocks):
            # Check if b is redundant
            redundant = True
            for triple in itertools.combinations(b, t):
                # Check if any other block covers this triple
                covered_by_other = False
                for j, other_b in enumerate(blocks):
                    if i == j:
                        continue
                    if all(p in other_b for p in triple):
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            
            if redundant:
                removed = True
            else:
                new_blocks.append(b)
        
        blocks = new_blocks
        
    return blocks

def local_search(blocks, v, k, t, deadline):
    """Apply local search with swap and merge moves to reduce block count."""
    blocks = [list(b) for b in blocks]
    current_len = len(blocks)
    
    # Precompute triple coverage counts
    def compute_coverage(blocks):
        from collections import Counter
        cov = Counter()
        for b in blocks:
            for triple in itertools.combinations(b, t):
                cov[triple] += 1
        return cov
    
    coverage = compute_coverage(blocks)
    
    # Local search iterations
    improved = True
    max_iterations = 50
    iteration = 0
    
    while improved and iteration < max_iterations:
        if time.time() > deadline:
            break
        improved = False
        iteration += 1
        
        # Try swap moves: replace one point in a block with another
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            b = blocks[i]
            b_set = set(b)
            
            # For each point in block, try replacing with a point not in block
            for pos in range(len(b)):
                old_point = b[pos]
                candidates = [p for p in range(v) if p not in b_set]
                
                # Shuffle candidates for randomization
                random.shuffle(candidates)
                
                for new_point in candidates[:10]:  # Limit to 10 candidates for speed
                    # Create new block
                    new_b = b.copy()
                    new_b[pos] = new_point
                    new_b.sort()
                    
                    # Check if this block is already in use
                    if any(new_b == ob for ob in blocks):
                        continue
                    
                    # Calculate gain: triples newly covered minus triples lost
                    old_triples = set(itertools.combinations(b, t))
                    new_triples = set(itertools.combinations(new_b, t))
                    
                    # Triples that will be uncovered if we make this change
                    lost_triples = old_triples - new_triples
                    gained_triples = new_triples - old_triples
                    
                    # Check if any lost triple would become uncovered
                    valid = True
                    for triple in lost_triples:
                        if coverage[triple] <= 1:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Calculate net gain
                    net_gain = sum(1 for triple in gained_triples if coverage[triple] == 0)
                    
                    if net_gain > 0:
                        # Accept the swap
                        blocks[i] = new_b
                        # Update coverage
                        for triple in old_triples:
                            coverage[triple] -= 1
                        for triple in new_triples:
                            coverage[triple] += 1
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        # Try merge moves: replace two blocks with one if possible
        if not improved:
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(blocks)):
                    if time.time() > deadline:
                        break
                    
                    bi = set(blocks[i])
                    bj = set(blocks[j])
                    
                    # Try to find a single block that covers all triples from both blocks
                    # that are only covered by these two blocks
                    critical_triples = []
                    for triple in itertools.combinations(bi | bj, t):
                        if triple in itertools.combinations(bi, t) or triple in itertools.combinations(bj, t):
                            # Check coverage
                            count = 0
                            for idx, block in enumerate(blocks):
                                if idx == i or idx == j:
                                    continue
                                if all(p in block for p in triple):
                                    count += 1
                            if count == 0:
                                critical_triples.append(triple)
                    
                    if len(critical_triples) > C(k, t):
                        continue
                    
                    # Try to find a block that covers all critical triples
                    all_points = bi | bj
                    if len(all_points) <= k:
                        # Check if we can form a block from these points
                        if len(all_points) == k:
                            new_block = sorted(all_points)
                            # Verify it covers all critical triples
                            if all(all(p in new_block for p in triple) for triple in critical_triples):
                                # Check if this block is already in use
                                if not any(new_block == ob for ob in blocks):
                                    # Remove blocks i and j, add new_block
                                    blocks.pop(j)
                                    blocks.pop(i)
                                    blocks.append(new_block)
                                    
                                    # Recompute coverage
                                    coverage = compute_coverage(blocks)
                                    improved = True
                                    break
                    
                    if improved:
                        break
                
                if improved:
                    break
    
    return blocks
>>>>>>> REPLACE
