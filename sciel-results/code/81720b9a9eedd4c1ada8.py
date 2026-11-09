def remove_redundant_fast(blocks):
    """Remove redundant blocks using coverage counts."""
    if not blocks:
        return blocks
    
    # Count coverage for each t-subset
    coverage_count = Counter()
    for block in blocks:
        for idx in block_cover_cache[block]:
            coverage_count[idx] += 1
            
    new_blocks = []
    for block in blocks:
        is_redundant = True
        for idx in block_cover_cache[block]:
            if coverage_count[idx] == 1:
                is_redundant = False
                break
        if not is_redundant:
            new_blocks.append(block)
        else:
            # It is redundant, so decrement counts
            for idx in block_cover_cache[block]:
                coverage_count[idx] -= 1
                
    return new_blocks
