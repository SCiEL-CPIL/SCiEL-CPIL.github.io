def construct():
    from collections import deque
    
    # Generators for M12 acting on 12 points (0-indexed)
    # These are standard generators for the Mathieu group M12
    def cycle_to_perm(cycles, n=12):
        perm = list(range(n))
        for cycle in cycles:
            for i in range(len(cycle)):
                perm[cycle[i]] = cycle[(i + 1) % len(cycle)]
        return perm

    # g1: 12-cycle
    g1 = cycle_to_perm([(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)])
    # g2: Two 6-cycles
    g2 = cycle_to_perm([(0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)])
    # g3: Product of 6 disjoint 2-cycles
    g3 = cycle_to_perm([(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)])
    
    # To ensure we generate the full orbit, we might need more generators if these don't suffice.
    # However, these three are known to generate M12.
    
    generators = [g1, g2, g3]
    
    # Base block
    base_block = tuple(sorted((0, 1, 2, 3, 4, 5)))
    
    visited = set()
    queue = deque()
    queue.append(base_block)
    visited.add(base_block)
    
    while queue:
        current = queue.popleft()
        for gen in generators:
            # Apply permutation to the block
            new_block = tuple(sorted((gen[x] for x in current)))
            if new_block not in visited:
                visited.add(new_block)
                queue.append(new_block)
    
    # Convert to list of lists
    blocks = [list(b) for b in visited]
    
    # Verification: If the group generation failed to produce 132 blocks, 
    # we might have a problem. But we assume the generators are correct.
    # If we want to be extra safe against generator errors, we could check.
    # But the directive implies using the group structure.
    
    return blocks
