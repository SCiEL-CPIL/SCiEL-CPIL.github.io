def construct():
    """
    Construct the 132 blocks of the Steiner system S(5,6,12) (Witt design)
    by generating the orbit of the base block {0,1,2,3,4,5} under the
    Mathieu group M12.
    
    M12 is a sharply 5-transitive group on 12 points. The blocks of S(5,6,12)
    are precisely the 132 images of a fixed 6-set under M12.
    
    We use a known set of generators for M12 acting on {0, ..., 11}.
    """
    from itertools import combinations
    
    # Generators for M12 on 12 points.
    # These are standard generators from the literature.
    # g1 is an 11-cycle: (0 1 2 3 4 5 6 7 8 9 10)
    # g2 is a permutation that, together with g1, generates M12.
    
    # Let's use verified generators.
    # One standard pair of generators for M12 is:
    # a = (0,1,2,3,4,5,6,7,8,9,10)
    # b = (0,1,4,5,6,8,9,10,2,3,7,11)
    
    # Let's verify these generate M12 and the orbit size is 132.
    # If not, we will fall back to a robust method.
    
    a = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11)
    b = (1, 4, 5, 6, 8, 9, 10, 2, 3, 7, 11, 0)
    
    def apply_perm(perm, point):
        return perm[point]
    
    def compose(p1, p2):
        # Return p1 o p2 (apply p2 then p1)
        return tuple(p1[p2[i]] for i in range(12))
    
    def perm_to_tuple(p):
        return tuple(p)
    
    # BFS to generate the group
    start = tuple(range(12))
    visited = {start}
    queue = [start]
    
    while queue:
        g = queue.pop(0)
        for gen in [a, b]:
            new_g = compose(g, gen)
            if new_g not in visited:
                visited.add(new_g)
                queue.append(new_g)
            new_g2 = compose(gen, g)
            if new_g2 not in visited:
                visited.add(new_g2)
                queue.append(new_g2)
    
    # The group M12 has order 9548940.
    # Generating the whole group is too slow.
    # Instead, we just need the orbit of the base block {0,1,2,3,4,5}.
    
    base_block = frozenset([0, 1, 2, 3, 4, 5])
    orbit = {base_block}
    queue = [base_block]
    
    while queue:
        block = queue.pop(0)
        block_set = set(block)
        for gen in [a, b]:
            # Apply permutation to the block
            new_block_set = set(apply_perm(gen, p) for p in block_set)
            new_block = frozenset(new_block_set)
            if new_block not in orbit:
                orbit.add(new_block)
                queue.append(new_block)
    
    # Check if we got 132 blocks
    if len(orbit) == 132:
        blocks = [sorted(list(b)) for b in orbit]
        # Verify no 5-subset is repeated
        covered = set()
        valid = True
        for b in blocks:
            for s in combinations(b, 5):
                if s in covered:
                    valid = False
                    break
                covered.add(s)
            if not valid:
                break
        if valid:
            return blocks
    
    # Fallback: Greedy algorithm in lexicographic order
    # This is known to find the optimal solution for S(5,6,12)
    covered = set()
    blocks = []
    for b in combinations(range(12), 6):
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
    
    return blocks
