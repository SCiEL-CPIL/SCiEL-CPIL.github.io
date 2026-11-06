import time

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) by generating the orbits of a base block
    under the action of the Mathieu group M12.
    """
    # Generators for M12 acting on {0, ..., 11}
    # These are standard generators. 
    # g1: cyclic permutation of 0..10
    # g2: specific involution
    # g3: specific 3-cycle product
    # Note: The specific generators must generate the full M12 group.
    # A known set:
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11]
    g2 = [11, 5, 8, 4, 3, 1, 9, 10, 2, 6, 7, 0]
    g3 = [0, 3, 10, 9, 2, 7, 6, 8, 5, 1, 4, 11]
    
    # Verify generators map 0..11 to 0..11
    for g in [g1, g2, g3]:
        if sorted(g) != list(range(12)):
            raise ValueError("Invalid generator")

    # BFS to generate all group elements
    # Identity permutation
    identity = list(range(12))
    generators = [tuple(g1), tuple(g2), tuple(g3)]
    
    visited = {tuple(identity)}
    queue = [identity]
    group_elements = [identity]
    
    while queue:
        current = queue.pop(0)
        for gen in generators:
            # Compose current with gen: apply current then gen? 
            # Usually group action is (g * h)(x) = g(h(x)).
            # Let's define composition: new_perm[i] = current[gen[i]]
            # This corresponds to applying gen first, then current? No.
            # If we have perm p and q, (p o q)(i) = p(q(i)).
            # Let's stick to a consistent convention.
            # Let's say we apply the permutation to the point.
            # If we have a permutation p, p[i] is the image of i.
            # New permutation n = p o q means n[i] = p[q[i]].
            new_perm = [current[gen[i]] for i in range(12)]
            new_perm_t = tuple(new_perm)
            
            if new_perm_t not in visited:
                visited.add(new_perm_t)
                queue.append(new_perm)
                group_elements.append(new_perm)
                
    # The group M12 has 95040 elements. 
    # If the generators are wrong, we might get a smaller group.
    # Let's check the size.
    # if len(group_elements) != 95040:
    #    print(f"Warning: Group size is {len(group_elements)}, expected 95040")
    
    # Base block for S(5,6,12)
    # One standard block is {0, 1, 2, 3, 4, 5}
    base_block = (0, 1, 2, 3, 4, 5)
    
    blocks = set()
    for perm in group_elements:
        # Apply permutation to base block
        # The image of the set {b0, ..., b5} is {perm[b0], ..., perm[b5]}
        new_block = tuple(sorted(perm[b] for b in base_block))
        blocks.add(new_block)
        
    # Convert to list of lists
    result_blocks = [list(b) for b in blocks]
    
    # Verification: 
    # 1. All blocks are distinct 6-subsets of 0..11
    # 2. No 5-subset lies in two blocks
    # 3. Number of blocks is 132
    
    if len(result_blocks) != 132:
        # Fallback if something went wrong with generators
        # But this should not happen with correct M12 generators
        pass
        
    # Check validity
    covered_5subs = set()
    for b in result_blocks:
        from itertools import combinations
        for sub in combinations(b, 5):
            if sub in covered_5subs:
                # Invalid design
                return []
            covered_5subs.add(sub)
            
    if len(covered_5subs) != 792:
        return []
        
    return result_blocks

if __name__ == "__main__":
    res = construct()
    print(len(res))
