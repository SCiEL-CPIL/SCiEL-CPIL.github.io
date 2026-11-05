import time

def construct():
    start_time = time.time()
    
    # Generators for M12
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    g2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1]
    g3 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2]
    
    generators = [g1, g2, g3]
    
    def apply_perm(perm, block):
        return tuple(perm[x] for x in block)
    
    base_block = (0, 1, 2, 3, 4, 5)
    orbit = {base_block}
    frontier = [base_block]
    
    while frontier:
        if time.time() - start_time > 110:
            break
        current = frontier.pop()
        for g in generators:
            new_block = apply_perm(g, current)
            if new_block not in orbit:
                orbit.add(new_block)
                frontier.append(new_block)
                
    result = [list(b) for b in orbit]
    return result
