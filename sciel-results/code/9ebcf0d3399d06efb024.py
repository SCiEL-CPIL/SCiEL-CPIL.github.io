import time

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    from itertools import combinations
    
    def apply_perm(p, x):
        return p[x]

    def compose(p1, p2):
        return [p1[p2[i]] for i in range(12)]

    g1 = [(i+1)%12 for i in range(12)]
    g2 = [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6]
    g3 = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
    
    gens = [g1, g2, g3]
    
    identity = list(range(12))
    group = {tuple(identity): identity}
    queue = [identity]
    head = 0
    
    while head < len(queue):
        if time.time() > deadline:
            break
        g = queue[head]
        head += 1
        for gen in gens:
            new_g = compose(g, gen)
            key = tuple(new_g)
            if key not in group:
                group[key] = new_g
                queue.append(new_g)
            new_g2 = compose(gen, g)
            key2 = tuple(new_g2)
            if key2 not in group:
                group[key2] = new_g2
                queue.append(new_g2)
                
    base_block = frozenset([0, 1, 2, 3, 4, 5])
    
    blocks = set()
    for g in group.values():
        new_block = frozenset([g[x] for x in base_block])
        blocks.add(new_block)
        
    result = [list(b) for b in blocks]
    result.sort()
    
    return result
