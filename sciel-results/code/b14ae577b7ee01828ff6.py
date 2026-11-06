"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    from itertools import combinations
    
    n = 34
    random.seed(12345)
    
    triples = list(combinations(range(n), 3))
    triple_index = {}
    for i, (a, b, c) in enumerate(triples):
        triple_index[(a, b, c)] = i
    
    num_triples = len(triples)
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Precompute for each triple, which 4-sets and 5-sets it belongs to
    # A triple (a,b,c) belongs to 4-sets: (a,b,c,x) for x not in {a,b,c}
    # A triple (a,b,c) belongs to 5-sets: (a,b,c,x,y) for x,y not in {a,b,c}, x<y
    
    # For efficiency, track red_count[quad] = number of red edges in that 4-set
    # and blue_count[quint] = number of blue edges in that 5-set
    
    all_quads = list(combinations(range(n), 4))
    all_quints = list(combinations(range(n), 5))
    
    quad_index = {}
    for i, q in enumerate(all_quads):
        quad_index[q] = i
    
    quint_index = {}
    for i, q in enumerate(all_quints):
        quint_index[q] = i
    
    red_count = [0] * len(all_quads)
    blue_count = [0] * len(all_quints)
    
    # For each triple, precompute which quads and quints it's in
    triple_to_quads = [[] for _ in range(num_triples)]
    triple_to_quints = [[] for _ in range(num_triples)]
    
    for i, (a, b, c) in enumerate(triples):
        for x in range(n):
            if x in (a, b, c):
                continue
            quad = tuple(sorted((a, b, c, x)))
            triple_to_quads[i].append(quad_index[quad])
        for x in range(n):
            if x in (a, b, c):
                continue
            for y in range(x+1, n):
                if y in (a, b, c):
                    continue
                quint = tuple(sorted((a, b, c, x, y)))
                triple_to_quints[i].append(quint_index[quint])
    
    # Initialize counts
    for i, (a, b, c) in enumerate(triples):
        if colors[i] == 0:  # red
            for qi in triple_to_quads[i]:
                red_count[qi] += 1
        else:  # blue
            for qi in triple_to_quints[i]:
                blue_count[qi] += 1
    
    def flip(i):
        """Flip color of triple i and update counts."""
        old = colors[i]
        new = 1 - old
        colors[i] = new
        if old == 0:  # was red, now blue
            for qi in triple_to_quads[i]:
                red_count[qi] -= 1
            for qi in triple_to_quints[i]:
                blue_count[qi] += 1
        else:  # was blue, now red
            for qi in triple_to_quints[i]:
                blue_count[qi] -= 1
            for qi in triple_to_quads[i]:
                red_count[qi] += 1
    
    # Local search
    max_iter = 2000
    for iteration in range(max_iter):
        # Find a quad with red_count == 4
        found = False
        for qi, rc in enumerate(red_count):
            if rc == 4:
                quad = all_quads[qi]
                a, b, c, d = quad
                # Find the 4 triples in this quad
                quad_triples = [
                    triple_index[(a,b,c)],
                    triple_index[(a,b,d)],
                    triple_index[(a,c,d)],
                    triple_index[(b,c,d)]
                ]
                # Flip one at random
                flip(quad_triples[random.randint(0, 3)])
                found = True
                break
        
        if found:
            continue
        
        # Find a quint with blue_count == 10
        for qi, bc in enumerate(blue_count):
            if bc == 10:
                quint = all_quints[qi]
                # Find all 10 triples in this quint
                quint_triples = []
                for t in combinations(quint, 3):
                    quint_triples.append(triple_index[t])
                flip(quint_triples[random.randint(0, 9)])
                found = True
                break
        
        if found:
            continue
        
        # No violations
        break
    
    # Verify
    def verify():
        for qi, rc in enumerate(red_count):
            if rc == 4:
                return False
        for qi, bc in enumerate(blue_count):
            if bc == 10:
                return False
        return True
    
    if not verify():
        n_small = 9
        return {'n': n_small, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_small), 3)]}
    
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
