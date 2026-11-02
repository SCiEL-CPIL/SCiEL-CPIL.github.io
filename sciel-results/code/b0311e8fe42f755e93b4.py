import random

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    all_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_triples.add((i, j, k))
    
    uncovered = set(all_triples)
    terms = []
    
    # Precompute all possible rectangles
    # A rectangle is defined by (u_sup, v_sup, w_sup)
    # u_sup ⊆ {(i,j)}, v_sup ⊆ {(j,k)}, w_sup ⊆ {(k,i)}
    # Covered: {(i,j,k) : (i,j) in u_sup, (j,k) in v_sup, (k,i) in w_sup}
    
    # Generate all possible rectangles with support size <= 2
    all_rects = []
    u_options = list(itertools.chain.from_iterable(
        itertools.combinations([(i,j) for i in range(n) for j in range(m)], s)
        for s in range(1, 3)
    ))
    v_options = list(itertools.chain.from_iterable(
        itertools.combinations([(j,k) for j in range(m) for k in range(p)], s)
        for s in range(1, 3)
    ))
    w_options = list(itertools.chain.from_iterable(
        itertools.combinations([(k,i) for k in range(p) for i in range(n)], s)
        for s in range(1, 3)
    ))
    
    for u_sup in u_options:
        for v_sup in v_options:
            for w_sup in w_options:
                covered = set()
                for (i,j) in u_sup:
                    for (j2,k) in v_sup:
                        if j == j2:
                            for (k2,i2) in w_sup:
                                if k == k2 and i == i2:
                                    covered.add((i,j,k))
                if covered:
                    all_rects.append((frozenset(u_sup), frozenset(v_sup), frozenset(w_sup), frozenset(covered)))
    
    # Greedy partition
    while uncovered:
        best_rect = None
        best_count = 0
        for u_sup, v_sup, w_sup, covered in all_rects:
            new_covered = covered & uncovered
            if len(new_covered) > best_count:
                best_count = len(new_covered)
                best_rect = (u_sup, v_sup, w_sup, new_covered)
        
        if not best_rect:
            # Should not happen
            break
        
        terms.append(best_rect)
        uncovered -= best_rect[3]
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_sup, v_sup, w_sup, _) in enumerate(terms):
        for (i, j) in u_sup:
            U[i * m + j][c] = 1
        for (j, k) in v_sup:
            V[j * p + k][c] = 1
        for (k, i) in w_sup:
            W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
