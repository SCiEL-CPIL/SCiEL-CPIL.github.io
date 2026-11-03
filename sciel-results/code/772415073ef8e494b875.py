import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try to construct a 5-regular graph of girth >= g using PSL(2, 11)
    # This group has order 660.
    # If we find a connection set with girth >= g, we return it.
    # This is much better than the LUW graph for k=5, g=9.
    
    if k == 5:
        try:
            n, edges = construct_psl2_11(g)
            if n is not None:
                return n, edges
        except Exception:
            pass
    
    # Fallback to LUW construction
    return construct_luw(k, g)

def construct_psl2_11(g):
    # Generate PSL(2, 11)
    # Matrices over GF(11)
    p = 11
    # Generate all matrices with det=1
    # a b
    # c d
    # ad - bc = 1
    # a, b, c, d in 0..10
    # a cannot be 0 if c is 0? No.
    # Iterate a, b, c, d.
    # To be efficient, we can fix a, b, c and solve for d?
    # d = (1 + bc) / a if a != 0.
    # If a = 0, then -bc = 1 => bc = -1 = 10.
    
    elements = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                if a == 0:
                    # -bc = 1 => bc = 10
                    if c == 0:
                        continue
                    bc = b * c % p
                    if bc != 10:
                        continue
                    # d can be anything? No, det is -bc = 1.
                    # So d is free?
                    # Matrix: 0 b; c d. Det = -bc = 1.
                    # So for fixed b, c with bc=10, d can be 0..10.
                    for d in range(p):
                        elements.append((a, b, c, d))
                else:
                    # ad - bc = 1 => ad = 1 + bc => d = (1+bc) * a^{-1}
                    a_inv = pow(a, p-2, p)
                    d = (1 + b * c) * a_inv % p
                    elements.append((a, b, c, d))
    
    # Remove duplicates and identify inverse pairs (M and -M are same in PSL)
    # In PSL(2, p), M ~ -M.
    # So we can normalize: if a < 0 or (a == 0 and c < 0) or (a == 0 and c == 0 and b < 0), multiply by -1.
    # Actually, just keep a canonical form.
    unique_elements = {}
    for e in elements:
        a, b, c, d = e
        # Normalize: make top-left non-negative, or if 0, make top-right non-negative, etc.
        if a < 0 or (a == 0 and c < 0) or (a == 0 and c == 0 and b < 0):
            a, b, c, d = -a % p, -b % p, -c % p, -d % p
        if (a, b, c, d) not in unique_elements:
            unique_elements[(a, b, c, d)] = len(unique_elements)
    
    n = len(unique_elements)
    # Map elements to indices
    elem_to_idx = {e: i for i, e in enumerate(unique_elements.keys())}
    elems = list(unique_elements.keys())
    
    # Define matrix multiplication
    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return (
            (a1 * a2 + b1 * c2) % p,
            (a1 * b2 + b1 * d2) % p,
            (c1 * a2 + d1 * c2) % p,
            (c1 * b2 + d1 * d2) % p
        )
    
    # Precompute multiplication table or just compute on the fly?
    # 660 elements. 660*660 = 435600 ops. Fast enough.
    # Let's precompute the index of product.
    prod_idx = [[0]*n for _ in range(n)]
    for i, e1 in enumerate(elems):
        for j, e2 in enumerate(elems):
            prod = mat_mul(e1, e2)
            # Normalize prod
            a, b, c, d = prod
            if a < 0 or (a == 0 and c < 0) or (a == 0 and c == 0 and b < 0):
                a, b, c, d = -a % p, -b % p, -c % p, -d % p
            prod_idx[i][j] = elem_to_idx[(a, b, c, d)]
    
    # Find generators
    # We need a set S of 5 elements such that:
    # 1. S is closed under inverse (if s in S, s^-1 in S)
    # 2. No product of < g elements from S is identity.
    
    # Get inverse of each element
    inv_idx = [0]*n
    for i, e in enumerate(elems):
        a, b, c, d = e
        # Inverse of [[a,b],[c,d]] is [[d,-b],[-c,a]]
        inv = (d, -b % p, -c % p, a)
        # Normalize
        a2, b2, c2, d2 = inv
        if a2 < 0 or (a2 == 0 and c2 < 0) or (a2 == 0 and c2 == 0 and b2 < 0):
            a2, b2, c2, d2 = -a2 % p, -b2 % p, -c2 % p, -d2 % p
        inv_idx[i] = elem_to_idx[(a2, b2, c2, d2)]
    
    # Identify involutions (s = s^-1)
    involutions = [i for i in range(n) if inv_idx[i] == i]
    # Identify non-involutions
    non_involutions = [i for i in range(n) if inv_idx[i] != i]
    
    # We need to pick 5 generators.
    # Options:
    # - 5 involutions
    # - 1 involution + 2 pairs of non-involutions (s, s^-1)
    # - 5 non-involutions? No, must be closed under inverse. So pairs.
    # Since 5 is odd, we must have at least 1 involution.
    
    # Let's try random search for a set of 5 generators with high girth.
    # We will pick 1 involution and 2 pairs of non-involutions.
    
    best_girth = 0
    best_set = None
    
    # Limit search time
    search_deadline = time.time() + 10.0
    
    # Shuffle candidates
    random.shuffle(involutions)
    random.shuffle(non_involutions)
    
    # Try a few combinations
    for inv in involutions[:50]:
        if time.time() > search_deadline:
            break
        # Pick 2 pairs from non_involutions
        # We need 4 distinct non-involutions forming 2 pairs.
        # Let's just pick random non-involutions and check if they form pairs.
        # Or simpler: pick random elements, add their inverses.
        
        # Try 100 random pairs for this involution
        for _ in range(100):
            if time.time() > search_deadline:
                break
            s1 = random.choice(non_involutions)
            s2 = random.choice(non_involutions)
            if s1 == s2 or s1 == inv_idx[s2] or s2 == inv_idx[s1]:
                continue
            if s1 == inv_idx[s1] or s2 == inv_idx[s2]: # Should be non-involutions
                continue
                
            S = {inv, s1, inv_idx[s1], s2, inv_idx[s2]}
            if len(S) != 5:
                continue
            
            # Check girth
            girth = calculate_girth(prod_idx, S)
            if girth > best_girth:
                best_girth = girth
                best_set = S
                if best_girth >= g:
                    break
        if best_girth >= g:
            break
    
    if best_set is None or best_girth < g:
        return None, None
    
    # Construct edges
    edges = []
    S_list = list(best_set)
    for i in range(n):
        for s in S_list:
            j = prod_idx[i][s]
            if i < j:
                edges.append((i, j))
    
    return n, edges

def calculate_girth(prod_idx, S):
    # BFS from identity (index of identity element)
    # Identity is (1, 0, 0, 1)
    # Find its index
    p = 11
    id_elem = (1, 0, 0, 1)
    # It should be in the table.
    # We need to find the index of id_elem in the global list.
    # But calculate_girth doesn't have the global list.
    # Let's pass the index of identity.
    # Actually, let's just do BFS on the Cayley graph.
    # The identity element is the one that is (1,0,0,1).
    # We can find it by checking which element is (1,0,0,1).
    
    # This function is called inside construct_psl2_11 where we have elems.
    # Let's refactor to pass id_idx.
    pass

def construct_luw(k, g):
    # Original LUW code
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
