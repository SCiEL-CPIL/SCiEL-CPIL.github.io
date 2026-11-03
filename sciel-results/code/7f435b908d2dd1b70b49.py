import time
import itertools

def construct(k, g):
    deadline = time.time() + 130.0
    best_result = None
    
    # Fallback: LUW construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges
    
    # PSL(2, q) Cayley graph search
    def try_cayley_psl2(q, max_attempts=50):
        if q % 2 == 0:
            return None
        
        # Elements of PSL(2, q) as (a,b,c,d) with ad-bc=1 mod q
        # Identity: (1,0,0,1)
        # We work in GF(q) where q is prime
        
        def mat_mult(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            return ((a1*a2 + b1*c2) % q, (a1*b2 + b1*d2) % q,
                    (c1*a2 + d1*c2) % q, (c1*b2 + d1*d2) % q)
        
        def mat_inv(m):
            a, b, c, d = m
            det = (a*d - b*c) % q
            # In PSL, we normalize so det=1
            return (d % q, (-b) % q, (-c) % q, a % q)
        
        identity = (1, 0, 0, 1)
        
        # Generate all elements
        elements = set()
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    for d in range(q):
                        if (a*d - b*c) % q == 1:
                            elements.add((a, b, c, d))
        
        # Also add negatives (since PSL identifies ±I)
        # Actually, let's just use SL(2,q) and divide by ±I
        # Better approach: generate SL(2,q) directly
        
        # Regenerate: SL(2,q) elements with det=1
        sl2_elements = set()
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    for d in range(q):
                        if (a*d - b*c) % q == 1:
                            sl2_elements.add((a, b, c, d))
        
        # PSL(2,q) = SL(2,q) / {±I}
        # Map each element to its canonical representative
        psl_elements = set()
        for e in sl2_elements:
            canonical = e if e >= mat_inv(e) else mat_inv(e)
            psl_elements.add(canonical)
        
        n = len(psl_elements)
        
        # Find involutions (elements of order 2)
        involutions = []
        for e in psl_elements:
            if e == identity:
                continue
            e2 = mat_mult(e, e)
            if e2 == identity:
                involutions.append(e)
        
        # Find elements of high order
        def get_order(e):
            cur = e
            order = 1
            while cur != identity:
                cur = mat_mult(cur, e)
                order += 1
                if order > n:
                    break
            return order
        
        # Precompute multiplication table for efficiency
        elem_list = list(psl_elements)
        elem_idx = {e: i for i, e in enumerate(elem_list)}
        
        mult_table = [[None] * n for _ in range(n)]
        for i, e1 in enumerate(elem_list):
            for j, e2 in enumerate(elem_list):
                prod = mat_mult(e1, e2)
                mult_table[i][j] = elem_idx[prod]
        
        inv_table = [None] * n
        for i, e in enumerate(elem_list):
            inv_table[i] = elem_idx[mat_inv(e)]
        
        # Try random Cayley graphs
        import random
        rng = random.Random(42)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Choose generators: one element of high order + two involutions
            # or three involutions
            
            # Strategy 1: a (high order) + b, c (involutions)
            # Find element with order close to q or (q+1)/2 etc.
            candidates = []
            for i, e in enumerate(elem_list):
                if e == identity:
                    continue
                if e in involutions:
                    continue
                ord_e = get_order(e)
                if ord_e >= q // 2:
                    candidates.append((ord_e, i, e))
            
            if not candidates:
                continue
            
            candidates.sort(reverse=True)
            
            # Pick a from candidates
            a_idx = rng.choice([c[1] for c in candidates[:10]])
            a_elem = elem_list[a_idx]
            a_inv_idx = inv_table[a_idx]
            
            # Pick b, c from involutions
            if len(involutions) < 2:
                continue
            
            b, c = rng.sample(involutions, 2)
            b_idx = elem_idx[b]
            c_idx = elem_idx[c]
            
            # Generators: a, a^{-1}, b, c
            # But we need degree 3, so we need exactly 3 generators
            # If b and c are both involutions, and a != a^{-1}, we have 4 generators
            # For degree 3: use {a, a^{-1}, b} where b is involution
            # Or {a, b, c} where all are involutions (but a is not involution)
            
            # Let's use {a, a^{-1}, b} with b an involution
            generators = [a_idx, a_inv_idx, b_idx]
            
            # Check they are distinct
            if len(set(generators)) != 3:
                continue
            
            # Build graph and check girth
            adj = [set() for _ in range(n)]
            for i in range(n):
                for g_idx in generators:
                    j = mult_table[i][g_idx]
                    adj[i].add(j)
                    adj[j].add(i)
            
            # Verify regularity
            for i in range(n):
                if len(adj[i]) != 3:
                    break
            else:
                # Check girth using BFS from each vertex (non-backtracking)
                # Actually, just check if any cycle < g exists
                min_girth = float('inf')
                
                # BFS from each vertex to find shortest cycle
                for start in range(n):
                    # Standard BFS finds shortest path, but we need cycles
                    # Use BFS from start, track parent, find back edges
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[start] = 0
                    queue = [start]
                    
                    for u in queue:
                        if time.time() > deadline - 10.0:
                            break
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            elif v != parent[u]:
                                # Found a cycle
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < min_girth:
                                    min_girth = cycle_len
                        if time.time() > deadline - 10.0:
                            break
                    
                    if time.time() > deadline - 10.0:
                        break
                    
                    if min_girth < g:
                        break
                
                if min_girth >= g:
                    # Collect edges
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
        
        return None
    
    # Try Cayley graph first
    try:
        result = try_cayley_psl2(13, max_attempts=20)
        if result:
            return result
    except Exception:
        pass
    
    # Fallback to LUW
    return luw_construct(k, g)
