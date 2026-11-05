"""Voltage graph construction for 3-regular girth-16 graph using PSL(2,p)."""
import time
import random
from collections import deque

def construct(k, g):
    """Construct a 3-regular graph with girth >= g using voltage graph on PSL(2,p)."""
    deadline = time.time() + 115  # safety margin before 120s limit
    best_n = None
    best_edges = None
    
    # Base graph: K4 (4 vertices, 3-regular)
    # Vertices: 0, 1, 2, 3
    # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    base_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    base_adj = {0: [1,2,3], 1: [0,2,3], 2: [0,1,3], 3: [0,1,2]}
    
    # Try different primes p for PSL(2,p)
    # n = 4*p vertices in the lift
    # We need girth >= g, so search for voltage assignments
    primes_to_try = []
    for p in range(101, 2000, 2):
        if p == 2:
            continue
        # Check if prime
        is_prime = True
        for d in range(3, int(p**0.5)+1, 2):
            if p % d == 0:
                is_prime = False
                break
        if is_prime and p > 2:
            primes_to_try.append(p)
    
    # Also try smaller primes
    for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if p not in primes_to_try:
            primes_to_try.append(p)
    
    primes_to_try = sorted(set(primes_to_try))
    
    # PSL(2,p) implementation
    # Elements are 2x2 matrices with determinant 1, modulo p, up to sign
    # Represent as (a, b, c, d) where matrix is [[a,b],[c,d]] with ad-bc=1
    
    def mat_mul(m1, m2, p):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return ((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p,
                (c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p)
    
    def mat_inv(m, p):
        a, b, c, d = m
        # det = ad - bc = 1, so inv = [[d, -b], [-c, a]]
        return (d % p, (-b) % p, (-c) % p, a % p)
    
    def mat_eq(m1, m2, p):
        # Two matrices are equal in PSL(2,p) if m1 == m2 or m1 == -m2
        if (m1[0] == m2[0] and m1[1] == m2[1] and m1[2] == m2[2] and m1[3] == m2[3]):
            return True
        if (m1[0] == (-m2[0]) % p and m1[1] == (-m2[1]) % p and 
            m1[2] == (-m2[2]) % p and m1[3] == (-m2[3]) % p):
            return True
        return False
    
    def mat_identity(p):
        return (1, 0, 0, 1)
    
    def mat_is_identity(m, p):
        return mat_eq(m, mat_identity(p), p)
    
    # Generate random elements of PSL(2,p)
    def random_psl_element(p, rng):
        while True:
            a = rng.randint(0, p-1)
            b = rng.randint(0, p-1)
            c = rng.randint(0, p-1)
            d = rng.randint(0, p-1)
            det = (a*d - b*c) % p
            if det == 1:
                return (a, b, c, d)
    
    # Check if a set of voltage assignments gives girth >= g
    # We need to check all non-backtracking closed walks of length < g in the base graph
    # and verify their voltage products are not identity
    
    def check_girth(voltages, p, required_girth):
        """Check if voltage assignment gives girth >= required_girth.
        voltages: dict mapping (u,v) -> matrix for u<v, and we handle orientation
        """
        # Build oriented voltage map
        # For edge (u,v), if we traverse u->v, use voltages[(u,v)]
        # if we traverse v->u, use inverse
        volt_map = {}
        for (u, v) in base_edges:
            volt_map[(u, v)] = voltages.get((u, v), None)
            volt_map[(v, u)] = mat_inv(voltages.get((u, v), None), p)
        
        # Find all non-backtracking closed walks of length < required_girth
        # A closed walk is (v0, v1, ..., vk) where v0=vk, and no consecutive edges are the same
        # For girth check, we need to check walks of length 3 to required_girth-1
        # (girth is the length of the shortest cycle, so we need no cycles of length < required_girth)
        
        # Use BFS in the state space of (current_vertex, previous_vertex)
        # State: (current, prev) where prev is the vertex we came from
        # Start: all (v, v) for self-loops? No, we start walks from each vertex
        
        # For each starting vertex, do BFS tracking (current, prev, product)
        # But product space is large, so we'll be selective
        
        # Actually, for small base graphs, we can enumerate all non-backtracking walks
        # of length up to required_girth-1 and check if any closed walk has identity product
        
        # Max walk length to check: required_girth - 1
        max_len = required_girth - 1
        
        # Enumerate all non-backtracking walks of length 3 to max_len
        # A walk is (v0, v1, ..., vk) with v0 = vk for closed walks
        # Non-backtracking: v_{i-1} != v_{i+1} for all i
        
        # For efficiency, use DFS with pruning
        def dfs(current, prev, length, product):
            if length >= max_len:
                return False
            
            for next_v in base_adj[current]:
                if next_v == prev and length > 0:
                    continue  # non-backtracking constraint
                if length == 0:
                    # First step
                    new_product = mat_mul(product, volt_map[(current, next_v)], p)
                else:
                    new_product = mat_mul(product, volt_map[(current, next_v)], p)
                
                if next_v == start_v and length + 1 >= 3:
                    # Closed walk of length >= 3
                    if mat_is_identity(new_product, p):
                        return True  # Found a short cycle
                
                if dfs(next_v, current, length + 1, new_product):
                    return True
            
            return False
        
        # Check from each starting vertex
        for start_v in range(4):
            start_product = mat_identity(p)
            if dfs(start_v, -1, 0, start_product):
                return False
        
        return True
    
    # Search for voltage assignments
    for p in primes_to_try:
        if time.time() > deadline - 10:
            break
        
        n = 4 * p
        if best_n is not None and n >= best_n:
            continue
        
        # Try random voltage assignments
        max_attempts = 20000
        rng = random.Random(42 + p)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random voltages for the 6 edges
            voltages = {}
            for (u, v) in base_edges:
                voltages[(u, v)] = random_psl_element(p, rng)
            
            # Quick check: verify no edge voltage is identity or its inverse
            valid = True
            for (u, v) in base_edges:
                v_mat = voltages[(u, v)]
                if mat_is_identity(v_mat, p):
                    valid = False
                    break
                inv = mat_inv(v_mat, p)
                if mat_is_identity(inv, p):
                    valid = False
                    break
            if not valid:
                continue
            
            # Check girth
            if check_girth(voltages, p, g):
                # Build the lift graph
                edges = []
                # Vertex (base_v, group_elem) where group_elem is represented by matrix
                # We need to enumerate all group elements
                # For PSL(2,p), |G| = p(p^2-1)/2
                # This is too large to enumerate for large p
                
                # Instead, build the graph by following edges
                # Vertex index: (base_v, elem_index)
                # We need to assign indices to group elements as we encounter them
                
                elem_index = {}
                elem_list = []
                def get_elem_index(elem):
                    # Use a canonical form: normalize so that first non-zero entry in (a,b,c,d) is positive
                    # and prefer (a,b,c,d) over (-a,-b,-c,-d)
                    a, b, c, d = elem
                    # Canonical: if a < 0 or (a == 0 and b < 0) or (a==0 and b==0 and c < 0) or (a==0 and b==0 and c==0 and d < 0): negate
                    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0) or (a == 0 and b == 0 and c == 0 and d < 0):
                        elem = ((-a) % p, (-b) % p, (-c) % p, (-d) % p)
                    if elem not in elem_index:
                        elem_index[elem] = len(elem_list)
                        elem_list.append(elem)
                    return elem_index[elem]
                
                # Start with (0, identity)
                identity = mat_identity(p)
                start_idx = get_elem_index(identity)
                
                # BFS to build the lift
                visited = set()
                queue = deque()
                queue.append((0, identity))
                visited.add((0, identity))
                
                while queue:
                    base_v, elem = queue.popleft()
                    for next_v in base_adj[base_v]:
                        # Traverse edge (base_v, next_v)
                        volt = volt_map_get(voltages, base_v, next_v, p)
                        new_elem = mat_mul(elem, volt, p)
                        # Canonicalize
                        a, b, c, d = new_elem
                        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0) or (a == 0 and b == 0 and c == 0 and d < 0):
                            new_elem = ((-a) % p, (-b) % p, (-c) % p, (-d) % p)
                        
                        if (next_v, new_elem) not in visited:
                            visited.add((next_v, new_elem))
                            queue.append((next_v, new_elem))
                
                # Now build edges
                n_vertices = 0
                vertex_map = {}
                for (bv, elem) in visited:
                    a, b, c, d = elem
                    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0) or (a == 0 and b == 0 and c == 0 and d < 0):
                        elem = ((-a) % p, (-b) % p, (-c) % p, (-d) % p)
                    key = (bv, elem)
                    if key not in vertex_map:
                        vertex_map[key] = n_vertices
                        n_vertices += 1
                
                edges = []
                for (bv, elem) in visited:
                    a, b, c, d = elem
                    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0) or (a == 0 and b == 0 and c == 0 and d < 0):
                        elem = ((-a) % p, (-b) % p, (-c) % p, (-d) % p)
                    u_idx = vertex_map[(bv, elem)]
                    for next_v in base_adj[bv]:
                        volt = volt_map_get(voltages, bv, next_v, p)
                        new_elem = mat_mul(elem, volt, p)
                        a2, b2, c2, d2 = new_elem
                        if a2 < 0 or (a2 == 0 and b2 < 0) or (a2 == 0 and b2 == 0 and c2 < 0) or (a2 == 0 and b2 == 0 and c2 == 0 and d2 < 0):
                            new_elem = ((-a2) % p, (-b2) % p, (-c2) % p, (-d2) % p)
                        v_idx = vertex_map.get((next_v, new_elem))
                        if v_idx is not None and u_idx < v_idx:
                            edges.append((u_idx, v_idx))
                
                # Verify
                if len(edges) == 3 * n_vertices // 2:
                    if best_n is None or n_vertices < best_n:
                        best_n = n_vertices
                        best_edges = edges
                        # Don't break; try to find smaller
                        break  # Found one for this p, try smaller p
        
        # Continue to next p (smaller n)
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: return the original LZW construction
    return _lzw_construct(k, g)

def volt_map_get(voltages, u, v, p):
    """Get voltage for directed edge (u,v)."""
    if (u, v) in voltages:
        return voltages[(u, v)]
    elif (v, u) in voltages:
        return mat_inv(voltages[(v, u)], p)
    else:
        # Should not happen
        return (1, 0, 0, 1)

def mat_inv(m, p):
    a, b, c, d = m
    return (d % p, (-b) % p, (-c) % p, a % p)

def _lzw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
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
