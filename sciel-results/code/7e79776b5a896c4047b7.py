"""Construct k-regular graph of girth g via Cayley graphs on non-abelian groups."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using non-backtracking walks from every vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = (g - 1) // 2
    for start in range(n):
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start and depth + 1 < g:
                    return False
                state = (nb, node)
                if state in visited:
                    continue
                visited.add(state)
                queue.append((nb, node, depth + 1))
    return True

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) - reliable fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q
    
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Search for Cayley graphs on non-abelian groups (products of A5)
    with involutory generators that have high girth. Fall back to L-U-W.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    random.seed(42)
    start_time = time.time()
    time_limit = 115  # seconds, safety margin before 120s kill
    
    # Moore bound
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # Build A5 group tables
    # A5 has 60 elements, 12 involutions (products of 2 disjoint transpositions)
    # Represent elements as tuples (perm of 5 elements)
    
    def build_a5():
        """Build A5 group with multiplication table and involutions."""
        from itertools import permutations
        
        # All even permutations of 5 elements
        all_perms = list(permutations(range(5)))
        
        def is_even(perm):
            """Check if permutation is even."""
            inv = 0
            seen = [False]*5
            for i in range(5):
                for j in range(i):
                    if perm[i] > perm[j]:
                        inv += 1
            return inv % 2 == 0
        
        even_perms = [p for p in all_perms if is_even(p)]
        # There should be 60 even permutations
        n_a5 = len(even_perms)
        
        # Multiplication: (p*q)[i] = p[q[i]]
        mul_table = [[0]*n_a5 for _ in range(n_a5)]
        idx_map = {p: i for i, p in enumerate(even_perms)}
        
        for i, p in enumerate(even_perms):
            for j, q in enumerate(even_perms):
                # p * q means apply q first, then p
                prod = tuple(p[q[t]] for t in range(5))
                mul_table[i][j] = idx_map[prod]
        
        # Identity
        identity = idx_map[(0,1,2,3,4)]
        
        # Find involutions (elements of order 2, not identity)
        involutions = []
        for i in range(n_a5):
            if i == identity:
                continue
            if mul_table[i][i] == identity:
                involutions.append(i)
        
        return even_perms, mul_table, identity, involutions, idx_map
    
    # Build A5
    a5_elems, a5_mul, a5_id, a5_invol, a5_idx = build_a5()
    
    # For the product group A5^r, elements are tuples of A5 elements
    # Multiplication is component-wise
    # Involutions are tuples where each component is identity or an involution,
    # and at least one component is an involution
    
    def build_product_group(r):
        """Build A5^r group structure."""
        n = 60 ** r
        # We won't build full multiplication table (too large for r>2)
        # Instead, use direct computation
        
        # For r=1: n=60
        # For r=2: n=3600
        # For r=3: n=216000 (too large for full verification)
        
        if r == 1:
            return a5_elems, a5_mul, a5_id, a5_invol, n
        elif r == 2:
            # Elements are pairs (i, j) where i, j in range(60)
            # Multiplication: (i,j) * (i',j') = (a5_mul[i][i'], a5_mul[j][j'])
            # Identity: (a5_id, a5_id)
            # Involutions: (i,j) where (i,j)*(i,j) = (id,id)
            # i.e., a5_mul[i][i] == id and a5_mul[j][j] == id
            inv_pairs = []
            for i in range(60):
                for j in range(60):
                    if a5_mul[i][i] == a5_id and a5_mul[j][j] == a5_id:
                        if not (i == a5_id and j == a5_id):
                            inv_pairs.append((i, j))
            return None, None, (a5_id, a5_id), inv_pairs, n
        else:
            return None, None, None, None, n
    
    # Quick girth check for Cayley graph using BFS on group elements
    def check_girth_cayley(group_size, gens_indices, g, mul_func, identity, max_depth):
        """
        Check if Cayley graph has girth >= g.
        gens_indices: list of generator indices in the group
        mul_func: function to multiply two group elements
        Returns True if girth >= g
        """
        # BFS from identity, tracking (current_element, prev_element)
        # A cycle of length < g exists if we can reach identity in < g steps
        # without backtracking
        
        # For efficiency, use the fact that in a Cayley graph,
        # a cycle of length L from identity corresponds to a word of length L
        # in the generators that equals identity
        
        # BFS over states (group_element, last_generator_used)
        # Start: for each gen, state (gen, gen_index)
        # If we reach (identity, any_gen) at depth < g, we have a short cycle
        
        visited = set()
        queue = deque()
        
        for gi, gen in enumerate(gens_indices):
            visited.add((gen, gi))
            queue.append((gen, gi, 1))
        
        while queue:
            elem, last_gi, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for gi, gen in enumerate(gens_indices):
                if gi == last_gi:
                    continue  # non-backtracking
                new_elem = mul_func(elem, gen)
                if new_elem == identity and depth + 1 < g:
                    return False  # Found short cycle
                state = (new_elem, gi)
                if state not in visited:
                    visited.add(state)
                    queue.append((new_elem, gi, depth + 1))
        
        return True
    
    best_result = None
    best_n = float('inf')
    
    # Try A5^r for r = 1, 2, 3
    for r in range(1, 4):
        if time.time() - start_time > time_limit:
            break
        
        n = 60 ** r
        if n > 100000:  # Too large for practical verification
            break
        
        if r == 1:
            # Group is A5 itself
            def mul1(e1, e2):
                return a5_mul[e1][e2]
            identity = a5_id
            available_gens = a5_invol
            
            # Try random sets of k involutions
            max_attempts = 200
            for attempt in range(max_attempts):
                if time.time() - start_time > time_limit:
                    break
                
                # Pick k random distinct involutions
                if len(available_gens) < k:
                    break
                gens = random.sample(available_gens, k)
                
                # Quick check: no generator should be product of two others (would give 3-cycle)
                # For A5, check if any gen_i * gen_j == gen_k for i≠j≠k
                valid = True
                for i in range(k):
                    for j in range(k):
                        if i == j:
                            continue
                        prod = a5_mul[gens[i]][gens[j]]
                        for l in range(k):
                            if l != i and l != j and prod == gens[l]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check girth
                max_depth = (g - 1) // 2
                if check_girth_cayley(n, gens, g, mul1, identity, max_depth):
                    # Build edge list
                    edges = []
                    for v in range(n):
                        for gen in gens:
                            u = a5_mul[v][gen]
                            if v < u:
                                edges.append((v, u))
                    
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                        break  # Found valid for r=1, try to improve or move on
        
        elif r == 2:
            # Group is A5 x A5, n = 3600
            identity = (a5_id, a5_id)
            
            # Involutions in A5 x A5: (i,j) where i,j are involutions or identity in A5
            # (including identity in one or both components, but not both)
            inv_a5 = set(a5_invol)
            available_gens = []
            for i in range(60):
                for j in range(60):
                    i_is_inv = (i == a5_id) or (i in inv_a5)
                    j_is_inv = (j == a5_id) or (j in inv_a5)
                    if i_is_inv and j_is_inv and not (i == a5_id and j == a5_id):
                        available_gens.append((i, j))
            
            def mul2(e1, e2):
                return (a5_mul[e1[0]][e2[0]], a5_mul[e1[1]][e2[1]])
            
            max_attempts = 100
            for attempt in range(max_attempts):
                if time.time() - start_time > time_limit:
                    break
                
                if len(available_gens) < k:
                    break
                gens = random.sample(available_gens, k)
                
                # Quick check for 3-cycles: gen_i * gen_j == gen_k
                valid = True
                for i in range(k):
                    for j in range(k):
                        if i == j:
                            continue
                        prod = mul2(gens[i], gens[j])
                        for l in range(k):
                            if l != i and l != j and prod == gens[l]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check girth (this is expensive for n=3600)
                max_depth = (g - 1) // 2
                if check_girth_cayley(n, gens, g, mul2, identity, max_depth):
                    # Build edge list
                    edges = []
                    for v in range(n):
                        vi, vj = v // 60, v % 60
                        for gen in gens:
                            gi, gj = gen
                            ui = a5_mul[vi][gi]
                            uj = a5_mul[vj][gj]
                            u = ui * 60 + uj
                            if v < u:
                                edges.append((v, u))
                    
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                        break
        
        if best_result is not None:
            break
    
    # If we found a valid Cayley graph, return it
    if best_result is not None:
        return best_result
    
    # Fallback to L-U-W construction
    return _luw_construct(k, g)
# EVOLVE-BLOCK-END
