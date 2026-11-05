import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # The task is for k=3, g=16.
    # We use the voltage lift of the Petersen graph over Z_p.
    # Petersen graph has 10 vertices, 15 edges, girth 5.
    # Lifted graph has 10*p vertices.
    # We need girth >= 16.
    # This requires that no non-backtracking closed walk of length < 16 in Petersen
    # has a voltage sum of 0 in Z_p.
    
    # Moore bound for k=3, g=16 is 512.
    # So we need 10*p >= 512 => p >= 52.
    # Smallest prime p >= 52 is 53.
    
    # Define Petersen graph
    # Vertices 0..9
    # Edges:
    # Outer cycle: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Inner star: (5,7), (7,9), (9,6), (6,8), (8,5)
    # Spokes: (0,5), (1,6), (2,7), (3,8), (4,9)
    
    edges_petersen = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Map edges to indices 0..14
    edge_index = {}
    for i, (u, v) in enumerate(edges_petersen):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    adj = [[] for _ in range(10)]
    for u, v in edges_petersen:
        adj[u].append((v, edge_index[(u, v)]))
        adj[v].append((u, edge_index[(v, u)]))
        
    # Generate all non-backtracking closed walks of length < 16
    # A walk is a sequence of vertices v_0, ..., v_L such that v_0 = v_L
    # and v_{i-1} != v_{i+1} for all i.
    # We represent a walk by the sequence of edge indices.
    # Since it's a graph, the sequence of edges determines the walk if we start at a vertex.
    # But for voltage sums, we just need the sequence of edges with signs?
    # Voltage of a walk: sum of voltages of edges.
    # Note: If we traverse an edge (u,v) with voltage v_e, does the direction matter?
    # In a voltage graph, the voltage of an edge is an element of the group.
    # If we go u->v, we add v_e. If v->u, we add -v_e (in additive group).
    # So we need to track direction.
    
    walks = []
    max_len = g - 1  # 15
    
    # BFS/DFS to find all non-backtracking closed walks
    # State: (current_vertex, previous_vertex, length, edge_sequence)
    # To avoid duplicates, we can just generate all and check closedness.
    # Number of walks grows exponentially. For L=15, it's manageable.
    
    # Optimization: We only need the linear combination of edge voltages.
    # Let coeff[e] be the net number of times edge e is traversed in the walk
    # (counting direction: +1 for u->v, -1 for v->u).
    # The constraint is sum(coeff[e] * v_e) != 0.
    
    # We will store the coefficient vector for each walk.
    # To save memory/time, we can stop if we find a duplicate coefficient vector?
    # No, different walks can have same coeffs, but we just need to satisfy all.
    # Actually, if two walks have the same coefficient vector, satisfying one satisfies the other.
    # So we can keep a set of unique coefficient vectors.
    
    unique_coeffs = set()
    
    # Start from each vertex, and each edge
    for start_v in range(10):
        for next_v, e_idx in adj[start_v]:
            # Walk starts with edge e_idx
            # State: curr, prev, length, coeffs
            # coeffs is a tuple of 15 integers
            curr_coeffs = [0]*15
            curr_coeffs[e_idx] = 1 # Direction start_v -> next_v
            
            stack = [(next_v, start_v, 1, tuple(curr_coeffs))]
            
            while stack:
                curr, prev, length, coeffs = stack.pop()
                
                if length == max_len:
                    if curr == start_v:
                        unique_coeffs.add(coeffs)
                    continue
                    
                for next_next, e_idx_next in adj[curr]:
                    if next_next == prev:
                        continue # Non-backtracking
                    
                    new_coeffs = list(coeffs)
                    # Determine direction sign
                    # If edge is (u, v) in edge_index, and we go u->v, sign is +1.
                    # We stored edge_index[(u,v)] and (v,u) to same index.
                    # We need to know the canonical direction.
                    # Let's define canonical direction as (min(u,v), max(u,v)).
                    # If we traverse in canonical direction, sign +1, else -1.
                    u_can, v_can = sorted(edges_petersen[e_idx_next])
                    if curr == u_can and next_next == v_can:
                        sign = 1
                    else:
                        sign = -1
                        
                    new_coeffs[e_idx_next] += sign
                    
                    if length + 1 < max_len:
                        stack.append((next_next, curr, length + 1, tuple(new_coeffs)))
                    else:
                        # length + 1 == max_len
                        if next_next == start_v:
                            unique_coeffs.add(tuple(new_coeffs))
                            
    # Convert to list of numpy arrays for vectorized operations
    coeff_list = np.array(list(unique_coeffs), dtype=np.int64)
    num_constraints = len(coeff_list)
    
    # Search for prime p and voltage assignment
    # Primes starting from 53
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0: return False
        return True
        
    p = 53
    best_result = None
    
    while time.time() < deadline:
        if not is_prime(p):
            p += 1
            continue
            
        # Try to find voltage assignment v in Z_p^15
        # such that coeff_list @ v != 0 (element-wise) for all rows.
        
        # Randomized Local Search
        found = False
        v = np.random.randint(0, p, 15)
        
        # Calculate initial violations
        def count_violations(v):
            sums = coeff_list @ v % p
            return np.count_nonzero(sums == 0)
            
        current_violations = count_violations(v)
        
        max_iters = 50000
        for it in range(max_iters):
            if current_violations == 0:
                found = True
                break
                
            if time.time() > deadline - 5:
                break
                
            # Pick a random variable
            var_idx = np.random.randint(0, 15)
            old_val = v[var_idx]
            
            # Try a new random value
            new_val = np.random.randint(0, p)
            if new_val == old_val:
                continue
                
            v[var_idx] = new_val
            new_violations = count_violations(v)
            
            if new_violations < current_violations:
                current_violations = new_violations
            else:
                # Revert with some probability, or always revert if not improving?
                # Standard WalkSAT: revert if not improving, or with prob p.
                # Let's use a simple hill climbing with random restarts if stuck.
                v[var_idx] = old_val
                
            # Occasional random restart if stuck for too long?
            # For now, simple hill climb.
            
        if found:
            # Construct the lifted graph
            n = 10 * p
            edges = []
            # Vertices are (v_base, group_elem) where v_base in 0..9, group_elem in 0..p-1
            # Vertex ID = v_base * p + group_elem
            
            for u_base, v_base in edges_petersen:
                e_idx = edge_index[(u_base, v_base)]
                voltage = v[e_idx]
                
                # Canonical direction u_base -> v_base
                # If we define edge (u,v) with voltage val,
                # then u_g ~ v_{g+val}
                # and v_g ~ u_{g-val}
                
                for g in range(p):
                    u_id = u_base * p + g
                    v_id = v_base * p + (g + voltage) % p
                    edges.append((u_id, v_id))
                    
            # Verify girth
            # Build adjacency list
            adj_lifted = [[] for _ in range(n)]
            for u, v in edges:
                adj_lifted[u].append(v)
                adj_lifted[v].append(u)
                
            # Check for cycles < 16
            # BFS from each vertex, tracking parent to avoid backtracking
            valid = True
            for start in range(n):
                # visited: dict (node, parent) -> dist
                # Or just BFS with parent tracking
                # Since n is small (530), we can do BFS from each node.
                # But 530 BFSs * 530 nodes * 3 degree is fast.
                
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = [(start, -1)]
                
                while queue and valid:
                    curr, par = queue.pop(0)
                    for nei in adj_lifted[curr]:
                        if nei == par:
                            continue
                        if dist[nei] == -1:
                            dist[nei] = dist[curr] + 1
                            parent[nei] = curr
                            if dist[nei] >= g:
                                # We only care about cycles < g
                                # If we find a node at dist >= g, we don't need to go deeper for THIS start?
                                # No, a cycle could be formed by connecting two branches.
                                # But if we are looking for cycles of length < g,
                                # we only need to explore up to depth g-2 from start?
                                # Actually, if we find a neighbor of curr that is already visited
                                # and is not the parent, we found a cycle.
                                # Cycle length = dist[curr] + dist[nei] + 1.
                                pass
                        else:
                            # Found a cycle
                            cycle_len = dist[curr] + dist[nei] + 1
                            if cycle_len < g:
                                valid = False
                                break
                                
                if not valid:
                    break
                    
            if valid:
                best_result = (n, edges)
                break # Found a valid graph, return it
                
        p += 1
        
    if best_result is None:
        # Fallback: Return the LUW graph from the seed (or a simple valid one if we can't generate LUW easily here)
        # But we should try to return something valid.
        # If we fail, we can try a larger p or different strategy.
        # For now, if no prime worked, we might timeout.
        # Let's try to return a known valid graph if we have one.
        # The seed's LUW is valid but large.
        # We can implement a quick LUW fallback if time permits.
        
        # Actually, let's just return the last attempted p's graph even if invalid? No, must be valid.
        # If we didn't find one, we fail.
        # But we expect to find one for p=53 or 59.
        
        # Fallback: Use the seed's LUW logic (copied)
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

        start_v = (0, (None,) + (0,) * d)
        index, order, edges_fallback = {start_v: 0}, [start_v], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_fallback.append((index[vertex], index[other]))
        return len(order), edges_fallback
        
    return best_result
