import time
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 145.0

    def get_luw_fallback(k, g):
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

    def verify_girth(n, adj, g):
        """Verify girth >= g using non-backtracking walk enumeration.
        A cycle of length L < g exists iff there's a non-backtracking walk
        of length L from some vertex back to itself.
        We enumerate all non-backtracking walks up to length g-1 from each vertex.
        """
        for s in range(n):
            # State: (current_vertex, previous_vertex, length)
            # We track visited (vertex, prev) pairs at each depth to avoid re-enumeration
            # Actually, for girth check, we just need to see if any NB walk of length < g returns to s
            # Enumerate NB walks from s up to length g-1
            # State: (curr, prev) where prev is the vertex we came from (or -1 for start)
            # At each step, we can go to any neighbor of curr except prev
            # We want to find if we can return to s at length L where 3 <= L < g
            # (length 1 and 2 can't form cycles in simple graphs)
            
            # BFS-like enumeration of NB walks
            # visited_at_depth[d] = set of (curr, prev) states reachable at depth d
            # But we need to track the full path to detect return to s
            # Actually, we just need: at depth L, is there a state (s, prev) for some prev != -1?
            # That means we returned to s via prev, which is a cycle of length L.
            
            max_depth = g - 1
            # states: list of sets, states[d] = set of (curr, prev)
            states = [set() for _ in range(max_depth + 1)]
            states[0].add((s, -1))
            
            for depth in range(1, max_depth + 1):
                if time.time() > deadline - 10:
                    return False  # assume valid if we're running out of time
                for (curr, prev) in states[depth - 1]:
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        new_state = (nxt, curr)
                        if depth >= 3 and nxt == s and curr != -1:
                            # Found a cycle of length depth
                            return False
                        states[depth].add(new_state)
        return True

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    if k == 4 and g == 10:
        # Try voltage graph (lift) construction first
        result = _voltage_lift_construct(4, 10, deadline)
        if result is not None:
            n, edges = result
            # Quick verification
            adj = build_adj(n, edges)
            valid = True
            for u in range(n):
                if len(adj[u]) != 4:
                    valid = False
                    break
            if valid and verify_girth(n, adj, g):
                return n, edges
        
        # Fallback to LUW construction
        n, edges = get_luw_fallback(k, g)
        return n, edges
    else:
        n, edges = get_luw_fallback(k, g)
        return n, edges

def _voltage_lift_construct(k, g, deadline):
    """Construct a 4-regular girth-10 graph using voltage lift over a 4-regular girth-5 base."""
    
    # Find a 4-regular graph of girth 5 with minimum vertices
    # The smallest 4-regular girth-5 graph has 20 vertices
    # We'll use the complement of a perfect matching in K_5 x K_4 or similar
    # Actually, let's use the incidence graph of projective plane or construct one
    
    # The 4-regular girth-5 graph on 20 vertices: 
    # Take vertices as pairs (i, j) where i in 0..3, j in 0..4
    # This is a 20-vertex graph. Let me use a known construction.
    
    # Use the 4-regular graph from the Hoffman-Singleton construction restricted
    # Or simply: the line graph of K_5 has 10 vertices, degree 6 - not what we want
    # 
    # Let's use the 20-vertex 4-regular girth-5 graph:
    # Vertices: (i, j) for i in 0..3, j in 0..4
    # Edges: connect (i,j) to (i',j') if j' = (j + i' - i) mod 5 and i' != i
    # Wait, that gives degree 4*1 = 4. Let me check girth.
    
    # Alternative: use the bipartite double cover or known (4,5)-cage
    # The (4,5)-cage has 20 vertices. Let me construct it as:
    # Two copies of a 10-cycle, with cross edges
    
    # Actually, simplest: use the complete bipartite K_{4,4} expanded
    # Let me just use a known 4-regular girth-5 graph on 20 vertices
    
    # The 20-vertex 4-regular girth-5 graph can be constructed as follows:
    # Vertices 0-19. Use the following adjacency:
    # This is the complement of 5 disjoint K_4's? No, that's 5*4=20 vertices, degree 16-3=13.
    
    # Let me use the Petersen graph doubled: no, Petersen is 3-regular.
    
    # Correct approach: The (4,5)-cage is the 4-regular graph on 20 vertices with girth 5.
    # It can be described as: take vertices (i, j) for i in Z_4, j in Z_5.
    # Edges: (i,j) ~ (i, j+1), (i,j) ~ (i, j-1), (i,j) ~ (i+1, j+1), (i,j) ~ (i-1, j-1)
    # This is 4-regular. Let's verify girth >= 5.
    
    base_n = 20
    base_adj = [[] for _ in range(base_n)]
    for i in range(4):
        for j in range(5):
            u = i * 5 + j
            # Horizontal edges
            v1 = i * 5 + ((j + 1) % 5)
            v2 = i * 5 + ((j - 1) % 5)
            # Diagonal edges
            v3 = ((i + 1) % 4) * 5 + ((j + 1) % 5)
            v4 = ((i - 1) % 4) * 5 + ((j - 1) % 5)
            for v in [v1, v2, v3, v4]:
                if v != u and v not in base_adj[u]:
                    base_adj[u].append(v)
                    base_adj[v].append(u)
    
    # Verify base graph is 4-regular and girth >= 5
    for u in range(base_n):
        if len(base_adj[u]) != 4:
            # Try alternative construction
            break
    else:
        # Check girth of base graph
        base_girth_ok = True
        for s in range(base_n):
            # BFS to find shortest cycle through s
            # Actually just check no 4-cycles
            for a in base_adj[s]:
                for b in base_adj[a]:
                    if b == s:
                        continue
                    for c in base_adj[b]:
                        if c == a:
                            continue
                        if c == s:
                            # Found a 4-cycle
                            base_girth_ok = False
                            break
                        if c in base_adj[s]:
                            # Found a 4-cycle: s-a-b-c-s
                            base_girth_ok = False
                            break
                    if not base_girth_ok:
                        break
                if not base_girth_ok:
                    break
            if not base_girth_ok:
                break
        
        if base_girth_ok:
            # Now search for voltage assignments in Z_2^m
            # The lift has base_n * 2^m vertices
            # We need girth >= 10 in the lift
            # A closed non-backtracking walk of length L in the base with voltage product 0 in Z_2^m
            # gives a cycle of length L in the lift
            
            # Enumerate all non-backtracking closed walks of length 4, 6, 8 in the base
            # For each, compute the voltage sum (XOR) and ensure it's not zero for our assignment
            
            # Get all NB closed walks of length 4, 6, 8
            nb_walks = {4: [], 6: [], 8: []}
            for target_len in [4, 6, 8]:
                for s in range(base_n):
                    # DFS to find NB closed walks of length target_len starting at s
                    # State: (current, prev, path_edges)
                    def dfs_nb(curr, prev, edge_path, depth):
                        if time.time() > deadline - 20:
                            return
                        if depth == target_len:
                            if curr == s:
                                nb_walks[target_len].append(list(edge_path))
                            return
                        for nxt in base_adj[curr]:
                            if nxt == prev:
                                continue
                            edge_path.append((curr, nxt))
                            dfs_nb(nxt, curr, edge_path, depth + 1)
                            edge_path.pop()
                    
                    dfs_nb(s, -1, [], 1)
            
            # Now we have constraints: for each NB closed walk of length L < 10,
            # the XOR of voltages along the walk must be non-zero
            # Voltage assignment: each edge e gets a vector in Z_2^m
            # We need to find m and assignments such that all these walks have non-zero total
            
            # This is a constraint satisfaction problem
            # For m=1: each edge gets 0 or 1. Need no closed walk of length 4,6,8 to have even XOR
            # For m=2: each edge gets one of 4 values
            
            # Start with small m and increase
            best_result = None
            
            for m in range(1, 6):
                if time.time() > deadline - 15:
                    break
                n_lift = base_n * (2 ** m)
                if n_lift >= 320:
                    # Can't beat LUW
                    break
                
                # Random search for voltage assignments
                import random
                random.seed(42)
                
                for attempt in range(500):
                    if time.time() > deadline - 10:
                        break
                    
                    # Assign random voltages to edges
                    edge_set = set()
                    for u in range(base_n):
                        for v in base_adj[u]:
                            if u < v:
                                edge_set.add((u, v))
                    edge_list = list(edge_set)
                    edge_index = {e: i for i, e in enumerate(edge_list)}
                    
                    # Voltage assignment: edge_list[i] -> random vector in Z_2^m
                    voltages = np.random.randint(0, 2, size=(len(edge_list), m), dtype=np.int8)
                    
                    # Check all NB closed walks
                    valid = True
                    for L in [4, 6, 8]:
                        for walk in nb_walks[L]:
                            total = np.zeros(m, dtype=np.int8)
                            for (u, v) in walk:
                                e = (min(u,v), max(u,v))
                                total ^= voltages[edge_index[e]]
                            if np.all(total == 0):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Build the lifted graph
                        lift_edges = []
                        for u in range(base_n):
                            for v in base_adj[u]:
                                e = (min(u,v), max(u,v))
                                vol = voltages[edge_index[e]]
                                # In the lift, vertex (u, x) connects to (v, x XOR vol) if u->v
                                # We need to handle direction
                                if u < v:
                                    for x in range(2 ** m):
                                        y = x ^ int(np.packbits(vol).tobytes() if m > 8 else vol, 2)
                                        # Simpler: compute XOR
                                        y = x
                                        for bit in range(m):
                                            if vol[bit]:
                                                y ^= (1 << bit)
                                        lift_edges.append((u * (2**m) + x, v * (2**m) + y))
                                else:
                                    for x in range(2 ** m):
                                        y = x
                                        for bit in range(m):
                                            if vol[bit]:
                                                y ^= (1 << bit)
                                        lift_edges.append((v * (2**m) + x, u * (2**m) + y))
                        
                        # Deduplicate
                        lift_edges = list(set(lift_edges))
                        
                        if len(lift_edges) == 2 * n_lift:  # 4-regular on n_lift vertices
                            best_result = (n_lift, lift_edges)
                            break
                
                if best_result is not None:
                    break
            
            if best_result is not None:
                return best_result
    
    return None
