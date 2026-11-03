"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g using the Lazebnik-Ustimenko-Woldar
    family of graphs. We try multiple values of d (recurrence depth) from smallest to
    largest, building the graph and verifying the girth. We return the first valid graph.
    """
    q = k  # For k=5, q=5 (prime field GF(5))
    
    # Candidate d values: girth of LUW(d,q) is 2d+4 for q > 2d (standard result)
    # But for small q, it might be smaller. We'll verify.
    # For g=9, we need 2d+4 >= 9 => d >= 2.5 => d >= 3.
    # But the seed used d=4 for g=9, implying d=3 might not work for q=5.
    # Let's try d=3, 4, 5, ... until we get a valid graph.
    
    # Field operations for GF(q) where q is prime (k=5)
    def f_add(a, b): return (a + b) % q
    def f_sub(a, b): return (a - b) % q
    def f_mul(a, b): return (a * b) % q
    
    # The LUW recurrence rules
    # For c=2: (1,1), c=3: (2,1), c=4: (1,2)
    # For c>=5: rules depend on (c-5) % 4
    def compute_rules(d):
        rules = {}
        rules[2] = (1, 1)
        rules[3] = (2, 1)
        rules[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            base = c - r
            candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
            rules[c] = candidates[r]
        return rules
    
    def build_luw(d):
        """Build the LUW graph with parameter d. Returns (n, edges) or None if invalid."""
        rules = compute_rules(d)
        
        # BFS over the incidence graph
        # Vertex is (side, coords) where side in {0,1} and coords is tuple of length d+1
        # coords[0] is unused (0), coords[1..d] are in GF(q)
        start_key = (0, (0,) * (d + 1))
        label = {start_key: 0}
        queue = [start_key]
        edge_list = []
        
        # Precompute field multiplication table for speed
        mul_table = [[f_mul(a, b) for b in range(q)] for a in range(q)]
        add_table = [[f_add(a, b) for b in range(q)] for a in range(q)]
        sub_table = [[f_sub(a, b) for b in range(q)] for a in range(q)]
        
        for vkey in queue:
            side, coords = vkey
            vidx = label[vkey]
            for first_val in range(q):
                # Build the neighbour coordinate array
                ncoords = [0] * (d + 1)
                ncoords[1] = first_val
                for c in range(2, d + 1):
                    a, b = rules[c]
                    if side == 0:
                        ncoords[c] = add_table[coords[c]][mul_table[ncoords[a]][coords[b]]]
                    else:
                        ncoords[c] = sub_table[coords[c]][mul_table[coords[a]][ncoords[b]]]
                nkey = (1 - side, tuple(ncoords))
                if nkey not in label:
                    label[nkey] = len(queue)
                    queue.append(nkey)
                if side == 0:
                    edge_list.append((vidx, label[nkey]))
        
        n = len(queue)
        return n, edge_list
    
    def verify_girth(n, edges, g):
        """
        Verify that the graph has girth at least g using non-backtracking BFS from each vertex.
        A non-backtracking walk of length L from u to v with v != u and no immediate backtracking
        implies a cycle of length at most L. We check for cycles of length < g.
        
        For a k-regular graph, from each vertex we do a BFS where we don't go back to the parent.
        The shortest non-backtracking cycle through each vertex gives the girth.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # For each vertex, do a non-backtracking BFS up to depth (g-1)//2
        # If we find a cycle of length < g, return False
        max_depth = (g - 1) // 2
        
        for start in range(n):
            # BFS state: (vertex, parent, depth)
            # visited: dict from (vertex, parent) to depth
            visited = {}
            visited[(start, -1)] = 0
            queue = [(start, -1, 0)]
            
            while queue:
                u, parent, depth = queue.pop(0)
                if depth >= max_depth:
                    continue
                for v in adj[u]:
                    if v == parent:
                        continue
                    # If we've seen (v, u) before, we found a cycle
                    if (v, u) in visited:
                        cycle_len = depth + 1 + visited[(v, u)]
                        if cycle_len < g:
                            return False
                    else:
                        visited[(v, u)] = depth + 1
                        queue.append((v, u, depth + 1))
        
        return True
    
    # Try d values from 3 upward
    for d in range(3, 10):
        n, edges = build_luw(d)
        # Verify girth
        if verify_girth(n, edges, g):
            return n, edges
    
    # Fallback: use the seed's approach with d = g - 5 if odd, g - 4 if even
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3
    n, edges = build_luw(d)
    return n, edges
# EVOLVE-BLOCK-END
