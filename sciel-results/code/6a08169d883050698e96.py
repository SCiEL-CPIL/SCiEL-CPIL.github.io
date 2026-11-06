import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try n=46
    n = 46
    min_deg_req = n - 39  # 7
    
    # We will try to construct a graph on n vertices
    # that is C4-free and has min degree >= min_deg_req
    
    def is_valid(G, n, min_deg_req):
        # Check min degree
        for u in range(n):
            if len(G[u]) < min_deg_req:
                return False
        # Check C4
        # A C4 exists if there are u < v with |N(u) cap N(v)| >= 2
        # This is O(n^3) roughly, but n=46 is small
        for u in range(n):
            for v in range(u + 1, n):
                if len(G[u].intersection(G[v])) >= 2:
                    return False
        return True

    def find_c4(G, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = G[u].intersection(G[v])
                if len(common) >= 2:
                    # Return the C4 edges
                    # u - w - v - x - u
                    w, x = list(common)[:2]
                    return [(u, w), (w, v), (v, x), (x, u)]
        return None

    def try_construct(n, min_deg_req, max_attempts=100):
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize random graph
            # Start with a random graph with p such that avg degree ~ min_deg_req + 2
            p = (min_deg_req + 2) / n
            G = [set() for _ in range(n)]
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        G[u].add(v)
                        G[v].add(u)
            
            # Phase 1: Remove edges to break C4s
            # We might go below min_deg_req, but we'll fix it later
            steps = 0
            max_steps = 1000
            while steps < max_steps:
                if time.time() > deadline:
                    return None
                c4 = find_c4(G, n)
                if c4 is None:
                    break
                # Remove a random edge from the C4
                u, v = random.choice(c4)
                G[u].remove(v)
                G[v].remove(u)
                steps += 1
            
            # Phase 2: Add edges to increase min degree
            # Only add edges that don't create C4s
            improved = True
            while improved:
                if time.time() > deadline:
                    return None
                improved = False
                # Find vertices with low degree
                low_deg_vertices = [u for u in range(n) if len(G[u]) < min_deg_req]
                if not low_deg_vertices:
                    break
                
                # Try to add an edge incident to a low degree vertex
                # Shuffle to avoid bias
                random.shuffle(low_deg_vertices)
                for u in low_deg_vertices:
                    if len(G[u]) >= min_deg_req:
                        continue
                    # Find non-neighbors v
                    non_neighbors = [v for v in range(n) if v != u and v not in G[u]]
                    random.shuffle(non_neighbors)
                    for v in non_neighbors:
                        # Check if adding (u, v) creates C4
                        # This happens if |N(u) cap N(v)| >= 1
                        # Wait, if |N(u) cap N(v)| >= 1, say w, then u-w-v-?
                        # If w is common neighbor, we have u-w-v.
                        # To form C4, we need another path of length 2 between u and v.
                        # Actually, adding (u,v) creates a C4 iff there exists w, x such that
                        # w in N(u) cap N(v) AND x in N(u) cap N(v) ? No.
                        # If we add (u,v), a C4 is formed if there is a path of length 3 between u and v in the current graph.
                        # Path u - a - b - v.
                        # This exists if N(u) and N(v) have a "connection" of length 1?
                        # No.
                        # Condition for NO new C4:
                        # For any w in N(u), w not in N(v)? No.
                        # If w in N(u) cap N(v), we have u-w-v.
                        # If there is another w' in N(u) cap N(v), we have u-w'-v.
                        # Then u-w-v-w'-u is a C4? No, that's a triangle if w=w'.
                        # If w != w', we have u-w-v-w'-u. This is a C4.
                        # So if |N(u) cap N(v)| >= 2, adding (u,v) creates a C4.
                        # What if |N(u) cap N(v)| == 1?
                        # Let w be the common neighbor.
                        # Is there a path u - a - b - v?
                        # If yes, adding (u,v) creates a C4.
                        # So we need to check if there is a path of length 3.
                        # This is equivalent to: (N(u) \ {v}) and (N(v) \ {u}) have a common neighbor?
                        # i.e., exists a in N(u), b in N(v) such that a-b is an edge?
                        # And a != v, b != u.
                        # Actually, simpler:
                        # Adding (u,v) creates a C4 iff there exists a, b such that
                        # a in N(u), b in N(v), a != v, b != u, and a-b is an edge.
                        # AND a != b?
                        # If a=b, then a is common neighbor.
                        # If a=b, we have u-a-v.
                        # If there is another common neighbor, we have C4.
                        # If only one common neighbor a, and no other path, no C4.
                        
                        # So condition to ADD (u,v):
                        # 1. No common neighbors? If |N(u) cap N(v)| == 0, safe.
                        # 2. If |N(u) cap N(v)| == 1, let w be the common neighbor.
                        #    Check if there is any other path of length 3.
                        #    i.e., exists a in N(u) \ {w, v} and b in N(v) \ {w, u} such that a-b is an edge?
                        #    Or simply: does the subgraph induced by N(u) U N(v) contain an edge between N(u)\{w,v} and N(v)\{w,u}?
                        
                        # For simplicity and speed, we can just try to add and then check if C4 exists.
                        # But that's slow.
                        # Let's use the condition:
                        # If |N(u) cap N(v)| >= 2, skip.
                        # If |N(u) cap N(v)| == 1, check if there is an edge between (N(u) - {w, v}) and (N(v) - {w, u}).
                        # If |N(u) cap N(v)| == 0, safe.
                        
                        common = G[u].intersection(G[v])
                        if len(common) >= 2:
                            continue
                        if len(common) == 1:
                            w = next(iter(common))
                            # Check for other paths
                            # Neighbors of u excluding v and w
                            nu = G[u] - {v, w}
                            # Neighbors of v excluding u and w
                            nv = G[v] - {u, w}
                            # Is there an edge between nu and nv?
                            found_path = False
                            for a in nu:
                                if G[a].intersection(nv):
                                    found_path = True
                                    break
                            if found_path:
                                continue
                        
                        # Safe to add
                        G[u].add(v)
                        G[v].add(u)
                        improved = True
                        break # Restart loop to update degrees
            
            # Final check
            if is_valid(G, n, min_deg_req):
                edges = []
                for u in range(n):
                    for v in G[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
        
        return None

    result = try_construct(46, 7)
    if result:
        return result
        
    # Fallback to n=39
    return {'n': 39, 'edges': []}
