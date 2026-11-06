import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(all_edges)
    
    # Helper to check if a set of 8 vertices contains a Q3
    # This specific check is used inside the search
    def is_q3_present(graph_adj, v, a, b, c):
        # v, a, b, c are the center and its 3 neighbors in the potential Q3
        # a, b, c must be pairwise non-adjacent
        if (a, b) in graph_adj or (b, a) in graph_adj: return False
        if (a, c) in graph_adj or (c, a) in graph_adj: return False
        if (b, c) in graph_adj or (c, b) in graph_adj: return False
        
        # Find candidates for x, y, z, w
        # x is neighbor of a and b
        # y is neighbor of a and c
        # z is neighbor of b and c
        # w is neighbor of x, y, z
        
        # Get neighbors
        na = graph_adj.get(a, set())
        nb = graph_adj.get(b, set())
        nc = graph_adj.get(c, set())
        
        # x candidates: in na and nb
        xs = na & nb
        # y candidates: in na and nc
        ys = na & nc
        # z candidates: in nb and nc
        zs = nb & nc
        
        # Filter out v, a, b, c
        forbidden = {v, a, b, c}
        xs = xs - forbidden
        ys = ys - forbidden
        zs = zs - forbidden
        
        for x in xs:
            for y in ys:
                for z in zs:
                    if x == y or y == z or x == z:
                        continue
                    # Check edges (x,y), (x,z), (y,z) should NOT exist in Q3?
                    # In Q3: 011(x), 101(y), 110(z).
                    # 011-101? No. 011-110? No. 101-110? No.
                    # So x, y, z must be pairwise non-adjacent.
                    if (x, y) in graph_adj or (y, x) in graph_adj: continue
                    if (x, z) in graph_adj or (z, x) in graph_adj: continue
                    if (y, z) in graph_adj or (z, y) in graph_adj: continue
                    
                    # Now find w
                    # w must be neighbor of x, y, z
                    nx = graph_adj.get(x, set())
                    ny = graph_adj.get(y, set())
                    nz = graph_adj.get(z, set())
                    
                    ws = nx & ny & nz
                    ws = ws - forbidden - {x, y, z}
                    
                    for w in ws:
                        # Check if w is connected to x, y, z (already ensured by intersection)
                        # Check if w is connected to a, b, c? No.
                        # In Q3: 111(w) is not connected to 001(a), 010(b), 100(c).
                        if (w, a) in graph_adj or (a, w) in graph_adj: continue
                        if (w, b) in graph_adj or (b, w) in graph_adj: continue
                        if (w, c) in graph_adj or (c, w) in graph_adj: continue
                        
                        # If we reach here, we found a Q3
                        return True
        return False

    def find_q3(graph_adj):
        # Iterate over all vertices v
        for v in range(n):
            neighbors = list(graph_adj.get(v, set()))
            if len(neighbors) < 3:
                continue
            # Iterate over triples of neighbors
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        a, b, c = neighbors[i], neighbors[j], neighbors[k]
                        if is_q3_present(graph_adj, v, a, b, c):
                            # Return the edges of this Q3 for removal
                            # Edges: (v,a), (v,b), (v,c)
                            # (a,x), (b,x), (a,y), (c,y), (b,z), (c,z)
                            # (x,w), (y,w), (z,w)
                            # We need to reconstruct x,y,z,w to get edges
                            # This is a bit messy to return just the edge list.
                            # Instead, let's just return one edge from the Q3 to remove.
                            # Or better, return the set of vertices involved.
                            # For now, just return the edge (v,a) as a candidate for removal?
                            # No, we should remove a random edge from the Q3.
                            # Let's find the specific edges.
                            
                            # Re-run the inner logic to get x,y,z,w
                            na = graph_adj.get(a, set())
                            nb = graph_adj.get(b, set())
                            nc = graph_adj.get(c, set())
                            forbidden = {v, a, b, c}
                            xs = (na & nb) - forbidden
                            ys = (na & nc) - forbidden
                            zs = (nb & nc) - forbidden
                            
                            for x in xs:
                                for y in ys:
                                    for z in zs:
                                        if x == y or y == z or x == z: continue
                                        if (x, y) in graph_adj or (y, x) in graph_adj: continue
                                        if (x, z) in graph_adj or (z, x) in graph_adj: continue
                                        if (y, z) in graph_adj or (z, y) in graph_adj: continue
                                        
                                        nx = graph_adj.get(x, set())
                                        ny = graph_adj.get(y, set())
                                        nz = graph_adj.get(z, set())
                                        ws = (nx & ny & nz) - forbidden - {x, y, z}
                                        
                                        for w in ws:
                                            if (w, a) in graph_adj or (a, w) in graph_adj: continue
                                            if (w, b) in graph_adj or (b, w) in graph_adj: continue
                                            if (w, c) in graph_adj or (c, w) in graph_adj: continue
                                            
                                            q3_edges = [
                                                (v, a), (v, b), (v, c),
                                                (a, x), (b, x),
                                                (a, y), (c, y),
                                                (b, z), (c, z),
                                                (x, w), (y, w), (z, w)
                                            ]
                                            return q3_edges
        return None

    # Initialize with a random graph
    # Start with a high edge count and remove until valid
    # Or start with a known good seed?
    # Let's start with a random graph of 50 edges
    random.seed(42)
    
    best_edges = []
    best_count = 0
    
    for restart in range(50): # Multiple restarts
        if time.time() > deadline - 5:
            break
            
        # Generate random graph
        # Use a biased random graph to be dense but not too dense
        # p = 0.4 -> ~48 edges
        current_edges = set()
        for e in all_edges:
            if random.random() < 0.4:
                current_edges.add(e)
                
        # Build adjacency list
        def build_adj(edge_set):
            adj = {i: set() for i in range(n)}
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            return adj
        
        adj = build_adj(current_edges)
        
        # Remove Q3s
        steps = 0
        while steps < 1000:
            if time.time() > deadline - 5:
                break
            q3_edges = find_q3(adj)
            if q3_edges is None:
                break
            # Remove a random edge from the Q3
            e_to_remove = random.choice(q3_edges)
            # Normalize edge
            if e_to_remove[0] > e_to_remove[1]:
                e_to_remove = (e_to_remove[1], e_to_remove[0])
            if e_to_remove in current_edges:
                current_edges.remove(e_to_remove)
                adj[e_to_remove[0]].remove(e_to_remove[1])
                adj[e_to_remove[1]].remove(e_to_remove[0])
            steps += 1
            
        # Now current_edges is Q3-free
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
        # Try to add edges (Hill Climbing)
        # Shuffle edges not in current_edges
        missing = [e for e in all_edges if e not in current_edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 5:
                break
            # Try adding e
            u, v = e
            # Check if adding e creates a Q3
            # We can just add it, check for Q3, and remove if found
            # But checking for Q3 is expensive.
            # Better: Add e, find Q3. If Q3 found, check if e is in it.
            # If e is in Q3, we can't add it.
            # If e is NOT in Q3, we can add it? No, the Q3 might have been there before?
            # No, we know the graph was Q3-free. So any new Q3 MUST contain e.
            
            current_edges.add(e)
            adj[u].add(v)
            adj[v].add(u)
            
            q3_edges = find_q3(adj)
            
            if q3_edges is not None:
                # Check if e is in this Q3
                e_norm = (u, v) if u < v else (v, u)
                # The q3_edges returned are tuples.
                # We need to check if e_norm is in q3_edges
                # Note: find_q3 returns edges as (u,v) or (v,u)?
                # In is_q3_present, we use graph_adj which is symmetric.
                # The returned edges are constructed from variables.
                # Let's just check if any edge in q3_edges is e or reversed e.
                contains_e = False
                for qe in q3_edges:
                    if (qe[0] == u and qe[1] == v) or (qe[0] == v and qe[1] == u):
                        contains_e = True
                        break
                
                if contains_e:
                    # Cannot add e
                    current_edges.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
                else:
                    # This shouldn't happen if graph was Q3-free
                    # But if it does, we have a bug or the Q3 was pre-existing?
                    # Since we verified Q3-free, this case implies logic error.
                    # For safety, remove e.
                    current_edges.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
            else:
                # e can be added
                pass # Already added
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
    # Return the best found
    # Ensure format is list of lists [u, v]
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
