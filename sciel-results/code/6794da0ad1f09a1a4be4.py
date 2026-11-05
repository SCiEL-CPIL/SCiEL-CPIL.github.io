import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        target_min_deg = n - 39
        
        # If we already found a valid graph for this n in a previous run (not applicable here, single run)
        # We try multiple restarts for each n
        
        found = False
        max_restarts = 50  # Adjust based on time
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize empty graph
            adj = [set() for _ in range(n)]
            degrees = [0] * n
            edges = []
            
            # We can start with a random set of edges that are valid?
            # Or just start empty and add greedily.
            
            # Greedy construction
            # To make it more robust, we can shuffle the order of vertices or edges
            # But the validity depends on the graph state.
            
            # Let's try a "greedy with random tie-breaking"
            # We maintain a list of valid edges. Recalculating all valid edges is O(n^4) or so.
            # n=47, n^4 is ~5M. Might be slow in Python if done naively.
            # We can optimize the check.
            
            # Check if (u, v) can be added:
            # No edge between N(u) and N(v).
            
            def can_add(u, v):
                if v in adj[u]:
                    return False
                # Check edges between N(u) and N(v)
                # Optimize: iterate over smaller set
                nu = adj[u]
                nv = adj[v]
                if len(nu) > len(nv):
                    nu, nv = nv, nu
                for w in nu:
                    if w in nv:
                        # w is common neighbor. This is fine (triangle).
                        # We need to check if w is connected to any x in nv?
                        # Wait. The condition is "No edge between N(u) and N(v)".
                        # If w is in N(u) AND w is in N(v), then w-w is not an edge (simple graph).
                        # But we need to check if there is ANY edge (a, b) with a in N(u), b in N(v).
                        # If a=b=w, it's not an edge.
                        # So we need to check pairs (a,b) with a!=b.
                        pass
                # Better check:
                # For each w in N(u), check if any neighbor of w is in N(v).
                # This is O(deg(u) * avg_deg).
                for w in adj[u]:
                    for x in adj[w]:
                        if x in adj[v]:
                            # Path u-w-x-v exists
                            return False
                return True

            # This check is O(deg(u) * deg(w)).
            
            # Let's try to construct the graph.
            # To speed up, we can keep a list of candidate edges.
            # But the set of valid edges changes as we add edges.
            
            # Simple loop:
            progress = True
            while progress and time.time() < deadline:
                progress = False
                # Find the vertex with minimum degree
                min_deg = min(degrees)
                # Find all vertices with this degree
                min_verts = [i for i in range(n) if degrees[i] == min_deg]
                
                # Try to add an edge incident to one of these vertices
                # Shuffle min_verts for randomness
                random.shuffle(min_verts)
                
                added = False
                for u in min_verts:
                    if added:
                        break
                    # Potential neighbors for u
                    # We want to pick v such that (u,v) is valid.
                    # We can iterate over all v != u.
                    # To add randomness, shuffle the list of potential v's
                    potential_vs = [v for v in range(n) if v != u and v not in adj[u]]
                    random.shuffle(potential_vs)
                    
                    for v in potential_vs:
                        if can_add(u, v):
                            # Add edge (u, v)
                            adj[u].add(v)
                            adj[v].add(u)
                            degrees[u] += 1
                            degrees[v] += 1
                            edges.append([u, v])
                            progress = True
                            added = True
                            break
                            
            # Check if we met the requirement
            if min(degrees) >= target_min_deg:
                return {'n': n, 'edges': edges}
                
        # If we exhausted attempts for this n, try smaller n
        # But we should check time
        if time.time() > deadline:
            break

    # Fallback: return the seed result or a known valid small graph
    # The seed returns n=39, edges=[]
    # For n=39, target_min_deg = 0. Empty graph works.
    return {'n': 39, 'edges': []}
