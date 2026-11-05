import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        target_min_deg = n - 39
        
        found = False
        max_restarts = 100  # More restarts possible with faster check
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize empty graph
            adj = [set() for _ in range(n)]
            degrees = [0] * n
            edges = []
            
            # Key insight: adding edge (u,v) creates a C4 iff |N(u) ∩ N(v)| >= 2
            # So we can check in O(min(deg(u),deg(v))) using set intersection
            
            # Greedy construction with random ordering
            # Use a priority-based approach: always try to increase the minimum degree vertex
            progress = True
            while progress and time.time() < deadline:
                progress = False
                # Find the vertex with minimum degree
                min_deg = min(degrees)
                # Find all vertices with this degree
                min_verts = [i for i in range(n) if degrees[i] == min_deg]
                
                # Try to add an edge incident to one of these vertices
                random.shuffle(min_verts)
                
                added = False
                for u in min_verts:
                    if added:
                        break
                    # Potential neighbors for u
                    potential_vs = [v for v in range(n) if v != u and v not in adj[u]]
                    random.shuffle(potential_vs)
                    
                    for v in potential_vs:
                        # Check if adding (u,v) creates a C4
                        # C4 exists iff |N(u) ∩ N(v)| >= 2
                        common = adj[u] & adj[v]
                        if len(common) < 2:
                            # Safe to add
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
        if time.time() > deadline:
            break

    # Fallback: return the seed result or a known valid small graph
    return {'n': 39, 'edges': []}
