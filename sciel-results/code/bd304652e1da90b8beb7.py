import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n = 18
    # Initial graph: K6,6,6
    # Parts: 0-5, 6-11, 12-17
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    
    # Adjacency matrix for fast lookup
    # adj[u][v] = 1 if edge exists
    adj = [[0]*n for _ in range(n)]
    
    # Initialize K6,6,6
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u][v] = 1
                    adj[v][u] = 1
    
    def is_k4_free(adj):
        # Check for K4
        # Optimized: For each edge (u,v), check if they share 2 common neighbors that are connected?
        # No, K4 is 4 vertices all connected.
        # For each pair (u,v), let C = common neighbors.
        # If there is an edge (w,x) in C, then u,v,w,x is K4?
        # No, u,v,w,x is K4 if w-x is edge AND u-w, u-x, v-w, v-x are edges.
        # Since w,x are common neighbors, u-w, v-w, u-x, v-x are 1.
        # So we just need to check if the subgraph induced by C has an edge.
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    common = []
                    for w in range(n):
                        if w != u and w != v and adj[u][w] and adj[v][w]:
                            common.append(w)
                    # Check if any two in common are connected
                    for i in range(len(common)):
                        for j in range(i+1, len(common)):
                            if adj[common[i]][common[j]]:
                                return False
        return True

    def find_triangle_free_set_size(adj):
        # Greedy upper bound on largest triangle-free induced subgraph
        # Start with all vertices
        vertices = list(range(n))
        while True:
            # Find a triangle
            tri_found = False
            for i in range(len(vertices)):
                u = vertices[i]
                for j in range(i+1, len(vertices)):
                    v = vertices[j]
                    if adj[u][v]:
                        for k in range(j+1, len(vertices)):
                            w = vertices[k]
                            if adj[v][w] and adj[u][w]:
                                # Triangle u,v,w
                                # Remove w
                                vertices.pop(k)
                                tri_found = True
                                break
                        if tri_found:
                            break
                    # Optimization: if u and v not connected, no triangle with u,v
                if tri_found:
                    break
            if not tri_found:
                break
        return len(vertices)

    best_adj = [row[:] for row in adj]
    best_score = find_triangle_free_set_size(best_adj)
    
    # Local Search
    current_adj = [row[:] for row in adj]
    current_score = best_score
    
    while time.time() < deadline:
        # Try random edge flip
        # Pick random u, v
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
            
        old_val = current_adj[u][v]
        new_val = 1 - old_val
        
        # Apply change
        current_adj[u][v] = new_val
        current_adj[v][u] = new_val
        
        # Check K4-free
        if not is_k4_free(current_adj):
            # Revert
            current_adj[u][v] = old_val
            current_adj[v][u] = old_val
            continue
            
        # Estimate score
        # To save time, only compute score if potential improvement
        # If we added an edge, it might reduce the max triangle-free set
        # If we removed an edge, it might increase it (bad)
        if new_val == 1:
            score = find_triangle_free_set_size(current_adj)
            if score < best_score:
                best_score = score
                best_adj = [row[:] for row in current_adj]
                # If we hit 11, we can stop or keep searching for 10
                if best_score <= 10:
                    break
            # If score is same, keep it? Or revert?
            # Adding edges generally helps minimize triangle-free sets.
            # So we keep the graph.
        else:
            # Removed edge. Likely worse.
            # Revert immediately to save time?
            # Or check?
            # Let's revert to avoid degrading the graph.
            current_adj[u][v] = old_val
            current_adj[v][u] = old_val
            continue

    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                edges.append([u, v])
                
    return edges
