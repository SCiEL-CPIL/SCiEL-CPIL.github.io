"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj, n):
    dist = [[-1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = [i]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)
    return dist

def _get_diameter(dist):
    max_d = 0
    for i in range(len(dist)):
        for j in range(len(dist)):
            if dist[i][j] > max_d:
                max_d = dist[i][j]
    return max_d

def _is_triangle_free(adj, u, v, n):
    # Check if adding (u,v) creates a triangle
    # Triangle exists if there is a common neighbor w
    # such that (u,w) and (v,w) are in adj
    # We only need to check common neighbors in the current graph
    neighbors_u = set(adj[u])
    neighbors_v = set(adj[v])
    common = neighbors_u.intersection(neighbors_v)
    if common:
        return False
    return True

def _construct_greedy():
    n = P['n']
    # Initialize adjacency list
    adj = [[] for _ in range(n)]
    base_edges = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    added_edges = []
    
    # Calculate initial diameter
    dist = _bfs_distances(adj, n)
    current_diam = _get_diameter(dist)
    
    # Heuristic: While diameter > 4, add edge that reduces diameter most
    # To avoid infinite loops or getting stuck, limit iterations
    max_iters = 50
    iter_count = 0
    
    while current_diam > 4 and iter_count < max_iters:
        iter_count += 1
        # Find pair with max distance
        max_d = -1
        best_u, best_v = -1, -1
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
                    best_u, best_v = i, j
        
        if best_u == -1:
            break
            
        # Try to add edge (best_u, best_v)
        if (best_u, best_v) in base_edges:
            # Should not happen if we track added edges properly, but base edges are fixed
            # We only add non-base edges
            pass
        else:
            # Check triangle freedom
            if _is_triangle_free(adj, best_u, best_v, n):
                # Add edge
                adj[best_u].append(best_v)
                adj[best_v].append(best_u)
                base_edges.add((best_u, best_v))
                base_edges.add((best_v, best_u))
                added_edges.append([best_u, best_v])
                # Recalculate distances
                dist = _bfs_distances(adj, n)
                current_diam = _get_diameter(dist)
            else:
                # If adding (best_u, best_v) creates triangle, try next best pair?
                # For simplicity in this step, we might just break or try a different strategy
                # Let's try to find any valid edge that reduces diameter
                # This simple greedy might fail if max pair is blocked.
                # Let's add a fallback: try all pairs sorted by distance desc
                found = False
                for i in range(n):
                    for j in range(i+1, n):
                        if i == best_u and j == best_v:
                            continue
                        if dist[i][j] < max_d: # Only consider if it's also a "long" edge?
                             # Actually, just try to reduce diameter
                            pass
                        if (i, j) in base_edges:
                            continue
                        if _is_triangle_free(adj, i, j, n):
                            # Check if this edge reduces diameter
                            # Temporarily add
                            adj[i].append(j)
                            adj[j].append(i)
                            temp_dist = _bfs_distances(adj, n)
                            temp_diam = _get_diameter(temp_dist)
                            # Remove
                            adj[i].remove(j)
                            adj[j].remove(i)
                            
                            if temp_diam < current_diam:
                                # Commit
                                adj[i].append(j)
                                adj[j].append(i)
                                base_edges.add((i, j))
                                base_edges.add((j, i))
                                added_edges.append([i, j])
                                dist = temp_dist
                                current_diam = temp_diam
                                found = True
                                break
                    if found:
                        break
                if not found:
                    # Stuck?
                    break

    return added_edges

def _verify_solution(added_edges):
    n = P['n']
    adj = [[] for _ in range(n)]
    base_edges = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    for u, v in added_edges:
        if (u, v) in base_edges:
            return False
        adj[u].append(v)
        adj[v].append(u)
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # Check triangle free
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    # Triangle u-v-w
                    return False
                    
    # Check connected
    dist = _bfs_distances(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1:
                return False
                
    # Check diameter
    diam = _get_diameter(dist)
    if diam > 4:
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50 # Safety margin
    
    # Try greedy
    added = _construct_greedy()
    
    if time.time() > deadline:
         # Fallback to seed-like solution if time runs out
         # But seed adds many edges. We want few.
         # If greedy fails, return a known valid small set if we have one, 
         # or just return the greedy result if it's valid.
         pass

    if _verify_solution(added):
        return added
    else:
        # If greedy failed, try a hardcoded solution or fallback
        # Let's try a specific pattern
        # (0,6), (6,12), (12,18), (18,23) failed diameter.
        # Let's try (0,5), (5,10), (10,15), (15,20), (20,23)
        # We verified diameter was 5.
        # Let's try (0,4), (4,8), (8,12), (12,16), (16,20), (20,23)
        # This is getting complicated.
        
        # Fallback: Use the seed's logic but it's not minimal.
        # However, we must return *something* valid.
        # The seed returns a valid solution (dense).
        base={tuple(e) for e in P['edges']}
        fallback = [[u,v] for u in range(P['n']) for v in range(u+1,P['n']) if (u-v)%2 and (u,v) not in base]
        
        if _verify_solution(fallback):
            return fallback
        else:
            return []

# EVOLVE-BLOCK-END
