import time
import random
import numpy as np

def _check_c5(adj):
    """Check if graph with adjacency matrix adj contains a C5."""
    n = adj.shape[0]
    # A 5-cycle exists if there is a path of length 4 from u to v where u and v are adjacent
    # and the path does not use the edge (u,v).
    # Simpler: iterate over all pairs (u,v) with adj[u,v]=1.
    # Check if there is a path of length 4 from u to v in the graph with edge (u,v) removed.
    # This is O(n^3 * n^2) which is okay for n=35.
    # Even simpler: count paths of length 4.
    # Number of 5-cycles = (1/10) * trace(A^5) - (number of 3-cycles * ...) -- this is complex.
    # Let's use a simple DFS/BFS for each edge (u,v) to see if there is a path of length 4 connecting u and v without using (u,v).
    
    # Actually, for n=35, we can just check all combinations of 5 vertices? No, too slow.
    # Let's use the property: C5 is a cycle of length 5.
    # We can use a standard algorithm.
    
    # Optimization: If the graph is a union of cliques, no C5.
    # In random search, we might create C5s.
    
    # Efficient check:
    # For each edge (u, v), remove it. Check if dist(u, v) <= 4.
    # If dist(u, v) == 4, there is a C5.
    # If dist(u, v) < 4, there is a shorter cycle, but we only care about C5.
    # Wait, if dist(u, v) == 4 in G \ {(u,v)}, then u-v-...-...-...-u is a 5-cycle.
    
    edges = np.where(adj > 0)
    for i in range(len(edges[0])):
        u, v = edges[0][i], edges[1][i]
        # Temporarily remove edge
        adj[u, v] = 0
        adj[v, u] = 0
        
        # BFS from u to find distance to v
        visited = np.zeros(n, dtype=bool)
        dist = np.full(n, -1, dtype=int)
        queue = [u]
        visited[u] = True
        dist[u] = 0
        head = 0
        found = False
        while head < len(queue):
            curr = queue[head]
            head += 1
            if dist[curr] >= 4: # We only care if we can reach v in exactly 4 steps
                # If we reach v in 4 steps, it's a C5.
                # If we reach v in < 4 steps, it's a C3 or C4, not C5.
                # But we need to know if there is a path of length 4.
                # BFS finds shortest path. If shortest path is 4, then yes.
                # If shortest path is < 4, there is no C5 using this edge as the closing edge?
                # Actually, if there is a C5, the distance between u and v in G\{u,v} must be 4.
                # If distance is < 4, the cycle is shorter.
                # So we just need to check if dist[u][v] == 4.
                pass
            # Standard BFS
            neighbors = np.where(adj[curr] > 0)[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    dist[nb] = dist[curr] + 1
                    if dist[nb] == 4 and nb == v:
                        found = True
                        break
                    if dist[nb] < 4:
                        queue.append(nb)
            if found:
                break
        
        # Restore edge
        adj[u, v] = 1
        adj[v, u] = 1
        
        if found:
            return True
            
    return False

def _alpha_ge(adj, k):
    """Check if independence number is at least k."""
    n = adj.shape[0]
    # Use a simple backtracking to find an independent set of size k.
    # If found, return True.
    
    # Greedy initial bound?
    # Just use a recursive search with pruning.
    
    def backtrack(start, current_set, remaining_needed):
        if remaining_needed == 0:
            return True
        # Pruning: if not enough vertices left, return False
        if n - start < remaining_needed:
            return False
        
        # Try to add vertices from 'start' to n-1
        for i in range(start, n):
            # Check if i is independent from all in current_set
            if all(adj[i, j] == 0 for j in current_set):
                current_set.append(i)
                if backtrack(i + 1, current_set, remaining_needed - 1):
                    return True
                current_set.pop()
        return False

    return backtrack(0, [], k)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Base solution: n=32, 8 disjoint K4s
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                best_edges.append([base + i, base + j])
        
    # Try to find better n (33, 34, 35)
    for target_n in [35, 34, 33]:
        if time.time() > deadline - 10: # Leave 10s for final checks
            break
            
        # Local search for valid graph of size target_n
        # Objective: Minimize penalty = C5_count + 10 * max(0, alpha - 8)
        # We want penalty == 0.
        
        # Initialize with random graph
        adj = np.random.randint(0, 2, (target_n, target_n))
        adj = np.triu(adj, 1)
        adj = adj + adj.T
        np.fill_diagonal(adj, 0)
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.9999
        max_iter = 20000
        current_penalty = float('inf')
        
        # Initial penalty
        # This check is expensive, so we do it every few steps or use a fast approximation?
        # For n=35, _check_c5 and _alpha_ge are fast enough for ~20k iterations?
        # _check_c5 is O(E * BFS). E ~ 300. BFS is O(V+E). 300 * 350 ~ 100k ops.
        # 20k * 100k = 2e9 ops. Might be too slow in Python.
        
        # We need a faster heuristic or fewer iterations.
        # Let's reduce iterations and rely on random restarts.
        
        found = False
        restarts = 0
        while not found and time.time() < deadline - 5:
            # Reinitialize if stuck
            if restarts > 0:
                adj = np.random.randint(0, 2, (target_n, target_n))
                adj = np.triu(adj, 1)
                adj = adj + adj.T
                np.fill_diagonal(adj, 0)
                T = 1.0
                
            restarts += 1
            
            # Run a short SA cycle
            for i in range(5000):
                if time.time() > deadline - 5:
                    break
                    
                # Pick random edge to flip
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                if adj[u, v] == adj[v, u]: # Should be symmetric
                    pass
                
                # Flip
                old_val = adj[u, v]
                adj[u, v] = 1 - old_val
                adj[v, u] = adj[u, v]
                
                # Check validity
                # To speed up, we can check C5 first (often the bottleneck for random graphs)
                # Or check Alpha first?
                
                # Fast check: if we just added an edge, it might create C5.
                # If we removed an edge, it might fix C5.
                
                # Let's just do the full check every 10 steps to save time?
                # No, we need to accept/reject based on penalty.
                
                # Let's try to be smarter.
                # If old_val == 0 (adding edge):
                #   It can only create C5s. It cannot increase alpha.
                #   So if alpha was < 9, it stays < 9.
                #   We only need to check if C5 is created.
                # If old_val == 1 (removing edge):
                #   It can only remove C5s. It cannot decrease alpha (might increase).
                #   So if C5 count was 0, it stays 0.
                #   We only need to check if alpha becomes >= 9.
                
                # This is a huge speedup!
                
                # We need to track current C5 status and Alpha status.
                # But state is global.
                
                # Let's maintain:
                # has_c5: bool
                # alpha_ge_9: bool
                
                # If we add an edge:
                #   new_has_c5 = has_c5 or _check_c5_specific(adj, u, v) # Check if this edge is in a C5
                #   new_alpha_ge_9 = alpha_ge_9 # Adding edge cannot create independent set
                #   Actually, adding edge can *break* independent sets, so alpha might decrease.
                #   So if alpha_ge_9 was True, it might become False.
                #   So we must recheck alpha if it was True?
                #   Or simpler: Just recheck both if we are close to valid?
                
                # Let's stick to full checks but optimize the loop.
                # 5000 iterations * full check might be too slow.
                # Let's reduce to 1000 iterations per restart, more restarts.
                
                # Actually, let's just try a few random graphs and check them.
                # Random graphs are unlikely to be valid.
                # We need to start from a valid structure and modify.
                
                # Better Strategy:
                # Start with n=32 valid graph.
                # Add vertex 32.
                # Connect it to some vertices.
                # Check validity.
                # If invalid, flip edges incident to 32.
                
                if restarts == 1:
                    # Initialize with known valid n=32 + 1 vertex
                    if target_n == 33:
                        adj = np.zeros((33, 33), dtype=int)
                        for b in range(8):
                            base = 4 * b
                            for i in range(4):
                                for j in range(i + 1, 4):
                                    adj[base + i, base + j] = 1
                                    adj[base + j, base + i] = 1
                        # Vertex 32 is isolated initially.
                        # We will connect it.
                    
                    elif target_n == 34:
                        adj = np.zeros((34, 34), dtype=int)
                        # Use 8 K4s (32 verts)
                        for b in range(8):
                            base = 4 * b
                            for i in range(4):
                                for j in range(i + 1, 4):
                                    adj[base + i, base + j] = 1
                                    adj[base + j, base + i] = 1
                        # Verts 32, 33 isolated.
                        
                    elif target_n == 35:
                        adj = np.zeros((35, 35), dtype=int)
                        for b in range(8):
                            base = 4 * b
                            for i in range(4):
                                for j in range(i + 1, 4):
                                    adj[base + i, base + j] = 1
                                    adj[base + j, base + i] = 1
                        # Verts 32, 33, 34 isolated.

                # Local search on the "new" vertices or all edges?
                # Flipping all edges is risky.
                # Let's flip random edges.
                
                # Check current state
                # This is still expensive.
                
                # Let's abort the complex SA and just do a simple random search with a hard time limit per target_n.
                # If we don't find it, we move on.
                
                # Given the complexity, I will simplify the search to:
                # 1. Try to construct n=33 by adding a vertex to the 8-K4 graph and optimizing its connections.
                # 2. If that fails, return n=32.
                
                # This is a heuristic.
                
                pass # Break out of inner loop to use simpler strategy below

        # Simpler Strategy Implementation
        if target_n == 33:
            # Base: 8 K4s (0..31)
            # New vertex 32
            # We want to connect 32 to some vertices such that:
            # 1. No C5 is formed.
            # 2. No independent set of size 9.
            
            # If 32 is connected to no one, alpha = 9 (invalid).
            # If 32 is connected to everyone, alpha = 8 (valid for alpha).
            # But connecting to everyone might create C5s?
            # If 32 is connected to u, v, w, x, y...
            # A C5 involving 32 would be 32-u-v-w-x-32.
            # This requires u-v, v-w, w-x, x-32, 32-u to be edges.
            # Since 32 is connected to all, we just need a path of length 3 between any two neighbors of 32.
            # In a union of K4s, the max distance between vertices in different K4s is 2 (via no edge? No, disconnected).
            # Wait, the base graph is DISJOINT.
            # So there are NO edges between K4s.
            # If we connect 32 to vertices in DIFFERENT K4s, say u in K4_0 and v in K4_1.
            # Path 32-u-...-v-32.
            # Since K4_0 and K4_1 are disconnected, there is no path between u and v.
            # So no C5 can be formed involving vertices from different components, UNLESS we add edges between components.
            
            # If we only connect 32 to vertices within a SINGLE K4, say K4_0.
            # Let 32 connect to u, v in K4_0.
            # C5: 32-u-w-x-v-32?
            # In K4, all vertices are connected.
            # If 32 connects to u, v, w in K4_0.
            # Cycle 32-u-v-w-32 is a C4.
            # Cycle 32-u-v-w-x-32 requires 4 neighbors in K4.
            # K4 has only 4 vertices.
            # If 32 connects to all 4 vertices of K4_0:
            # 32 is connected to u, v, w, x.
            # Edges: 32-u, u-v, v-w, w-x, x-32. This is a C5!
            # So 32 cannot connect to all 4 vertices of a K4.
            
            # If 32 connects to 3 vertices of K4_0:
            # 32-u, u-v, v-w, w-32 is C4.
            # No C5.
            
            # So, connect 32 to 3 vertices of K4_0.
            # Does this create alpha >= 9?
            # Base alpha = 8 (one from each K4).
            # If we pick the independent set, we can pick the 4th vertex of K4_0 (not connected to 32) and one from each other K4.
            # That's 1 + 7 = 8.
            # Can we pick 32?
            # If we pick 32, we cannot pick any of its 3 neighbors in K4_0.
            # We can pick the 4th vertex of K4_0.
            # We can pick one from each of the other 7 K4s.
            # Total: 1 (32) + 1 (K4_0 remaining) + 7 (others) = 9.
            # So alpha >= 9. Invalid.
            
            # We need to prevent this.
            # We need to connect 32 to MORE vertices, or connect vertices between K4s.
            
            # If we connect 32 to all 4 vertices of K4_0, we get C5.
            # If we connect 32 to 3 vertices of K4_0 AND 1 vertex of K4_1?
            # Neighbors: u,v,w in K4_0, x in K4_1.
            # Independent set containing 32:
            # Cannot pick u,v,w. Can pick 4th of K4_0.
            # Cannot pick x. Can pick other 3 of K4_1.
            # Can pick 1 from each other 6 K4s.
            # Total: 1 (32) + 1 (K4_0) + 3 (K4_1) + 6 (others) = 11. Much worse.
            
            # We need to reduce the independent set size.
            # This implies we need to add edges between the K4s to break the independence.
            
            # This suggests the 8-K4 structure is not easily extendable to 33 without significant restructuring.
            
            pass

    # Fallback to n=32
    return {'n': best_n, 'edges': best_edges}
