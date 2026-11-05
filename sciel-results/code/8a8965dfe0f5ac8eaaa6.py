import numpy as np
import time
import itertools

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    # Precompute triangle indices
    tri_list = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri_list.append((i, j, k))
    tri_arr = np.array(tri_list)
    I = tri_arr[:, 0]
    J = tri_arr[:, 1]
    K = tri_arr[:, 2]
    n_tri = len(tri_arr)

    # Initialize best_adj with K_{8,8} + edge (0,1)
    best_adj = np.zeros((n, n), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            best_adj[u, v] = 1
            best_adj[v, u] = 1
    best_adj[0, 1] = 1
    best_adj[1, 0] = 1

    def get_state(adj):
        deg = adj.sum(axis=1)
        valid = (adj[I, J] == 1) & (adj[J, K] == 1) & (adj[K, I] == 1)
        if not np.any(valid):
            return 0, np.zeros(n_tri, dtype=np.int32)
        sums = deg[I[valid]] + deg[J[valid]] + deg[K[valid]]
        all_sums = np.zeros(n_tri, dtype=np.int32)
        all_sums[valid] = sums
        return int(np.max(all_sums)), all_sums

    best_score, best_sums = get_state(best_adj)
    current_adj = best_adj.copy()
    current_score, current_sums = best_score, best_sums.copy()

    # Precompute triangles incident to each vertex for fast updates
    # tri_incident[v] = list of triangle indices involving v
    tri_incident = [[] for _ in range(n)]
    for t_idx in range(n_tri):
        u, v, w = tri_arr[t_idx]
        tri_incident[u].append(t_idx)
        tri_incident[v].append(t_idx)
        tri_incident[w].append(t_idx)
    
    # Convert to arrays for speed
    tri_incident_arr = [np.array(lst) for lst in tri_incident]

    def update_state(adj, sums, u, v, added):
        """
        Update degrees and triangle sums after adding/removing edge (u,v).
        Returns new score and updated sums array.
        """
        deg = adj.sum(axis=1) # Recompute deg is fast for n=16
        
        # Identify affected triangles: those containing u or v
        # A triangle is affected if it contains u or v.
        # We need to recalculate validity and sum for these triangles.
        
        # Get indices of triangles involving u or v
        affected = np.unique(np.concatenate([tri_incident_arr[u], tri_incident_arr[v]]))
        
        # For each affected triangle, check if it's valid in the NEW adj
        # And compute its sum using the NEW deg
        for t_idx in affected:
            i, j, k = tri_arr[t_idx]
            if adj[i, j] and adj[j, k] and adj[k, i]:
                sums[t_idx] = deg[i] + deg[j] + deg[k]
            else:
                sums[t_idx] = 0
                
        new_score = int(np.max(sums))
        return new_score

    # Local Search
    # We focus on swapping edges to reduce the max triangle sum.
    # Strategy: 
    # 1. Identify the worst triangle(s).
    # 2. Try to break them or reduce their degree sums by swapping edges.
    
    # Precompute all possible edges and non-edges for quick access
    # But we can just iterate.
    
    # To speed up, we can maintain lists of edges and non-edges, but rebuilding is O(n^2)=256, which is fine.
    
    def get_edges_and_non_edges(adj):
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        return edges, non_edges

    improved = True
    while improved and time.time() - start_time < deadline - 10:
        improved = False
        # Find worst triangle
        max_score = int(np.max(current_sums))
        # Get all triangles with max score
        worst_tris = np.where(current_sums == max_score)[0]
        
        # Try to improve by swapping an edge incident to a worst triangle vertex
        # or an edge that forms a worst triangle.
        
        # Heuristic: Try swapping an edge (u,v) that is part of a worst triangle
        # with a non-edge (x,y).
        
        edges, non_edges = get_edges_and_non_edges(current_adj)
        
        # Prioritize swaps involving vertices of worst triangles
        # Let's just do a random restart-like local search but guided.
        
        # 1. Try all single edge swaps that involve a vertex from the worst triangle
        # To avoid O(N^2) scans every time, we can just do a few random restarts or
        # systematic scan of a subset.
        
        # Let's do a systematic scan of all edges and non-edges, but stop early if we find improvement.
        # Given N=16, total pairs = 120. Swaps = 120*119/2 ~ 7000.
        # Update cost is small. 7000 * 100ns ~ 0.7ms. So we can do full local search steps.
        
        for u, v in edges:
            if time.time() - start_time > deadline - 10:
                break
            
            # Only consider swapping if (u,v) is in a worst triangle or incident to one?
            # Actually, any swap might help. But let's prioritize.
            # Check if u or v is in any worst triangle
            is_worst_involved = False
            for t_idx in worst_tris:
                i, j, k = tri_arr[t_idx]
                if u in (i, j, k) or v in (i, j, k):
                    is_worst_involved = True
                    break
            
            if not is_worst_involved:
                continue
                
            for x, y in non_edges:
                if time.time() - start_time > deadline - 10:
                    break
                if x == u or x == v or y == u or y == v:
                    continue
                
                # Apply swap
                current_adj[u, v] = 0
                current_adj[v, u] = 0
                current_adj[x, y] = 1
                current_adj[y, x] = 1
                
                new_score = update_state(current_adj, current_sums, u, v, False)
                # Note: update_state updates sums for u,v. 
                # But x,y are also changed! 
                # My update_state only handles the removed edge's endpoints?
                # No, I passed u,v. I need to handle x,y too.
                # Let's refine update_state to take both edges.
                
                # For now, let's just recompute if needed or fix update_state.
                # Fix: update_state should update all affected triangles.
                # The affected triangles are those containing u, v, x, or y.
                
                # Let's rewrite the logic to be robust.
                
                # Revert if not better
                if new_score >= current_score:
                    current_adj[u, v] = 1
                    current_adj[v, u] = 1
                    current_adj[x, y] = 0
                    current_adj[y, x] = 0
                    # sums are already reverted by the logic inside update_state if we make it symmetric?
                    # No, update_state modified sums. We need to revert sums too if we revert adj.
                    # This is getting messy. 
                    
                    # Better approach: 
                    # 1. Save old sums for affected triangles.
                    # 2. Apply change.
                    # 3. Update sums.
                    # 4. Check score.
                    # 5. If bad, restore old sums and adj.
                    
                    # Let's implement a helper `try_swap`
                    
                    pass
                else:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj.copy()
                        best_sums = current_sums.copy()
                        # Update worst_tris for next iteration
                        worst_tris = np.where(current_sums == current_score)[0]
                    break # Restart scan with new state? Or continue?
                    # Breaking inner loop to restart outer loop ensures we always work from the new best state
                    # which is safer for local search convergence.
                    break
                
            if improved:
                break

    # Final polish: Random restarts / Simulated Annealing if time remains
    if time.time() - start_time < deadline - 5:
        T = 2.0
        cooling = 0.999
        for _ in range(10000):
            if time.time() - start_time > deadline - 5:
                break
            # Random swap
            edges, non_edges = get_edges_and_non_edges(current_adj)
            if not edges or not non_edges:
                break
            u, v = edges[np.random.randint(len(edges))]
            x, y = non_edges[np.random.randint(len(non_edges))]
            if x in (u, v) or y in (u, v):
                continue
            
            # Save state
            old_sums = current_sums.copy()
            
            current_adj[u, v] = 0
            current_adj[v, u] = 0
            current_adj[x, y] = 1
            current_adj[y, x] = 1
            
            # Update sums for u,v,x,y
            affected = np.unique(np.concatenate([tri_incident_arr[u], tri_incident_arr[v], tri_incident_arr[x], tri_incident_arr[y]]))
            deg = current_adj.sum(axis=1)
            for t_idx in affected:
                i, j, k = tri_arr[t_idx]
                if current_adj[i, j] and current_adj[j, k] and current_adj[k, i]:
                    current_sums[t_idx] = deg[i] + deg[j] + deg[k]
                else:
                    current_sums[t_idx] = 0
            
            new_score = int(np.max(current_sums))
            
            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj.copy()
                    best_sums = current_sums.copy()
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    # Revert
                    current_adj[u, v] = 1
                    current_adj[v, u] = 1
                    current_adj[x, y] = 0
                    current_adj[y, x] = 0
                    current_sums = old_sums
            
            T *= cooling
            if T < 0.5:
                T = 0.5

    # Return result
    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] == 1:
                edges_list.append([i, j])
    
    return {'n': n, 'edges': edges_list}
