import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def verify_c4_free(n, edges):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        for i in range(n):
            for j in range(i + 1, n):
                common = np.sum(adj[i] & adj[j])
                if common >= 2:
                    return False
        return True

    def build_polarity_graph(q):
        # Generate points of PG(2, q)
        points = []
        for x in range(q):
            for y in range(q):
                points.append((1, x, y))
        for z in range(q):
            points.append((0, 1, z))
        points.append((0, 0, 1))
        
        n_pts = len(points)
        adj = np.zeros((n_pts, n_pts), dtype=bool)
        
        for i in range(n_pts):
            p1 = points[i]
            for j in range(i + 1, n_pts):
                p2 = points[j]
                # Dot product mod q
                dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
                if dot == 0:
                    adj[i][j] = adj[j][i] = True
                    
        return adj

    def find_subgraph(adj, n_target, min_deg_target, seed=0):
        n_pts = adj.shape[0]
        if n_pts < n_target:
            return None
            
        rng = random.Random(seed)
        
        # Greedy removal to find a dense subgraph of size n_target
        # We want to maximize min degree.
        # Strategy: Iteratively remove the vertex with the lowest degree in the current subgraph.
        
        best_subgraph = None
        best_min_deg = -1
        
        # Try multiple random orderings for tie-breaking or initial perturbation
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
                
            # Create a copy of the adjacency matrix for the subgraph search
            # We maintain a set of active vertices
            active = set(range(n_pts))
            
            # To make it deterministic but varied, we can shuffle the initial "priority"
            # Or just run the greedy removal.
            
            # Optimization: Use a list of active vertices and update degrees
            # Since n_pts is small (133 for q=11), we can afford O(n^2) per step.
            
            active_list = list(active)
            # Shuffle to break symmetries if needed?
            # The greedy removal is deterministic if we always pick the min degree.
            # To get different results, we can randomize the tie-breaking.
            
            current_degrees = np.zeros(n_pts, dtype=np.int32)
            for u in active_list:
                for v in active_list:
                    if u != v and adj[u][v]:
                        current_degrees[u] += 1
                        
            # We need to remove (n_pts - n_target) vertices
            removals_needed = n_pts - n_target
            
            for _ in range(removals_needed):
                if time.time() > deadline - 2:
                    break
                
                # Find min degree among active
                min_deg = n_pts
                candidates = []
                for u in active_list:
                    if current_degrees[u] < min_deg:
                        min_deg = current_degrees[u]
                        candidates = [u]
                    elif current_degrees[u] == min_deg:
                        candidates.append(u)
                
                # Pick a candidate randomly to allow variation
                v_to_remove = rng.choice(candidates)
                
                # Update degrees
                active_list.remove(v_to_remove)
                for u in active_list:
                    if adj[u][v_to_remove]:
                        current_degrees[u] -= 1
                current_degrees[v_to_remove] = -1 # Mark as removed
                
            # Check result
            if len(active_list) < n_target:
                continue
                
            min_deg_in_subgraph = min(current_degrees[u] for u in active_list)
            
            if min_deg_in_subgraph > best_min_deg:
                best_min_deg = min_deg_in_subgraph
                best_subgraph = active_list.copy()
                
            if best_min_deg >= min_deg_target:
                break
                
        if best_subgraph is not None and best_min_deg >= min_deg_target:
            # Map to 0..n_target-1
            sorted_subgraph = sorted(best_subgraph)
            # Ensure size is exactly n_target (it should be)
            if len(sorted_subgraph) != n_target:
                # If we removed too many or too few, adjust?
                # The loop removes exactly n_pts - n_target vertices.
                # So len should be n_target.
                pass
                
            idx_map = {old: new for new, old in enumerate(sorted_subgraph)}
            edges = []
            for i in range(n_target):
                old_i = sorted_subgraph[i]
                for j in range(i + 1, n_target):
                    old_j = sorted_subgraph[j]
                    if adj[old_i][old_j]:
                        edges.append([i, j])
                        
            if verify_c4_free(n_target, edges):
                return {'n': n_target, 'edges': edges}
                
        return None

    best_result = None
    
    # Try q=11 (N=133) for n=47, 46
    adj_11 = build_polarity_graph(11)
    
    for n in [47, 46]:
        min_deg_req = n - 39
        if time.time() > deadline - 20:
            break
        res = find_subgraph(adj_11, n, min_deg_req, seed=42)
        if res:
            best_result = res
            break
            
    if not best_result:
        # Try q=7 (N=57) for n=46, 45
        adj_7 = build_polarity_graph(7)
        for n in [46, 45]:
            min_deg_req = n - 39
            if time.time() > deadline - 10:
                break
            res = find_subgraph(adj_7, n, min_deg_req, seed=42)
            if res:
                best_result = res
                break
                
    if not best_result:
        # Fallback to original greedy if all else fails? Or just return a small valid graph.
        # The original code had a fallback to n=39.
        # But polarity graphs should work.
        # Let's add a quick greedy fallback for n=46 using q=11 graph but simpler.
        pass

    if not best_result:
        # Absolute fallback
        best_result = {'n': 39, 'edges': []}
        
    return best_result
