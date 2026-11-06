import time
import random
import math

DEADLINE = 115

def _generate_triangles(n):
    """Generate all possible triangles in a complete graph of n vertices."""
    triangles = []
    tri_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
                tri_index[(u, v, w)] = idx
                idx += 1
    return triangles, tri_index

def _state_to_edges(adj_masks):
    n = 16
    edges = []
    for i in range(n):
        mask = adj_masks[i]
        for j in range(i + 1, n):
            if (mask >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    # Precompute triangles
    triangles, tri_index = _generate_triangles(n)
    num_tris = len(triangles)
    
    # Precompute for each edge (u,v) which triangles it belongs to
    # And for each vertex u, which triangles it belongs to
    edge_to_tris = {}
    vert_to_tris = {}
    for i, (u, v, w) in enumerate(triangles):
        e1 = (u, v) if u < v else (v, u)
        e2 = (u, w) if u < w else (w, u)
        e3 = (v, w) if v < w else (w, v)
        
        if e1 not in edge_to_tris: edge_to_tris[e1] = []
        edge_to_tris[e1].append(i)
        
        if e2 not in edge_to_tris: edge_to_tris[e2] = []
        edge_to_tris[e2].append(i)
        
        if e3 not in edge_to_tris: edge_to_tris[e3] = []
        edge_to_tris[e3].append(i)
        
        for vert in (u, v, w):
            if vert not in vert_to_tris: vert_to_tris[vert] = []
            vert_to_tris[vert].append(i)

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    non_edges = [e for e in all_pairs]

    def _init_graph():
        adj_masks = [0] * n
        degs = [0] * n
        edges_list = []
        edges_set = set()
        
        for u in range(8):
            for v in range(8, 16):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
                e = (u, v)
                edges_list.append(e)
                edges_set.add(e)
                
        # Add one random edge within a partition to create triangles
        # Let's add (0, 1)
        u, v = 0, 1
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        degs[u] += 1
        degs[v] += 1
        e = (u, v)
        edges_list.append(e)
        edges_set.add(e)
        
        # Calculate initial scores
        tri_scores = [0] * num_tris
        for i, (u, v, w) in enumerate(triangles):
            # Check if triangle exists
            if (adj_masks[u] >> v) & 1 and (adj_masks[u] >> w) & 1 and (adj_masks[v] >> w) & 1:
                tri_scores[i] = degs[u] + degs[v] + degs[w]
                
        current_score = max(tri_scores)
        return adj_masks, degs, edges_list, edges_set, tri_scores, current_score

    def _local_search(adj_masks, degs, edges_list, edges_set, tri_scores, current_score, time_limit):
        best_score = current_score
        best_adj_masks = adj_masks[:]
        best_degs = degs[:]
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9999
        
        while time.time() - start_time < time_limit:
            if not edges_list:
                break
                
            # Select edge to remove
            # Bias towards edges in high-scoring triangles
            rem_e = random.choice(edges_list)
            rem_u, rem_v = rem_e
            
            # Select edge to add
            # Try to find a non-edge
            add_e = None
            for _ in range(10): # Few retries
                cand = random.choice(all_pairs)
                if cand not in edges_set:
                    add_e = cand
                    break
            if not add_e:
                continue
                
            add_u, add_v = add_e
            
            # Apply swap
            # 1. Remove rem_e
            adj_masks[rem_u] &= ~(1 << rem_v)
            adj_masks[rem_v] &= ~(1 << rem_u)
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            edges_list.remove(rem_e)
            edges_set.remove(rem_e)
            
            # 2. Add add_e
            adj_masks[add_u] |= (1 << add_v)
            adj_masks[add_v] |= (1 << add_u)
            degs[add_u] += 1
            degs[add_v] += 1
            edges_list.append(add_e)
            edges_set.add(add_e)
            
            # Update tri_scores
            # We need to update scores for triangles affected by deg changes of rem_u, rem_v, add_u, add_v
            # And existence changes for triangles containing rem_e or add_e
            
            # Affected vertices: rem_u, rem_v, add_u, add_v
            # Affected triangles: 
            # 1. Those containing rem_e (u,v) -> no longer triangles (score 0)
            # 2. Those containing add_e (u,v) -> become triangles (score deg sum)
            # 3. Those containing rem_u or rem_v (but not both) -> deg sum changed
            # 4. Those containing add_u or add_v (but not both) -> deg sum changed
            
            # To avoid complex bookkeeping of overlapping updates, 
            # and since n=16 is small, we can just recalculate the scores for all triangles 
            # that involve any of the 4 changed vertices.
            
            changed_verts = {rem_u, rem_v, add_u, add_v}
            
            # Identify triangles to update
            tris_to_update = set()
            for v in changed_verts:
                if v in vert_to_tris:
                    tris_to_update.update(vert_to_tris[v])
                    
            # Update scores for these triangles
            for t_idx in tris_to_update:
                u, v, w = triangles[t_idx]
                # Check if triangle exists in new graph
                if (adj_masks[u] >> v) & 1 and (adj_masks[u] >> w) & 1 and (adj_masks[v] >> w) & 1:
                    tri_scores[t_idx] = degs[u] + degs[v] + degs[w]
                else:
                    tri_scores[t_idx] = 0
                    
            new_score = max(tri_scores)
            
            # Acceptance
            if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj_masks = adj_masks[:]
                    best_degs = degs[:]
                    # Optional: restart cooling if improvement
                    # T = 1.0 
            else:
                # Revert
                # Remove add_e
                adj_masks[add_u] &= ~(1 << add_v)
                adj_masks[add_v] &= ~(1 << add_u)
                degs[add_u] -= 1
                degs[add_v] -= 1
                edges_list.remove(add_e)
                edges_set.remove(add_e)
                
                # Add rem_e
                adj_masks[rem_u] |= (1 << rem_v)
                adj_masks[rem_v] |= (1 << rem_u)
                degs[rem_u] += 1
                degs[rem_v] += 1
                edges_list.append(rem_e)
                edges_set.add(rem_e)
                
                # Revert tri_scores
                for t_idx in tris_to_update:
                    u, v, w = triangles[t_idx]
                    if (adj_masks[u] >> v) & 1 and (adj_masks[u] >> w) & 1 and (adj_masks[v] >> w) & 1:
                        tri_scores[t_idx] = degs[u] + degs[v] + degs[w]
                    else:
                        tri_scores[t_idx] = 0

            T *= cooling
            if T < T_min:
                T = 1.0 # Restart annealing

        return best_score, best_adj_masks, best_degs

    # Run multiple restarts
    best_global_score = float('inf')
    best_global_adj = None
    best_global_degs = None
    
    # Use 80% of time for the first restart, then divide remaining time among others
    time_per_restart = (DEADLINE - (time.time() - start_time)) / 5.0
    time_per_restart = max(time_per_restart, 1.0)
    
    for restart in range(5):
        adj_masks, degs, edges_list, edges_set, tri_scores, current_score = _init_graph()
        
        # Add some randomization to the initial state
        if restart > 0:
            for _ in range(5):
                rem_e = random.choice(edges_list)
                rem_u, rem_v = rem_e
                adj_masks[rem_u] &= ~(1 << rem_v)
                adj_masks[rem_v] &= ~(1 << rem_u)
                degs[rem_u] -= 1
                degs[rem_v] -= 1
                edges_list.remove(rem_e)
                edges_set.remove(rem_e)
                
                add_e = random.choice(all_pairs)
                while add_e in edges_set:
                    add_e = random.choice(all_pairs)
                add_u, add_v = add_e
                adj_masks[add_u] |= (1 << add_v)
                adj_masks[add_v] |= (1 << add_u)
                degs[add_u] += 1
                degs[add_v] += 1
                edges_list.append(add_e)
                edges_set.add(add_e)
                
                # Recalculate scores
                for i, (u, v, w) in enumerate(triangles):
                    if (adj_masks[u] >> v) & 1 and (adj_masks[u] >> w) & 1 and (adj_masks[v] >> w) & 1:
                        tri_scores[i] = degs[u] + degs[v] + degs[w]
                    else:
                        tri_scores[i] = 0
                current_score = max(tri_scores)
        
        time_limit = min(DEADLINE - 1.0, time.time() - start_time + time_per_restart)
        best_score, best_adj, best_degs = _local_search(adj_masks, degs, edges_list, edges_set, tri_scores, current_score, time_limit)
        
        if best_score < best_global_score:
            best_global_score = best_score
            best_global_adj = best_adj
            best_global_degs = best_degs
        
        if time.time() - start_time > DEADLINE - 2.0:
            break

    # If no valid result found, return the initial one
    if best_global_adj is None:
        adj_masks, degs, edges_list, edges_set, tri_scores, current_score = _init_graph()
        best_global_adj = adj_masks
        best_global_degs = degs

    best_edges = _state_to_edges(best_global_adj)
    return {'n': n, 'edges': best_edges}
