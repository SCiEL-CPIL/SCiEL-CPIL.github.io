import time
import random
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    target_edges = 65
    
    # Precompute all triangles
    all_tris = list(itertools.combinations(range(n), 3))
    tri_to_idx = {t: i for i, t in enumerate(all_tris)}
    num_tris = len(all_tris)
    
    # Initial state: K_{8,8} + (0,1)
    adj = [0] * n
    deg = [0] * n
    current_edges = set()
    
    for u in range(8):
        for v in range(8, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            current_edges.add((u, v))
            
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)
    deg[0] += 1
    deg[1] += 1
    current_edges.add((0, 1))
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Initialize triangle sums
    tri_sum = [0] * num_tris
    
    def update_triangle_sums():
        for i, (u, v, w) in enumerate(all_tris):
            # Check if triangle exists
            if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                tri_sum[i] = deg[u] + deg[v] + deg[w]
            else:
                tri_sum[i] = 0
                
    update_triangle_sums()
    current_score = max(tri_sum)
    best_score = current_score
    best_adj = list(adj)
    best_deg = list(deg)
    best_edges = set(current_edges)
    
    rng = random.Random(42)
    
    T_start = 5.0
    T_end = 0.01
    i = 0
    n_iter_total = 500000 # Can go higher now
    
    # Cache neighbors for faster checks? 
    # adj is bitmask, so (adj[u] >> v) & 1 is fast.
    
    while time.time() < deadline:
        i += 1
        T = T_start * (T_end / T_start) ** (min(i, n_iter_total) / n_iter_total)
        
        # Move Selection
        # 1. Pick edge to remove
        if rng.random() < 0.5 and current_score > 0:
            # Find a triangle with max sum
            max_val = current_score
            # Find indices with max sum
            candidates = [idx for idx, s in enumerate(tri_sum) if s == max_val]
            if candidates:
                idx = rng.choice(candidates)
                u, v, w = all_tris[idx]
                e_remove = rng.choice([(u, v), (u, w), (v, w)])
                e_remove = (min(e_remove), max(e_remove))
            else:
                e_remove = rng.choice(list(current_edges))
                e_remove = (min(e_remove), max(e_remove))
        else:
            e_remove = rng.choice(list(current_edges))
            e_remove = (min(e_remove), max(e_remove))
            
        # 2. Pick edge to add
        e_add = None
        for _ in range(10):
            u, v = rng.choice(all_edges)
            if (u, v) not in current_edges:
                e_add = (u, v)
                break
        if e_add is None:
            non_edges = [e for e in all_edges if e not in current_edges]
            if not non_edges:
                break
            e_add = rng.choice(non_edges)
            
        if e_add == e_remove:
            continue
            
        u1, v1 = e_remove
        u2, v2 = e_add
        
        prev_score = current_score
        
        # Apply Move
        # Remove (u1, v1)
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        deg[u1] -= 1
        deg[v1] -= 1
        current_edges.remove(e_remove)
        
        # Add (u2, v2)
        adj[u2] |= (1 << v2)
        adj[v2] |= (1 << u2)
        deg[u2] += 1
        deg[v2] += 1
        current_edges.add(e_add)
        
        # Update Triangle Sums Incrementally
        # Affected triangles are those containing (u1, v1) or (u2, v2)
        # And those where deg[u1], deg[v1], deg[u2], deg[v2] appear.
        # Actually, deg change affects ALL triangles containing u1, v1, u2, or v2.
        # It's safer to just recalculate sums for all triangles involving any of the 4 vertices.
        
        # Vertices involved: u1, v1, u2, v2
        affected_verts = {u1, v1, u2, v2}
        
        # We can just recalculate all 560 sums. It's 560 * 3 additions = 1680 ops.
        # 500k iterations * 1680 = 840M ops. Might be tight in 110s.
        # Let's try to optimize the recalculation.
        
        # Alternative: Only update sums for triangles that are currently valid or might become valid.
        # But validity changes for triangles containing the swapped edges.
        
        # Let's stick to full recalculation but optimize the inner loop.
        new_max = 0
        for i, (u, v, w) in enumerate(all_tris):
            # Check validity
            # We can skip if none of u,v,w are in affected_verts AND the edge status didn't change?
            # Edge status only changed for (u1,v1) and (u2,v2).
            # So only triangles containing (u1,v1) or (u2,v2) change validity.
            # Triangles not containing these edges keep their validity, but their SUM changes if they contain u1, v1, u2, or v2.
            
            # This is getting complex. Let's just do the full loop but in Python it might be slow.
            # Let's try a hybrid:
            # 1. Identify triangles that MUST be updated (contain u1,v1 or u2,v2).
            # 2. Identify triangles that CONTAIN any of u1,v1,u2,v2 (sum changes).
            # 3. Update only those.
            
            pass # Placeholder for logic below
            
        # Optimized Update Logic:
        # Set of triangle indices to update
        update_indices = set()
        
        # Triangles containing edge (u1, v1)
        for w in range(n):
            if w != u1 and w != v1:
                idx = tri_to_idx.get(tuple(sorted((u1, v1, w))))
                if idx is not None:
                    update_indices.add(idx)
                    
        # Triangles containing edge (u2, v2)
        for w in range(n):
            if w != u2 and w != v2:
                idx = tri_to_idx.get(tuple(sorted((u2, v2, w))))
                if idx is not None:
                    update_indices.add(idx)
                    
        # Triangles containing any of the 4 vertices (for sum update)
        for vert in affected_verts:
            for i, (u, v, w) in enumerate(all_tris):
                if vert in (u, v, w):
                    update_indices.add(i)
                    
        # Update sums for these indices
        for idx in update_indices:
            u, v, w = all_tris[idx]
            if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                s = deg[u] + deg[v] + deg[w]
                tri_sum[idx] = s
                if s > new_max:
                    new_max = s
            else:
                tri_sum[idx] = 0
                
        # What about triangles NOT in update_indices?
        # Their validity didn't change (they don't contain the swapped edges).
        # Their sum didn't change (they don't contain the affected vertices).
        # So their values in tri_sum are still correct.
        
        # However, if the previous max was in a triangle NOT in update_indices, and we didn't check it,
        # new_max might be lower than the actual max.
        # So we must take max(new_max, max(tri_sum[idx] for idx not in update_indices)).
        # That's O(560) anyway.
        
        # So, just do:
        current_score = max(tri_sum)
        
        # Acceptance
        delta = current_score - prev_score
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            if current_score < best_score:
                best_score = current_score
                best_adj = list(adj)
                best_deg = list(deg)
                best_edges = set(current_edges)
        else:
            # Revert
            # Add back (u1, v1)
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            deg[u1] += 1
            deg[v1] += 1
            current_edges.add(e_remove)
            
            # Remove (u2, v2)
            adj[u2] &= ~(1 << v2)
            adj[v2] &= ~(1 << u2)
            deg[u2] -= 1
            deg[v2] -= 1
            current_edges.remove(e_add)
            
            # Revert triangle sums
            # We can just recalculate the update_indices again
            # Or store old values? Recalculating is safer.
            for idx in update_indices:
                u, v, w = all_tris[idx]
                if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                    tri_sum[idx] = deg[u] + deg[v] + deg[w]
                else:
                    tri_sum[idx] = 0
                    
            current_score = prev_score
            
    edges_list = [list(e) for e in best_edges]
    return {'n': n, 'edges': edges_list}
