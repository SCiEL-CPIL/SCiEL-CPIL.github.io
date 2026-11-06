import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin
    
    n = 16
    m_target = 65
    
    # Initialize with K_{8,8} + 1 edge within partition A
    # A: 0-7, B: 8-15
    adj = np.zeros((n, n), dtype=int)
    
    # Bipartite part K_{8,8}
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
            
    # One edge within A
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    deg = np.array([np.sum(adj[i]) for i in range(n)], dtype=int)
    
    # Verify edge count
    edge_count = np.sum(adj) // 2
    if edge_count != m_target:
        # Fallback: K_{6,10} + 5 matching
        adj = np.zeros((n, n), dtype=int)
        for u in range(6):
            for v in range(6, 16):
                adj[u, v] = 1
                adj[v, u] = 1
        for u, v in [(6,7),(8,9),(10,11),(12,13),(14,15)]:
            adj[u, v] = 1
            adj[v, u] = 1
        deg = np.array([np.sum(adj[i]) for i in range(n)], dtype=int)

    # Precompute neighbor masks and triangle list
    def build_masks_and_triangles():
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j]:
                    mask |= (1 << j)
            masks[i] = mask
        
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    common = masks[u] & masks[v]
                    common &= ~((1 << (v+1)) - 1)
                    c = common
                    while c:
                        w = c & (-c)
                        c ^= w
                        wi = w.bit_length() - 1
                        triangles.append((u, v, wi))
        return masks, triangles

    def full_score(masks, triangles):
        """Compute max triangle degree sum."""
        if not triangles:
            return 0
        max_sum = 0
        for u, v, w in triangles:
            s = deg[u] + deg[v] + deg[w]
            if s > max_sum:
                max_sum = s
        return max_sum

    # Initial full computation
    masks, triangles = build_masks_and_triangles()
    current_score = full_score(masks, triangles)
    best_score = current_score
    best_adj = adj.copy()
    
    # Edge/non-edge sets
    edge_set = set()
    non_edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edge_set.add((i, j))
            else:
                non_edge_set.add((i, j))
    
    # Precompute for each vertex, which triangles it's in (by index)
    tri_index = {}
    for idx, (u, v, w) in enumerate(triangles):
        for vtx in (u, v, w):
            if vtx not in tri_index:
                tri_index[vtx] = []
            tri_index[vtx].append(idx)
    
    def incremental_score(affected_vertices):
        """
        Recompute score considering only triangles that include at least one
        vertex from affected_vertices. Returns (new_score, new_triangles, new_masks).
        """
        # Rebuild masks for affected vertices
        for vtx in affected_vertices:
            mask = 0
            for j in range(n):
                if adj[vtx, j]:
                    mask |= (1 << j)
            masks[vtx] = mask
        
        # Find which triangles are affected
        affected_tris = set()
        for vtx in affected_vertices:
            if vtx in tri_index:
                affected_tris.update(tri_index[vtx])
        
        # But some triangles may no longer exist (edges removed), so we need to
        # re-derive triangles. For correctness, rebuild triangles from scratch
        # but only if the graph structure changed significantly.
        # Actually, for a single swap, we can just rebuild all triangles.
        # Let's just do a full rebuild but optimized.
        
        # Rebuild all triangles
        new_triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    common = masks[u] & masks[v]
                    common &= ~((1 << (v+1)) - 1)
                    c = common
                    while c:
                        w = c & (-c)
                        c ^= w
                        wi = w.bit_length() - 1
                        new_triangles.append((u, v, wi))
        
        # Update tri_index
        tri_index.clear()
        for idx, (u, v, w) in enumerate(new_triangles):
            for vtx in (u, v, w):
                if vtx not in tri_index:
                    tri_index[vtx] = []
                tri_index[vtx].append(idx)
        
        # Compute new score
        new_score = 0
        for u, v, w in new_triangles:
            s = deg[u] + deg[v] + deg[w]
            if s > new_score:
                new_score = s
        
        return new_score, new_triangles

    T = 1.0
    cooling = 0.999
    min_T = 0.05
    
    def apply_swap(rem, add):
        """Apply a swap and update deg, edge_set, non_edge_set."""
        u, v = rem
        x, y = add
        adj[u, v] = 0
        adj[v, u] = 0
        adj[x, y] = 1
        adj[y, x] = 1
        deg[u] -= 1
        deg[v] -= 1
        deg[x] += 1
        deg[y] += 1
        edge_set.discard(rem)
        edge_set.discard(add)
        non_edge_set.discard(rem)
        non_edge_set.discard(add)
        edge_set.add(add)
        non_edge_set.add(rem)
    
    def revert_swap(rem, add):
        """Revert a swap."""
        u, v = rem
        x, y = add
        adj[u, v] = 1
        adj[v, u] = 1
        adj[x, y] = 0
        adj[y, x] = 0
        deg[u] += 1
        deg[v] += 1
        deg[x] -= 1
        deg[y] -= 1
        edge_set.discard(rem)
        edge_set.discard(add)
        non_edge_set.discard(rem)
        non_edge_set.discard(add)
        edge_set.add(rem)
        non_edge_set.add(add)
    
    # Main search loop
    while time.time() < deadline - 15:  # Leave 15s for polish
        if not edge_set or not non_edge_set:
            break
            
        removed = random.choice(tuple(edge_set))
        added = random.choice(tuple(non_edge_set))
        
        u, v = removed
        x, y = added
        
        # Apply
        apply_swap(removed, added)
        
        # Incremental score update
        affected = {u, v, x, y}
        new_score, triangles = incremental_score(affected)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
        elif new_score == current_score:
            pass
        else:
            delta = new_score - current_score
            if random.random() < np.exp(-delta / T):
                pass
            else:
                revert_swap(removed, added)
                
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Final polish: targeted moves around high-degree triangles
    polish_deadline = deadline - 2
    polish_T = 0.1
    polish_cooling = 0.9995
    
    # Identify vertices in high-scoring triangles for targeted moves
    high_score_threshold = best_score - 2
    
    while time.time() < polish_deadline:
        if not edge_set or not non_edge_set:
            break
            
        improved = False
        for _ in range(8):
            removed = random.choice(tuple(edge_set))
            added = random.choice(tuple(non_edge_set))
            
            u, v = removed
            x, y = added
            
            apply_swap(removed, added)
            affected = {u, v, x, y}
            new_score, triangles = incremental_score(affected)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj.copy()
            elif new_score <= current_score + 1:
                if random.random() < np.exp(-(new_score - current_score) / polish_T):
                    current_score = new_score
                    improved = True
                else:
                    revert_swap(removed, added)
            else:
                revert_swap(removed, added)
        
        if not improved:
            polish_T *= polish_cooling
            if polish_T < 0.01:
                polish_T = 0.01
    
    # Reconstruct edge list from best_adj
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
