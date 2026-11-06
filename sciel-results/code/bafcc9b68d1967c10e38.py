import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # 1. Define Grötzsch graph (11 vertices)
    # Vertices 0..4 form C5.
    # Vertex 5..9 are copies of 0..4.
    # Vertex 10 is the top.
    # Edges:
    # C5: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Copies: (0,5), (1,6), (2,7), (3,8), (4,9) ? No, Mycielski construction:
    # u in C5, v in C5 (u-v edge) -> u-v' and v-u'
    # Let's use the explicit edges from the seed program to be safe, or generate.
    
    # Seed program edges:
    # cycle = [(i, (i + 1) % 5) for i in range(5)]
    # edges = [list(e) for e in cycle]
    # for u, w in cycle:
    #     edges += [[u, 5 + w], [w, 5 + u]]
    # return edges + [[5 + u, 10] for u in range(5)]
    
    base_edges = []
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, w in cycle:
        base_edges.append((u, w))
        base_edges.append((u, 5 + w))
        base_edges.append((w, 5 + u))
    for u in range(5):
        base_edges.append((5 + u, 10))
        
    # Remove duplicates if any (Mycielski construction usually doesn't create dups)
    base_edges = list(set(base_edges))
    
    # Build adjacency matrix for O (0..10)
    n_old = 11
    adj_old = [[False] * n_old for _ in range(n_old)]
    for u, v in base_edges:
        adj_old[u][v] = True
        adj_old[v][u] = True
        
    # Precompute independent sets in O
    # Represent as bitmask of 11 bits
    independent_sets = []
    for mask in range(1 << n_old):
        is_ind = True
        for i in range(n_old):
            if (mask >> i) & 1:
                for j in range(i + 1, n_old):
                    if (mask >> j) & 1:
                        if adj_old[i][j]:
                            is_ind = False
                            break
            if not is_ind:
                break
        if is_ind:
            independent_sets.append(mask)
            
    # Sort independent sets by size (popcount) descending to find max faster?
    # Or just iterate.
    independent_sets.sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    best_total_edges = len(base_edges)
    best_config = None
    
    # Enumerate graphs on N (5 vertices: 11..15 -> 0..4 in local indexing)
    n_new = 5
    # There are 10 possible edges in N
    edge_indices = [(i, j) for i in range(n_new) for j in range(i + 1, n_new)]
    
    for mask_n in range(1 << 10):
        # Check if triangle-free
        adj_n = [[False] * n_new for _ in range(n_new)]
        edge_count_n = 0
        for k, (i, j) in enumerate(edge_indices):
            if (mask_n >> k) & 1:
                adj_n[i][j] = True
                adj_n[j][i] = True
                edge_count_n += 1
        
        # Check triangles in N
        has_tri = False
        for i in range(n_new):
            for j in range(i + 1, n_new):
                if adj_n[i][j]:
                    for k in range(j + 1, n_new):
                        if adj_n[i][k] and adj_n[j][k]:
                            has_tri = True
                            break
                if has_tri:
                    break
            if has_tri:
                break
        
        if has_tri:
            continue
            
        # Solve for max edges to O
        # Backtracking on 5 vertices
        # State: masks assigned to 0..4
        masks = [0] * n_new
        max_edges_to_o = 0
        
        def backtrack(idx):
            nonlocal max_edges_to_o
            if time.time() > deadline:
                return False # Timeout signal
            
            if idx == n_new:
                # Calculate total edges to O
                total = 0
                for m in masks:
                    total += bin(m).count('1')
                if total > max_edges_to_o:
                    max_edges_to_o = total
                return True
            
            # Try all independent sets for vertex idx
            # Optimization: If idx has edges to previous vertices,
            # its mask must be disjoint from their masks.
            required_disjoint = 0
            for prev in range(idx):
                if adj_n[idx][prev]:
                    required_disjoint |= masks[prev]
            
            for m in independent_sets:
                if (m & required_disjoint) != 0:
                    continue
                masks[idx] = m
                if not backtrack(idx + 1):
                    return False
            return True
        
        # Run backtracking
        # To speed up, we can just find the max, not all solutions.
        # But we need the max for this G_N.
        
        # Simple optimization: Greedy or limited backtracking?
        # Given small domain, full backtracking is fine.
        
        # We want to maximize, so we should try larger sets first.
        # independent_sets is already sorted by size desc.
        
        backtrack(0)
        
        total_edges = len(base_edges) + edge_count_n + max_edges_to_o
        if total_edges > best_total_edges:
            best_total_edges = total_edges
            # Store the best configuration
            # We need to reconstruct the edges
            # For now, just store the count. We need to return the edges.
            # We should store the best mask_n and masks.
            if best_config is None:
                best_config = {
                    'mask_n': mask_n,
                    'masks': masks[:],
                    'total': total_edges
                }
            else:
                best_config['mask_n'] = mask_n
                best_config['masks'] = masks[:]
                best_config['total'] = total_edges

    # Reconstruct edges from best_config
    if best_config is None:
        # Fallback to seed
        return [list(e) for e in base_edges]
        
    edges = [list(e) for e in base_edges]
    
    # Add edges in N
    mask_n = best_config['mask_n']
    for k, (i, j) in enumerate(edge_indices):
        if (mask_n >> k) & 1:
            # Vertices in N are 11..15, so i -> 11+i, j -> 11+j
            edges.append([11 + i, 11 + j])
            
    # Add edges to O
    masks = best_config['masks']
    for i in range(n_new):
        m = masks[i]
        for j in range(n_old):
            if (m >> j) & 1:
                edges.append([11 + i, j])
                
    return edges
