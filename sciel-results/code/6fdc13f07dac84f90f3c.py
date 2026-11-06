import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    
    # Precompute all 8-vertex subsets
    subsets_8 = []
    subset_masks = []
    for s in itertools.combinations(range(n), 8):
        mask = 0
        for v in s:
            mask |= (1 << v)
        subsets_8.append(s)
        subset_masks.append(mask)
    
    # Precompute all partitions of 8 positions into two sets of 4
    partitions = []
    for x_combo in itertools.combinations(range(8), 4):
        x_indices = tuple(x_combo)
        y_indices = tuple(i for i in range(8) if i not in x_indices)
        partitions.append((x_indices, y_indices))
    
    num_subsets = len(subsets_8)
    num_partitions = len(partitions)
    
    # Precompute masks and vertex lists for each (subset, partition) pair
    precomputed = [[None for _ in range(num_partitions)] for _ in range(num_subsets)]
    
    for k, s in enumerate(subsets_8):
        for j, (x_indices, y_indices) in enumerate(partitions):
            x_mask = 0
            y_mask = 0
            x_verts = []
            y_verts = []
            for i in x_indices:
                v = s[i]
                x_mask |= (1 << v)
                x_verts.append(v)
            for i in y_indices:
                v = s[i]
                y_mask |= (1 << v)
                y_verts.append(v)
            precomputed[k][j] = (x_mask, y_mask, tuple(x_verts), tuple(y_verts))

    def find_q3_edges(adj_masks):
        """Returns a list of edges forming a Q3 subgraph, or None if not found."""
        for k in range(num_subsets):
            mask = subset_masks[k]
            s = subsets_8[k]
            
            # Quick edge count check
            edge_count = 0
            for v in s:
                edge_count += (adj_masks[v] & mask).bit_count()
            if edge_count < 24: # 12 edges * 2
                continue
            
            for j in range(num_partitions):
                x_mask, y_mask, x_verts, y_verts = precomputed[k][j]
                
                # Check if min degree >= 3
                valid = True
                for v in x_verts:
                    if (adj_masks[v] & y_mask).bit_count() < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                for v in y_verts:
                    if (adj_masks[v] & x_mask).bit_count() < 3:
                        valid = False
                        break
                
                if valid:
                    # We found a Q3. We need to return the edges.
                    # The edges are all pairs (u, v) in x_verts, y_verts 
                    # EXCEPT those missing in the graph.
                    # Wait, we established that if min degree >= 3, it contains Q3.
                    # The Q3 is $K_{4,4} \setminus M$ where M is a perfect matching 
                    # containing the missing edges.
                    # To return a specific set of 12 edges, we can just return 
                    # all edges present between x_verts and y_verts. 
                    # There are at least 12. Any 12 of them that form a Q3?
                    # Actually, we just need to remove ONE edge from this set 
                    # to break the Q3? No, we need to remove an edge such that 
                    # the resulting graph is Q3-free.
                    # If we return the list of ALL edges between x_verts and y_verts,
                    # removing any one of them might not break the Q3 if there are 
                    # multiple Q3s in this 4+4 subgraph.
                    # However, if we remove an edge, we might still have a Q3.
                    # For repair, we just need to remove an edge that is part of 
                    # SOME Q3.
                    # The set of edges between x_verts and y_verts contains at least one Q3.
                    # So any edge in this set is part of at least one Q3?
                    # Not necessarily. But if we pick a random edge from this set 
                    # and remove it, we reduce the edge count.
                    # It is likely to break at least one Q3.
                    # To be safe, we can return the list of edges.
                    edges_in_subgraph = []
                    for u in x_verts:
                        for v in y_verts:
                            if adj_masks[u] & (1 << v):
                                edges_in_subgraph.append((u, v))
                    return edges_in_subgraph
        return None

    best_edges = set()
    best_count = 0
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        # Initialize random graph with ~50% density
        adj_masks = [0] * n
        current_edges = set()
        
        # Randomly add edges
        random.shuffle(all_edges)
        for u, v in all_edges:
            if random.random() < 0.5:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                current_edges.add((u, v))
        
        # Repair phase
        while time.time() < deadline - 5:
            q3_edges = find_q3_edges(adj_masks)
            if q3_edges is None:
                break
            # Remove a random edge from the Q3
            u, v = random.choice(q3_edges)
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            current_edges.remove((u, v))
        
        # Local search phase
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add a random edge
            random.shuffle(all_edges)
            for u, v in all_edges:
                if (u, v) in current_edges:
                    continue
                # Try adding
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
                if find_q3_edges(adj_masks) is None:
                    current_edges.add((u, v))
                    improved = True
                    break
                else:
                    # Revert
                    adj_masks[u] &= ~(1 << v)
                    adj_masks[v] &= ~(1 << u)
            
            if not improved:
                # Try swaps
                # Pick a random edge to remove
                if not current_edges:
                    break
                u_rem, v_rem = random.choice(list(current_edges))
                adj_masks[u_rem] &= ~(1 << v_rem)
                adj_masks[v_rem] &= ~(1 << u_rem)
                current_edges.remove((u_rem, v_rem))
                
                # Try to add a random non-edge
                added = False
                for u, v in all_edges:
                    if (u, v) in current_edges:
                        continue
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    
                    if find_q3_edges(adj_masks) is None:
                        current_edges.add((u, v))
                        added = True
                        improved = True
                        break
                    else:
                        adj_masks[u] &= ~(1 << v)
                        adj_masks[v] &= ~(1 << u)
                
                if not added:
                    # Revert the removal
                    adj_masks[u_rem] |= (1 << v_rem)
                    adj_masks[v_rem] |= (1 << u_rem)
                    current_edges.add((u_rem, v_rem))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
