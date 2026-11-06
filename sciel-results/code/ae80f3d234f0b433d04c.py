import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n = 16
    
    # Precompute all 8-vertex subsets as bitmasks
    subsets_8 = []
    for s in itertools.combinations(range(n), 8):
        mask = 0
        for v in s:
            mask |= (1 << v)
        subsets_8.append((mask, s))
    
    # Precompute all partitions of 8 positions into two sets of 4
    # Each partition is (x_mask, y_mask) where x_mask, y_mask are 8-bit masks
    partitions = []
    all_8 = (1 << 8) - 1
    for x_combo in itertools.combinations(range(8), 4):
        x_mask = 0
        for i in x_combo:
            x_mask |= (1 << i)
        y_mask = all_8 ^ x_mask
        partitions.append((x_mask, y_mask))
    
    def has_q3(adj_masks):
        # adj_masks[v] is a bitmask of neighbors of v
        for mask, s in subsets_8:
            # Count edges within s using bitmask
            edge_count = 0
            for i in range(8):
                vi = s[i]
                # Count neighbors of vi that are in s
                neighbors_in_s = adj_masks[vi] & mask
                edge_count += bin(neighbors_in_s).count('1')
            edge_count //= 2  # Each edge counted twice
            
            if edge_count < 12:
                continue
            
            # Check all partitions
            for x_mask, y_mask in partitions:
                # Map partition indices to actual vertices
                # x_mask has bits 0-7 set for positions in X
                valid = True
                # Check each vertex in X has degree >= 3 in Y
                for i in range(8):
                    if (x_mask >> i) & 1:
                        vi = s[i]
                        # Count neighbors of vi in Y
                        y_neighbors = 0
                        for j in range(8):
                            if (y_mask >> j) & 1:
                                vj = s[j]
                                if (adj_masks[vi] >> vj) & 1:
                                    y_neighbors += 1
                        if y_neighbors < 3:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Check each vertex in Y has degree >= 3 in X
                for j in range(8):
                    if (y_mask >> j) & 1:
                        vj = s[j]
                        x_neighbors = 0
                        for i in range(8):
                            if (x_mask >> i) & 1:
                                vi = s[i]
                                if (adj_masks[vj] >> vi) & 1:
                                    x_neighbors += 1
                        if x_neighbors < 3:
                            valid = False
                            break
                
                if valid:
                    return True
        return False
    
    # Build initial graph: K_3,13 with A={0,1,2}, B={3..15}
    adj_masks = [0] * n
    current_edges = set()
    
    for a in range(3):
        for b in range(3, 16):
            adj_masks[a] |= (1 << b)
            adj_masks[b] |= (1 << a)
            current_edges.add((a, b))
    
    # Add A-A edges to make K3 on A
    a_edges = [(0,1), (1,2), (0,2)]
    for u, v in a_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        current_edges.add((u, v))
    
    b_vertices = list(range(3, 16))
    b_edges = [(b_vertices[i], b_vertices[j]) for i in range(len(b_vertices)) for j in range(i+1, len(b_vertices))]
    
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Local search with greedy addition and edge swaps
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        # Reset to base state
        adj_masks = [0] * n
        current_edges = set()
        
        for a in range(3):
            for b in range(3, 16):
                adj_masks[a] |= (1 << b)
                adj_masks[b] |= (1 << a)
                current_edges.add((a, b))
        
        for u, v in a_edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            current_edges.add((u, v))
        
        # Greedy phase: try adding B-B edges in random order
        random.shuffle(b_edges)
        for u, v in b_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) in current_edges:
                continue
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            current_edges.add((u, v))
            
            if has_q3(adj_masks):
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                current_edges.remove((u, v))
        
        # Local search phase: try edge swaps
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(b_edges)
            for u, v in b_edges:
                if time.time() > deadline - 5:
                    break
                if (u, v) in current_edges:
                    continue
                
                # Try adding (u,v)
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
                if not has_q3(adj_masks):
                    current_edges.add((u, v))
                    improved = True
                    break
                else:
                    # Revert
                    adj_masks[u] &= ~(1 << v)
                    adj_masks[v] &= ~(1 << u)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
