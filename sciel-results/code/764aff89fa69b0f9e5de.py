import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 31
    q = 5
    
    # 1. Build Polarity Graph
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
        
    adj_masks = [0] * n
    edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edge_set.add((i, j))
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
                
    if len(edge_set) != 90:
        # Fallback if construction failed (should not happen)
        # Just add random edges to reach 90?
        # But polarity graph is standard.
        pass

    # 2. Add 1st edge (91st)
    # Evaluate all non-edges
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                non_edges.append((i, j))
                
    # Helper to count C4s
    def count_c4(adj):
        total = 0
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                common = ai & adj[j]
                c = common.bit_count() if hasattr(int, 'bit_count') else bin(common).count('1')
                if c > 1:
                    total += c * (c - 1) // 2
        return total // 2

    best_first_edge = None
    best_first_c4 = float('inf')
    
    for (i, j) in non_edges:
        adj_masks[i] |= (1 << j)
        adj_masks[j] |= (1 << i)
        c4_count = count_c4(adj_masks)
        adj_masks[i] &= ~(1 << j)
        adj_masks[j] &= ~(1 << i)
        
        if c4_count < best_first_c4:
            best_first_c4 = c4_count
            best_first_edge = (i, j)
            
    # Apply best first edge
    if best_first_edge:
        i, j = best_first_edge
        edge_set.add((i, j))
        non_edges.remove((i, j))
        adj_masks[i] |= (1 << j)
        adj_masks[j] |= (1 << i)
        
    current_c4 = best_first_c4
    best_c4 = current_c4
    best_adj_masks = adj_masks[:]
    
    # 3. Local Search: Cyclic Coordinate Descent with Perturbation
    random.seed(42)
    
    # Convert sets to lists for faster iteration? Sets are fine for membership.
    # We need to iterate over edges.
    # non_edges is a list, edges is a set.
    
    # To speed up inner loop, we can precompute something?
    # No, bitmask intersection is fast.
    
    # We will iterate through edges.
    # If no improvement in a full pass, do a random perturbation.
    
    passes = 0
    max_passes = 1000 # Safety
    
    while time.time() < deadline and passes < max_passes:
        passes += 1
        improved_in_pass = False
        
        # Get list of current edges to iterate
        # Note: modifying edge_set while iterating is bad.
        # Create a list of edges to try.
        edges_to_try = list(edge_set)
        
        for (u, v) in edges_to_try:
            if time.time() > deadline - 5:
                break
            
            # Check if (u,v) is still in edge_set (might have been swapped out in previous iteration of this pass? 
            # No, we only swap once per (u,v) if improved. 
            # But if we swapped out (u,v) earlier in the pass, it's no longer in edge_set.
            # However, edges_to_try is a snapshot.
            # So we must check if (u,v) is still an edge.
            if (u, v) not in edge_set:
                continue
            
            # Remove (u,v)
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            
            best_swap = None
            best_swap_c4 = current_c4
            
            # Try all non-edges
            # To speed up, we can skip non-edges that share vertices with u,v?
            # Adding (x,y) where x=u or y=v or x=v or y=u creates a triangle or multi-edge?
            # No, simple graph.
            # If we add (u, x), it's valid.
            # But adding (u, x) when we just removed (u, v) is a valid swap.
            
            for (x, y) in non_edges:
                # Add (x,y)
                adj_masks[x] |= (1 << y)
                adj_masks[y] |= (1 << x)
                
                new_c4 = count_c4(adj_masks)
                
                # Remove (x,y)
                adj_masks[x] &= ~(1 << y)
                adj_masks[y] &= ~(1 << x)
                
                if new_c4 < best_swap_c4:
                    best_swap_c4 = new_c4
                    best_swap = (x, y)
                    
            # Restore (u,v)
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
            if best_swap is not None:
                x, y = best_swap
                # Apply swap
                edge_set.remove((u, v))
                non_edges.remove((x, y))
                edge_set.add((x, y))
                non_edges.append((u, v))
                
                # Update masks
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                adj_masks[x] |= (1 << y)
                adj_masks[y] |= (1 << x)
                
                current_c4 = best_swap_c4
                improved_in_pass = True
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj_masks = adj_masks[:]
        
        if not improved_in_pass:
            # Local minimum. Perturb.
            # Pick a random edge and random non-edge and swap them, accepting worse C4.
            if edge_set and non_edges:
                e1 = random.choice(list(edge_set))
                e2 = random.choice(non_edges)
                
                # Swap
                u, v = e1
                x, y = e2
                
                # Remove e1
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                edge_set.remove(e1)
                non_edges.append(e1)
                
                # Add e2
                adj_masks[x] |= (1 << y)
                adj_masks[y] |= (1 << x)
                edge_set.add(e2)
                non_edges.remove(e2)
                
                current_c4 = count_c4(adj_masks)
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj_masks = adj_masks[:]

    # Restore best state
    adj_masks = best_adj_masks
    
    # Convert back to edge list
    final_edges = []
    for i in range(n):
        mask = adj_masks[i]
        for j in range(i+1, n):
            if mask & (1 << j):
                final_edges.append([i, j])
                
    # Verify
    if len(final_edges) != 91:
        # Fallback
        return [[i, j] for i in range(30) for j in [i+1]][:91] # Dummy
        
    return final_edges
