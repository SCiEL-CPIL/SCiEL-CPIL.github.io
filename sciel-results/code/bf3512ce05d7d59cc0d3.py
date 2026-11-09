import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Set a random seed for reproducibility in this run, but allow variation
    # random.seed(42) # Un-comment for debugging, comment for exploration
    
    for restart in range(2000): # Limit restarts by time
        if time.time() > deadline:
            break
        
        # 1. Greedy Construction
        adj = np.zeros((n, n), dtype=np.uint8)
        current_edges = []
        
        # Random order for diversity
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            if not _creates_c6(adj, u, v):
                adj[u, v] = 1
                adj[v, u] = 1
                current_edges.append((u, v))
        
        # 2. Local Search / Improvement
        # We want to maximize edges.
        # First, try to add any missing edges that were skipped or became possible.
        # Then, use swaps to escape local optima.
        
        improved, adj, current_edges = _local_search(adj, current_edges, deadline)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
            # If we hit the theoretical best or close to it, we might stop early?
            # 105 is best known.
            if best_count >= 105:
                # Don't break immediately, maybe we can find 106?
                # But 105 is the target. Let's keep searching a bit longer just in case.
                pass

    return [list(e) for e in best_edges]

def _local_search(adj, current_edges, deadline):
    n = 40
    edge_set = set(current_edges)
    
    # Phase 1: Greedy Add
    # Try to add all missing edges that can be added
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    missing_edges = [e for e in all_edges if e not in edge_set]
    
    # We can just iterate through missing edges and add if possible
    # This is O(E_missing * C6_check)
    # To be efficient, we can just try to add them one by one.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u, v in missing_edges:
            if (u, v) in edge_set:
                continue
            if not _creates_c6(adj, u, v):
                adj[u, v] = 1
                adj[v, u] = 1
                current_edges.append((u, v))
                edge_set.add((u, v))
                changed = True
        # If we added edges, the list of missing edges is stale, but we just skip existing ones.
        # We should probably re-shuffle missing edges if we want randomness, but for "add all possible", order doesn't matter for the final maximal set?
        # Actually, it does. Adding edge A might prevent adding edge B.
        # So the order of addition matters.
        # The initial greedy pass already did this. This phase just cleans up.
        # To improve, we should randomize the order of missing edges in this phase too.
        if changed:
            random.shuffle(missing_edges)

    # Phase 2: Swaps to escape local optima
    # If no more edges can be added, we are at a local maximum.
    # We try to swap an edge (remove one, add another) to potentially reach a different local maximum with more edges.
    
    swap_iterations = 500
    for _ in range(swap_iterations):
        if time.time() > deadline:
            break
        
        # Try a few random swaps
        # Remove a random edge
        if not current_edges:
            break
            
        idx = random.randint(0, len(current_edges) - 1)
        e_remove = current_edges[idx]
        u, v = e_remove
        
        # Remove it
        adj[u, v] = 0
        adj[v, u] = 0
        current_edges.pop(idx)
        edge_set.discard(e_remove)
        
        # Try to add a random missing edge
        # We need a list of missing edges. Reconstructing it is expensive.
        # We can keep a list of missing edges and update it.
        # Or just pick random u, v and check if they are missing.
        
        added = False
        # Try to add a random edge
        # Pick random u, v
        for _ in range(10): # Try a few candidates
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            
            e_add = (x, y)
            if e_add in edge_set:
                continue
            
            if not _creates_c6(adj, x, y):
                adj[x, y] = 1
                adj[y, x] = 1
                current_edges.append(e_add)
                edge_set.add(e_add)
                added = True
                break
        
        if not added:
            # Revert the removal
            adj[u, v] = 1
            adj[v, u] = 1
            current_edges.append(e_remove)
            edge_set.add(e_remove)
            
            # If we can't add anything, we are stuck. 
            # We can try a different edge to remove.
            # Or we can terminate this swap phase if we fail many times.
            # For simplicity, we just continue.
            
        # After a successful swap (or even failed), we might be able to add more edges.
        # Let's do a quick "add all possible" pass again.
        # This is expensive, so we do it sparingly.
        if added and random.random() < 0.1:
             # Quick add pass
             all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
             missing_edges = [e for e in all_edges if e not in edge_set]
             random.shuffle(missing_edges)
             for u, v in missing_edges:
                 if (u, v) in edge_set:
                     continue
                 if not _creates_c6(adj, u, v):
                     adj[u, v] = 1
                     adj[v, u] = 1
                     current_edges.append((u, v))
                     edge_set.add((u, v))

    return True, adj, current_edges

def _creates_c6(adj, u, v):
    """
    Check if adding edge (u, v) creates a C6.
    This is true if there is a simple path of length 5 from u to v in the current graph.
    Path: u - n1 - n2 - n3 - n4 - v
    """
    n = 40
    
    # Get neighbors of u
    # We can use a list comprehension or iterate
    # adj is numpy array, adj[u] is a row
    
    # Optimization: Use bitmasks? 
    # For n=40, Python integers can act as bitmasks.
    # But explicit iteration is often faster in Python for small degrees.
    
    # Let's stick to iterative DFS/BFS with visited set.
    # Since depth is fixed at 5, we can use 5 nested loops.
    
    # Neighbors of u
    # We need to iterate over neighbors efficiently.
    # adj[u, :] == 1 gives a boolean array.
    # np.where(adj[u, :] == 1)[0] gives indices.
    
    # However, calling np.where 5 times per check might be slow if called millions of times.
    # Let's precompute neighbor lists? 
    # The graph changes, so neighbor lists change.
    # Maintaining neighbor lists dynamically is complex.
    
    # Alternative: Use the numpy array directly in the loops.
    # adj[u, n1] == 1
    
    # To speed up, we can first check if deg(u) * deg(v) is small? No.
    
    # Let's try the 5-loop approach with numpy indexing.
    
    # n1 in N(u) \ {v}
    # n2 in N(n1) \ {u, v, n1}
    # n3 in N(n2) \ {u, v, n1, n2}
    # n4 in N(n3) \ {u, v, n1, n2, n3}
    # v in N(n4) \ {u, v, n1, n2, n3, n4}
    
    # This is O(D^5) where D is max degree.
    
    # We can optimize by pre-fetching neighbor lists for u and v?
    # No, intermediate nodes change.
    
    # Let's use a helper to get neighbors as a list.
    # To avoid creating lists in the inner loop, we can just check adj values.
    
    for n1 in range(n):
        if n1 == v or adj[u, n1] == 0:
            continue
        for n2 in range(n):
            if n2 == u or n2 == v or n2 == n1 or adj[n1, n2] == 0:
                continue
            for n3 in range(n):
                if n3 == u or n3 == v or n3 == n1 or n3 == n2 or adj[n2, n3] == 0:
                    continue
                for n4 in range(n):
                    if n4 == u or n4 == v or n4 == n1 or n4 == n2 or n4 == n3 or adj[n3, n4] == 0:
                        continue
                    if adj[n4, v] == 1:
                        return True
    return False

# Optimization: The nested loops over range(40) are slow because they check all 40 nodes.
# If we can get a list of neighbors, we only iterate over actual neighbors.
# Let's modify _creates_c6 to use neighbor lists.
# But maintaining neighbor lists is overhead.
# Maybe we can use a hybrid: 
# Get neighbors of u once.
# Get neighbors of n1 once.
# etc.

def _creates_c6_fast(adj, u, v):
    n = 40
    # Get neighbors of u
    # adj[u] is a numpy array of uint8
    # We can use .nonzero() or .indices()
    # But creating lists is expensive.
    
    # Let's try to use the fact that degrees are small.
    # We'll iterate over all nodes but skip non-neighbors quickly.
    # The previous implementation does this.
    
    # Is there a faster way?
    # Bitset representation?
    # If we represent the graph as a list of integers (bitmasks), 
    # adj_mask[i] is an int where bit j is 1 if (i,j) is an edge.
    
    # Let's rewrite the graph representation to use bitmasks.
    # This might be significantly faster for checking neighbors.
    
    # However, the current code uses numpy. 
    # Let's stick to numpy but optimize the loops.
    
    # Actually, the 5 nested loops with range(40) are 40^5 = 100M operations worst case?
    # No, the `if` statements prune heavily.
    # If degree is 5, it's 5 * 4 * 3 * 2 * 1 = 120 iterations.
    # The overhead of Python loops is the issue.
    
    # Let's try to implement _creates_c6 using bitmasks.
    # This requires changing the graph representation in construct and _local_search.
    
    return False # Placeholder, we will replace this logic

# Let's implement the bitmask version.
# It's cleaner and likely faster for this specific problem.

def _construct_with_bitmasks():
    start_time = time.time()
    deadline = start_time + 115
    n = 40
    best_mask = [0] * n
    best_count = 0
    best_edges = []
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        masks = [0] * n
        current_edges = []
        
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            if not _creates_c6_masks(masks, u, v):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                current_edges.append((u, v))
                
        # Local search
        current_edges = _local_search_masks(masks, current_edges, deadline)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_mask = masks[:]
            best_edges = current_edges[:]

    return [list(e) for e in best_edges]

def _creates_c6_masks(masks, u, v):
    n = 40
    # Path u - n1 - n2 - n3 - n4 - v
    
    # Neighbors of u (excluding v)
    # masks[u] is a bitmask
    # We can iterate over set bits.
    
    # Helper to iterate over set bits
    # But Python doesn't have built-in bit iteration.
    # We can use:
    # while x:
    #   lsb = x & -x
    #   idx = lsb.bit_length() - 1
    #   x ^= lsb
    
    # Let's do this.
    
    n1s = masks[u] & ~(1 << v)
    
    while n1s:
        lsb = n1s & -n1s
        n1 = lsb.bit_length() - 1
        n1s ^= lsb
        
        # Neighbors of n1 (excluding u, v, n1)
        n2s = masks[n1] & ~(1 << u) & ~(1 << v) & ~(1 << n1)
        
        while n2s:
            lsb = n2s & -n2s
            n2 = lsb.bit_length() - 1
            n2s ^= lsb
            
            # Neighbors of n2 (excluding u, v, n1, n2)
            n3s = masks[n2] & ~(1 << u) & ~(1 << v) & ~(1 << n1) & ~(1 << n2)
            
            while n3s:
                lsb = n3s & -n3s
                n3 = lsb.bit_length() - 1
                n3s ^= lsb
                
                # Neighbors of n3 (excluding u, v, n1, n2, n3)
                n4s = masks[n3] & ~(1 << u) & ~(1 << v) & ~(1 << n1) & ~(1 << n2) & ~(1 << n3)
                
                while n4s:
                    lsb = n4s & -n4s
                    n4 = lsb.bit_length() - 1
                    n4s ^= lsb
                    
                    # Check if v is neighbor of n4
                    if masks[n4] & (1 << v):
                        return True
                        
    return False

def _local_search_masks(masks, current_edges, deadline):
    n = 40
    edge_set = set(current_edges)
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Phase 1: Add missing
    changed = True
    while changed and time.time() < deadline:
        changed = False
        missing = [e for e in all_edges if e not in edge_set]
        random.shuffle(missing)
        for u, v in missing:
            if (u, v) in edge_set:
                continue
            if not _creates_c6_masks(masks, u, v):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                current_edges.append((u, v))
                edge_set.add((u, v))
                changed = True
        if changed:
            continue # Restart loop with new missing list
            
    # Phase 2: Swaps
    for _ in range(1000):
        if time.time() > deadline:
            break
        if not current_edges:
            break
            
        idx = random.randint(0, len(current_edges) - 1)
        e_rem = current_edges[idx]
        u, v = e_rem
        
        # Remove
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
        current_edges.pop(idx)
        edge_set.discard(e_rem)
        
        added = False
        # Try to add random edge
        for _ in range(20):
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            e_add = (x, y)
            if e_add in edge_set:
                continue
                
            if not _creates_c6_masks(masks, x, y):
                masks[x] |= (1 << y)
                masks[y] |= (1 << x)
                current_edges.append(e_add)
                edge_set.add(e_add)
                added = True
                break
                
        if not added:
            # Revert
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            current_edges.append(e_rem)
            edge_set.add(e_rem)
        else:
            # Try to add more
            if random.random() < 0.5:
                missing = [e for e in all_edges if e not in edge_set]
                random.shuffle(missing)
                for u, v in missing:
                    if (u, v) in edge_set:
                        continue
                    if not _creates_c6_masks(masks, u, v):
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        current_edges.append((u, v))
                        edge_set.add((u, v))

    return current_edges

def construct():
    return _construct_with_bitmasks()
