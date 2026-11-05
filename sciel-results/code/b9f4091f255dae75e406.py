import time
import random

def construct():
    N = 20
    deadline = time.time() + 140  # 10s margin for 150s limit
    
    best_edges = 0
    best_graph = None
    
    # Helper to check validity of adding edge (u, v)
    def can_add(u, v, masks):
        # Check pair (u, v) itself
        # Common neighbors are in masks[u] & masks[v]
        # Adding u-v doesn't change common neighbors of u,v
        if (masks[u] & masks[v]).bit_count() > 2:
            return False
        
        # Check pairs (u, x) and (v, x) for x != u, v
        for x in range(N):
            if x == u or x == v:
                continue
            
            # For pair (u, x)
            # New common neighbors: (masks[u] | (1<<v)) & masks[x]
            # Count = bit_count(masks[u] & masks[x]) + (1 if v in masks[x] else 0)
            cu_x = (masks[u] & masks[x]).bit_count()
            if masks[x] & (1 << v):
                if cu_x > 1:
                    return False
            else:
                if cu_x > 2:
                    return False
            
            # For pair (v, x)
            cv_x = (masks[v] & masks[x]).bit_count()
            if masks[x] & (1 << u):
                if cv_x > 1:
                    return False
            else:
                if cv_x > 2:
                    return False
                    
        return True

    def greedy_fill(masks, edges, order):
        # Try to add edges in given order
        added = 0
        for u, v in order:
            if not (masks[u] & (1 << v)): # Not already connected
                if can_add(u, v, masks):
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    edges.append([u, v])
                    added += 1
        return added

    def local_search(masks, edges):
        # Try to improve by removing an edge and refilling
        current_edges = len(edges)
        improved = False
        # Try a few random removals
        for _ in range(10):
            if time.time() > deadline:
                break
            if current_edges == 0:
                break
            # Pick random edge
            idx = random.randint(0, current_edges - 1)
            u, v = edges[idx]
            
            # Remove it
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            edges.pop(idx)
            
            # Try to refill
            # Generate random order of non-edges
            non_edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if not (masks[i] & (1 << j)):
                        non_edges.append((i, j))
            random.shuffle(non_edges)
            
            # Greedy add
            for u2, v2 in non_edges:
                if can_add(u2, v2, masks):
                    masks[u2] |= (1 << v2)
                    masks[v2] |= (1 << u2)
                    edges.append([u2, v2])
            
            if len(edges) > current_edges:
                current_edges = len(edges)
                improved = True
            else:
                # Revert? No, we modified masks and edges in place.
                # We need to restore state if not improved.
                # This is complex. Let's copy state before modification.
                pass
        
        return improved

    # Main loop
    while time.time() < deadline:
        # Random restart
        masks = [0] * N
        edges = []
        
        # Generate random order of all pairs
        pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(pairs)
        
        # Greedy fill
        greedy_fill(masks, edges, pairs)
        
        # Local search (simple version: try removing edges)
        # To avoid complex state restoration, we can just do a few full restarts.
        # But local search is better.
        # Let's implement a safe local search with state copying.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_len = len(edges)
            # Try removing random edges
            for _ in range(5): # Try 5 random removals
                if time.time() > deadline:
                    break
                if not edges:
                    break
                idx = random.randint(0, len(edges) - 1)
                u, v = edges[idx]
                
                # Save state
                masks_copy = masks[:]
                edges_copy = edges[:]
                
                # Remove
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
                edges.pop(idx)
                
                # Refill
                non_edges = [(i, j) for i in range(N) for j in range(i+1, N) if not (masks[i] & (1 << j))]
                random.shuffle(non_edges)
                
                for u2, v2 in non_edges:
                    if can_add(u2, v2, masks):
                        masks[u2] |= (1 << v2)
                        masks[v2] |= (1 << u2)
                        edges.append([u2, v2])
                
                if len(edges) > current_len:
                    current_len = len(edges)
                    improved = True
                else:
                    # Restore
                    masks = masks_copy
                    edges = edges_copy
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = edges[:]
            
    return best_graph
