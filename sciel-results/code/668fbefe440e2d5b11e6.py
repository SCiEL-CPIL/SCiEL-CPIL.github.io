import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 20
    
    # Precompute edge indices
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    idx_to_edge = {i: e for i, e in enumerate(all_edges)}
    m_total = len(all_edges)

    def is_valid_add(adj, u, v):
        # Check common neighbors of u and v
        if (adj[u] & adj[v]).bit_count() >= 3:
            return False
        
        # Check neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if (adj[v] & adj[w]).bit_count() >= 2:
                return False
            mask_u ^= lsb
            
        # Check neighbors of v
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            x = lsb.bit_length() - 1
            if (adj[u] & adj[x]).bit_count() >= 2:
                return False
            mask_v ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def greedy_fill(adj):
        """Add all possible edges in random order."""
        # We need a list of missing edges
        missing = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] >> v) & 1:
                    missing.append((u, v))
        random.shuffle(missing)
        count = count_edges(adj)
        for u, v in missing:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        return count

    def local_search(adj):
        """
        Iteratively try to improve the graph.
        1. Greedy fill.
        2. Try removing each edge and refilling. If it leads to a larger graph, keep it.
        """
        improved = True
        count = count_edges(adj)
        while improved and time.time() < deadline - 10:
            improved = False
            # Greedy fill first
            count = greedy_fill(adj)
            
            # Try removing edges
            present_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        present_edges.append((u, v))
            
            random.shuffle(present_edges)
            
            for u, v in present_edges:
                if time.time() > deadline - 10:
                    break
                # Remove edge
                remove_edge(adj, u, v)
                # Refill
                new_count = greedy_fill(adj)
                
                if new_count > count:
                    count = new_count
                    improved = True
                else:
                    # Restore edge (it was valid before, and we only removed it, 
                    # but wait, greedy_fill might have added other edges that conflict? 
                    # No, greedy_fill only adds valid edges. 
                    # However, if we restore (u,v), is it still valid?
                    # The graph after refill is a superset of (graph - (u,v)).
                    # If (u,v) was valid in the original graph, it might not be valid now 
                    # if the refill added edges that create a K2,3 with u and v.
                    # So we must check validity before restoring.
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                    else:
                        # This case shouldn't happen if we track counts correctly?
                        # Actually, if we can't restore, the count is new_count.
                        # But we only accept if new_count > count.
                        # If new_count <= count, we should restore if possible to maintain state,
                        # or just accept the worse state? 
                        # Better: only accept change if strictly better.
                        # If not strictly better, revert to previous state.
                        # But we mutated adj.
                        # Simplest: keep a copy? Too expensive.
                        # Alternative: Only perform the move if we can guarantee restoration or if it's an improvement.
                        # If it's not an improvement, we want to undo.
                        # If is_valid_add fails, we are stuck with a smaller graph.
                        # So we should check is_valid_add BEFORE deciding to keep the change?
                        # No, the change is "remove u,v and refill".
                        # If new_count <= count, we discard the change.
                        # To discard, we need to restore the original adj.
                        # This suggests we should copy adj before the move.
                        pass

        return count

    # Better Local Search with Copy
    def local_search_safe(adj):
        improved = True
        count = count_edges(adj)
        while improved and time.time() < deadline - 10:
            improved = False
            count = greedy_fill(adj)
            
            present_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        present_edges.append((u, v))
            random.shuffle(present_edges)
            
            for u, v in present_edges:
                if time.time() > deadline - 10:
                    break
                # Copy adj
                adj_copy = adj[:]
                remove_edge(adj_copy, u, v)
                new_count = greedy_fill(adj_copy)
                
                if new_count > count:
                    adj[:] = adj_copy
                    count = new_count
                    improved = True
        return count

    # Run multiple restarts
    best_adj = [0]*n
    best_count = 0
    
    while time.time() < deadline - 5:
        # Random initialization
        adj = [0]*n
        # Random greedy start
        order = all_edges[:]
        random.shuffle(order)
        for u, v in order:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
        
        # Polish
        count = local_search_safe(adj)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
        # If we reached a high count, do more SA
        if best_count >= 40 and time.time() < deadline - 20:
            # Simulated Annealing on best_adj
            T = 1.0
            cooling = 0.99
            current_adj = best_adj[:]
            current_count = best_count
            
            while T > 0.01 and time.time() < deadline - 5:
                # Move: Remove a random edge, refill
                present = [(u,v) for u in range(n) for v in range(u+1,n) if (current_adj[u]>>v)&1]
                if not present: break
                u, v = random.choice(present)
                
                next_adj = current_adj[:]
                remove_edge(next_adj, u, v)
                next_count = greedy_fill(next_adj)
                
                delta = next_count - current_count
                if delta > 0 or random.random() < math.exp(-delta/T):
                    current_adj = next_adj
                    current_count = next_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = current_adj[:]
                
                T *= cooling

    # Convert to list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
