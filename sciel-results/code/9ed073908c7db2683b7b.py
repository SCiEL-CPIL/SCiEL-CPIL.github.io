import random
import time
import sys

def construct():
    n = 20
    # Deadline: 110 seconds to be safe
    start_time = time.time()
    max_time = 110.0
    
    best_edges = []
    best_count = 0
    
    def is_valid_add(u, v, adj_mask, common):
        # u and v must not be connected
        if (adj_mask[u] >> v) & 1:
            return False
        
        # Check common neighbors of u and v
        if common[u][v] > 2:
            return False
        
        # Check neighbors of v
        # If w is neighbor of v, then C(u, w) will increase by 1
        # So we need C(u, w) < 2 currently
        mask_v = adj_mask[v]
        # Iterate through set bits in mask_v
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u: # w cannot be u if u-v not connected? Actually if u-v not connected, u is not in N(v).
                if common[u][w] >= 2:
                    return False
            m -= lsb
            
        # Check neighbors of u
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v:
                if common[v][w] >= 2:
                    return False
            m -= lsb
            
        return True

    def add_edge(u, v, adj_mask, common, edges):
        # Add edge u-v
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append([u, v])
        
        # Update common neighbors
        # C(u, v) doesn't change value but is now "active"? No, C is just count.
        # For w in N(v): C(u, w) += 1
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u:
                common[u][w] += 1
                common[w][u] += 1
            m -= lsb
            
        # For w in N(u): C(v, w) += 1
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v:
                common[v][w] += 1
                common[w][v] += 1
            m -= lsb

    def remove_edge(u, v, adj_mask, common, edges_set, edges_list_idx):
        # Remove edge u-v
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        
        # Update common neighbors
        # For w in N(v): C(u, w) -= 1
        mask_v = adj_mask[v] # After removal, v is not in N(v)? No, N(v) is neighbors of v.
        # Wait, N(v) in the context of update:
        # Before removal, N(v) contained u? No, we are removing (u,v).
        # So u was in N(v).
        # The neighbors w of v (excluding u) are those that contributed to C(u, w) via v.
        # So we iterate through current N(v) (which does not include u anymore? 
        # If we update mask first, N(v) no longer has u.
        # The set of w such that v in N(w) and w in N(u) is what matters?
        # Let's trace:
        # C(u, w) = |N(u) \cap N(w)|.
        # v was in N(u) and v was in N(w) (if w in N(v)).
        # So if w was in N(v) (before removal), then v was a common neighbor.
        # So we need to decrement C(u, w) for all w in N(v) (before removal).
        # Since we cleared the bit, we should iterate through the OLD N(v).
        # Let's pass the old mask or compute before clearing.
        
        # Let's re-implement remove to take care of order.
        pass # This function is a bit tricky to inline in the loop efficiently without state issues.
        # Let's just use a simple list of edges and recompute common?
        # n=20, recomputing common is O(n^3) = 8000 ops. Very fast.
        # So for local search, we can just keep the edge list and recompute common when needed?
        # Or maintain it carefully.

    # Given the complexity of maintaining 'common' during removal, and n=20 being small,
    # we can use a simpler state representation for the local search:
    # Just the adjacency matrix (list of lists or sets).
    # And recompute 'common' on the fly for validation?
    # Validation is O(n^2) if we recompute common.
    # Total edges ~100. Pairs ~1000.
    # 1000 * 400 = 400k ops.
    # 1000 restarts * 1000 steps = 1e6 validations * 400 = 4e8 ops.
    # Might be too slow in Python if done naively.
    
    # Let's stick to the bitmask + common matrix, but handle removal carefully.
    
    def try_add_edge(u, v, state):
        # state: adj_mask, common, edges
        if is_valid_add(u, v, state['adj_mask'], state['common']):
            add_edge(u, v, state['adj_mask'], state['common'], state['edges'])
            return True
        return False

    def remove_edge_from_state(u, v, state):
        # Remove edge u-v
        state['adj_mask'][u] &= ~(1 << v)
        state['adj_mask'][v] &= ~(1 << u)
        state['edges'].remove([u, v])
        
        # Update common
        # We need to decrement common[u][w] for all w in OLD N(v)
        # and common[v][w] for all w in OLD N(u)
        # OLD N(v) is state['adj_mask'][v] BEFORE clearing?
        # We already cleared it.
        # So we should have saved it or iterate differently.
        
        # Let's re-write add/remove to be symmetric and safe.
        # Actually, let's just recompute 'common' from 'adj_mask' after any change.
        # n=20. Recompute is fast.
        
        pass

    # Revised Plan:
    # Use adjacency matrix (list of bitmasks).
    # Function `check_validity(adj_mask, u, v)`:
    #   Computes common neighbors count on the fly?
    #   Or maintains common matrix.
    #   Maintaining is better.
    
    # Let's implement a robust local search with explicit common matrix updates.
    
    def run_local_search(initial_edges):
        # Initialize state
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        
        for u, v in initial_edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edges.append([u, v])
            
        # Compute initial common
        for u in range(n):
            for v in range(u+1, n):
                # Count common neighbors
                # N(u) & N(v)
                mask = adj_mask[u] & adj_mask[v]
                cnt = bin(mask).count('1')
                common[u][v] = cnt
                common[v][u] = cnt
                
        best_edges = list(edges)
        best_count = len(edges)
        
        # Local Search Loop
        # We will try to add edges. If stuck, remove a random one and retry.
        # To avoid infinite loops, limit steps per restart.
        
        for step in range(1000):
            # Try to add any edge
            added = False
            # Randomize order of potential edges
            # We can just iterate through all pairs and pick a random valid one
            # Or iterate through all and add first valid one?
            # Better: collect all valid edges, pick one randomly.
            
            valid_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if is_valid_add(u, v, adj_mask, common):
                        valid_edges.append((u, v))
                        
            if valid_edges:
                u, v = random.choice(valid_edges)
                # Add it
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append([u, v])
                
                # Update common
                # For w in N(v): common[u][w] += 1
                mask_v = adj_mask[v]
                m = mask_v
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if w != u:
                        common[u][w] += 1
                        common[w][u] += 1
                    m -= lsb
                    
                # For w in N(u): common[v][w] += 1
                mask_u = adj_mask[u]
                m = mask_u
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if w != v:
                        common[v][w] += 1
                        common[w][v] += 1
                    m -= lsb
                    
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = list(edges)
                    
            else:
                # No edge can be added. Remove a random edge to escape local optimum.
                if not edges:
                    break
                idx = random.randint(0, len(edges)-1)
                u, v = edges[idx]
                
                # Remove edge
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                edges.pop(idx)
                
                # Update common (decrement)
                # Need OLD N(v) and OLD N(u)
                # Since we cleared bits, we can't easily get old N.
                # But we know u was in N(v) and v was in N(u).
                # The neighbors w of v (excluding u) are those where v was common neighbor of u and w.
                # So we need to iterate through current N(v) (which is OLD N(v) \ {u})?
                # No, OLD N(v) included u.
                # Current N(v) does not include u.
                # The set of w such that v in N(w) and w in N(u) is what we need to decrement?
                # C(u, w) = |N(u) \cap N(w)|.
                # v was in N(u) and v was in N(w) (if w in N(v)).
                # So for all w in OLD N(v) \ {u}, we decrement C(u, w).
                # OLD N(v) \ {u} is exactly the CURRENT N(v) (since we removed u).
                # So we iterate through CURRENT N(v).
                
                mask_v = adj_mask[v]
                m = mask_v
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    # w is in N(v). Since u is not in N(v) anymore, w != u is guaranteed?
                    # Yes, because u was removed.
                    common[u][w] -= 1
                    common[w][u] -= 1
                    m -= lsb
                    
                mask_u = adj_mask[u]
                m = mask_u
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    common[v][w] -= 1
                    common[w][v] -= 1
                    m -= lsb
                    
        return best_edges

    # Main loop with restarts
    while time.time() - start_time < max_time:
        # Random initialization: sparse random graph
        # Or empty?
        # Let's start empty and let local search build it.
        # Or start with a random set of edges?
        # Starting empty is safer.
        
        # Greedy initialization
        init_edges = []
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        
        # Random order of edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if is_valid_add(u, v, adj_mask, common):
                # Add
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                init_edges.append([u, v])
                
                # Update common
                mask_v = adj_mask[v]
                m = mask_v
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if w != u:
                        common[u][w] += 1
                        common[w][u] += 1
                    m -= lsb
                    
                mask_u = adj_mask[u]
                m = mask_u
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if w != v:
                        common[v][w] += 1
                        common[w][v] += 1
                    m -= lsb

        # Local search from this state
        # We need to pass the state to run_local_search or replicate the logic.
        # Let's refactor run_local_search to take initial state.
        
        # For simplicity, I will just call a function that takes the current state and improves it.
        # Since I defined run_local_search to take initial_edges, I need to pass the state.
        # Let's modify run_local_search to take adj_mask, common, edges.
        
        # Re-implementing the improvement part directly here to avoid function call overhead/state passing issues.
        
        current_edges = init_edges
        current_adj = adj_mask
        current_common = common
        current_count = len(current_edges)
        
        # Local search
        for step in range(500): # Limit steps per restart
            valid_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if is_valid_add(u, v, current_adj, current_common):
                        valid_edges.append((u, v))
                        
            if valid_edges:
                u, v = random.choice(valid_edges)
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges.append([u, v])
                
                # Update common
                mask_v = current_adj[v]
                m = mask_v
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if w != u:
                        current_common[u][w] += 1
                        current_common[w][u] += 1
                    m -= lsb
                    
                mask_u = current_adj[u]
                m = mask_u
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if w != v:
                        current_common[v][w] += 1
                        current_common[w][v] += 1
                    m -= lsb
                    
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = list(current_edges)
            else:
                if not current_edges:
                    break
                idx = random.randint(0, len(current_edges)-1)
                u, v = current_edges[idx]
                
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                current_edges.pop(idx)
                
                mask_v = current_adj[v]
                m = mask_v
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    current_common[u][w] -= 1
                    current_common[w][u] -= 1
                    m -= lsb
                    
                mask_u = current_adj[u]
                m = mask_u
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    current_common[v][w] -= 1
                    current_common[w][v] -= 1
                    m -= lsb

    return best_edges

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Found {len(edges)} edges")
