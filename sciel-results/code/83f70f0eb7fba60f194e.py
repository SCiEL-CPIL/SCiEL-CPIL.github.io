import time
import random

def construct():
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Precompute bit masks for vertices
    # adj[v] is an integer where bit i is 1 if (v,i) is an edge
    
    def iter_bits(mask):
        """Yields indices of set bits in mask."""
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    def would_create_c6(u, v, adj):
        """
        Check if adding edge (u, v) creates a C6.
        C6: u-a-x-y-b-v
        a in N(u), b in N(v), a != b, a != v, b != u.
        x in N(a) \ {u, v, b}, y in N(b) \ {u, v, a}.
        Edge (x, y) must exist.
        """
        nu = adj[u]
        nv = adj[v]
        
        # a in N(u), a != v
        a_mask = nu & ~(1 << v)
        
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            # b in N(v), b != u, b != a
            b_mask = nv & ~(1 << u) & ~(1 << a)
            
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                # x in N(a) \ {u, v, b}
                x_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << b)
                if not x_mask:
                    continue
                
                # y in N(b) \ {u, v, a}
                y_mask = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if not y_mask:
                    continue
                
                # Check if any edge (x, y) with x in x_mask, y in y_mask
                # Iterate over smaller set? Both are small.
                xm = x_mask
                while xm:
                    x_bit = xm & -xm
                    x = x_bit.bit_length() - 1
                    xm ^= x_bit
                    if adj[x] & y_mask:
                        return True
        return False

    def greedy_construct(seed):
        rng = random.Random(seed)
        adj = [0] * N
        edge_set = set()
        
        # Generate all edges
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if not would_create_c6(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                
        return edge_set, adj, len(edge_set)

    def local_search(edge_set, adj, max_iters=2000, seed=0):
        """
        Local search to improve edge count.
        1. Try to add any missing edge.
        2. Try 1-swap: remove e1, add e2, e3 (if possible).
        """
        rng = random.Random(seed)
        current_edges = set(edge_set)
        current_adj = list(adj)
        current_count = len(current_edges)
        
        # Precompute list of all possible edges for random sampling
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        no_improve = 0
        max_no_improve = 100
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            
            improved = False
            
            # Phase 1: Try to add any edge
            # We can iterate through all non-edges or sample
            # Sampling is faster if graph is dense
            added = False
            for _ in range(50):
                u, v = rng.choice(all_edges)
                if (u, v) not in current_edges:
                    if not would_create_c6(u, v, current_adj):
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        current_edges.add((u, v))
                        current_count += 1
                        improved = True
                        no_improve = 0
                        added = True
                        break
            
            if improved:
                continue
            
            no_improve += 1
            if no_improve > max_no_improve:
                break
            
            # Phase 2: Try 2-swap (remove 1, add 2)
            # This allows escaping local optima where no single edge can be added
            if current_count == 0:
                break
                
            e_remove = rng.choice(list(current_edges))
            u1, v1 = e_remove
            
            # Temporarily remove
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            current_edges.discard(e_remove)
            
            # Try to add two random non-edges
            added_count = 0
            added_edges = []
            
            for _ in range(100):
                u2, v2 = rng.choice(all_edges)
                if (u2, v2) in current_edges or (u2, v2) == (u1, v1):
                    continue
                if not would_create_c6(u2, v2, current_adj):
                    current_adj[u2] |= (1 << v2)
                    current_adj[v2] |= (1 << u2)
                    current_edges.add((u2, v2))
                    added_count += 1
                    added_edges.append((u2, v2))
                    if added_count == 2:
                        break
            
            if added_count == 2:
                current_count += 1 # Net gain +1
                no_improve = 0
            else:
                # Revert
                for u2, v2 in added_edges:
                    current_adj[u2] &= ~(1 << v2)
                    current_adj[v2] &= ~(1 << u2)
                    current_edges.discard((u2, v2))
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                current_edges.add(e_remove)
                
        return current_edges, current_adj, current_count

    best_edges = []
    best_adj = [0] * N
    best_count = 0
    
    # Phase 1: Multiple Greedy Restarts
    # Greedy is very fast. Let's do as many as possible.
    num_greedy = 0
    while time.time() < deadline - 20: # Reserve 20s for local search
        seed = num_greedy
        edges, adj, count = greedy_construct(seed)
        if count > best_count:
            best_count = count
            best_edges = edges
            best_adj = adj
        num_greedy += 1
        
    # Phase 2: Local Search on Best
    if best_edges and time.time() < deadline - 5:
        # Run local search on the best found
        # We might want to run it on top few, but best is usually sufficient
        # Let's run it on the best one found
        new_edges, new_adj, new_count = local_search(best_edges, best_adj, max_iters=5000, seed=42)
        if new_count > best_count:
            best_count = new_count
            best_edges = new_edges
            best_adj = new_adj
            
        # Try a few more seeds for local search if time permits
        for ls_seed in range(5):
            if time.time() >= deadline - 2:
                break
            # Restart local search from best_edges but with different randomization
            # Note: local_search modifies adj, so we need to pass a copy or reconstruct
            adj_copy = list(best_adj)
            new_edges, new_adj, new_count = local_search(best_edges, adj_copy, max_iters=2000, seed=100+ls_seed)
            if new_count > best_count:
                best_count = new_count
                best_edges = new_edges
                best_adj = new_adj

    # Final check: try to add any remaining edges
    if best_edges and time.time() < deadline:
        for u in range(N):
            for v in range(u + 1, N):
                if (u, v) not in best_edges:
                    if not would_create_c6(u, v, best_adj):
                        best_adj[u] |= (1 << v)
                        best_adj[v] |= (1 << u)
                        best_edges.add((u, v))
                        best_count += 1

    return [[u, v] for u, v in best_edges]
