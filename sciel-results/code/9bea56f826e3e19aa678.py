import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    n = 20
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_k33_free(adj):
        """
        Check if graph defined by adj (list of bitmasks) is K3,3-free.
        A K3,3 exists if there are 3 vertices A={a1,a2,a3} and 3 vertices B={b1,b2,b3}
        such that all 9 edges between A and B exist.
        Equivalent: No 3 vertices have 3 common neighbors.
        """
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common_ij = ai & adj[j]
                if common_ij == 0:
                    continue
                # If common_ij has < 3 bits, it can't be part of a K3,3 with k
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    common_ijk = common_ij & adj[k]
                    if common_ijk.bit_count() >= 3:
                        return False
        return True

    def edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    def get_edges(adj):
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    def greedy_construct(p_order):
        """
        Build a valid graph by adding edges in a specific order if they don't violate K3,3.
        p_order is a list of (u,v) pairs.
        """
        adj = [0] * n
        # We can speed this up by checking violations locally, but for n=20, full check is okay if not done too often.
        # However, greedy construction is O(E * CheckCost). CheckCost is O(n^3) worst case.
        # 190 edges * 8000 ops = ~1.5M ops. Very fast.
        
        for u, v in p_order:
            if (adj[u] >> v) & 1:
                continue
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_k33_free(adj):
                # Keep it
                pass
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj

    def local_search(adj, max_iters=50):
        """
        Improve the graph by trying to add missing edges or performing swaps.
        Returns improved adj.
        """
        current_edges = edge_count(adj)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 1.0:
                break
                
            # 1. Try to add any missing edge
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] >> v) & 1]
            if missing:
                random.shuffle(missing)
                for u, v in missing:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if is_k33_free(adj):
                        current_edges += 1
                        improved = True
                        break
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
            
            # 2. If no edge could be added, try a swap (remove one, add one)
            if not improved:
                existing = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
                if existing and missing:
                    random.shuffle(existing)
                    # Limit swap attempts to avoid slowness
                    attempts = 0
                    for u1, v1 in existing[:20]:
                        if improved or attempts > 100:
                            break
                        for u2, v2 in missing[:20]:
                            if attempts > 100:
                                break
                            attempts += 1
                            if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                                continue
                            
                            # Perform swap
                            adj[u1] &= ~(1 << v1)
                            adj[v1] &= ~(1 << u1)
                            adj[u2] |= (1 << v2)
                            adj[v2] |= (1 << u2)
                            
                            if is_k33_free(adj):
                                improved = True
                                break
                            else:
                                # Revert
                                adj[u1] |= (1 << v1)
                                adj[v1] |= (1 << u1)
                                adj[u2] &= ~(1 << v2)
                                adj[v2] &= ~(1 << u2)
                    if improved:
                        break # Restart loop to try adding edges again
                    else:
                        # Try a random kick: remove 2 random edges, add 2 random missing edges
                        # This is a 2-opt move.
                        if len(existing) >= 2 and len(missing) >= 2:
                            rem1, rem2 = random.sample(existing, 2)
                            add1, add2 = random.sample(missing, 2)
                            # Check if we are removing/adding the same edges
                            if len({(min(a[0],a[1]), max(a[0],a[1])) for a in [rem1, rem2, add1, add2]}) == 4:
                                # Apply
                                for u,v in [rem1, rem2]:
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                for u,v in [add1, add2]:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                                
                                if is_k33_free(adj):
                                    improved = True
                                else:
                                    # Revert
                                    for u,v in [rem1, rem2]:
                                        adj[u] |= (1 << v)
                                        adj[v] |= (1 << u)
                                    for u,v in [add1, add2]:
                                        adj[u] &= ~(1 << v)
                                        adj[v] &= ~(1 << u)

        return adj

    best_edges = 0
    best_adj = None

    # Strategy: Multiple restarts of Greedy Construction + Local Search
    # Greedy construction with random order is often better than random density + purge
    while time.time() < deadline - 2.0:
        # Create a random order of edges
        random_order = all_edges[:]
        random.shuffle(random_order)
        
        # Greedy phase
        adj = greedy_construct(random_order)
        
        # Local Search phase
        adj = local_search(adj, max_iters=10)
        
        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
            # If we found a very good graph, we can be more patient, but let's keep moving
            # Known ex(20, K3,3) is likely around 80-90. 
            # If we hit 90+, we can slow down search to verify/improve slightly.
            
        # If we have a high score, we might want to do more local search on the best one
        if best_edges > 80 and time.time() < deadline - 10.0:
             # Do a few more iterations of local search on the best found so far
             adj_copy = best_adj[:]
             adj_copy = local_search(adj_copy, max_iters=5)
             ec = edge_count(adj_copy)
             if ec > best_edges:
                 best_edges = ec
                 best_adj = adj_copy[:]

    if best_adj:
        return get_edges(best_adj)
    else:
        # Fallback: empty graph is valid
        return []
