"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 140s.
    deadline = start_time + 140.0
    
    n = 20
    
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u, v) maintains K2,3-freeness.
        Uses NumPy for fast bit operations.
        """
        # 1. Check common neighbors of u and v.
        # The set N(u) ∩ N(v) does not change when adding (u,v) to the graph 
        # in terms of *other* vertices w. u is not in N(u), v is not in N(v).
        # So we just need current count <= 2.
        common_uv = np.sum(adj[u] & adj[v])
        if common_uv > 2:
            return False
            
        # 2. Check pairs (u, w) for w in N(v).
        # Adding (u,v) adds v to N(u).
        # For any w in N(v) (w != u), v is now a common neighbor of u and w.
        # So new |N(u) ∩ N(w)| = old |N(u) ∩ N(w)| + 1.
        # We need new <= 2, so old <= 1.
        # adj[v] is a boolean array of neighbors of v.
        # We need to check all w where adj[v][w] is True.
        # We can iterate over these indices.
        neighbors_v = np.where(adj[v])[0]
        for w in neighbors_v:
            if w == u:
                continue
            # Check common neighbors of u and w
            # If this is > 1, adding (u,v) makes it > 2, which is invalid.
            common_uw = np.sum(adj[u] & adj[w])
            if common_uw > 1:
                return False
                
        # 3. Check pairs (v, w) for w in N(u).
        # Adding (u,v) adds u to N(v).
        # For any w in N(u) (w != v), u is now a common neighbor of v and w.
        # So new |N(v) ∩ N(w)| = old |N(v) ∩ N(w)| + 1.
        # We need new <= 2, so old <= 1.
        neighbors_u = np.where(adj[u])[0]
        for w in neighbors_u:
            if w == v:
                continue
            # Check common neighbors of v and w
            common_vw = np.sum(adj[v] & adj[w])
            if common_vw > 1:
                return False
                
        return True

    def greedy_min_common(adj, time_limit):
        """
        Greedy construction picking the edge with minimum common neighbors.
        This heuristic tends to distribute edges evenly and avoid premature
        saturation of the common-neighbor limit.
        """
        edges = []
        # Precompute all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        remaining_edges = list(all_edges)
        
        while remaining_edges:
            if time.time() > time_limit:
                break
                
            # Find valid edges and their costs
            valid_edges = []
            min_cost = float('inf')
            
            # Optimization: If we have many edges, checking all remaining is O(E * V^2).
            # E <= 190, V=20. 190 * 400 = 76k ops per step. 190 steps -> 14M ops.
            # This is fine in Python if the inner loop is tight, but is_valid_add is a function call.
            # We will just filter and score.
            
            for u, v in remaining_edges:
                if is_valid_add(adj, u, v):
                    # Cost: number of common neighbors. Lower is better.
                    cost = np.sum(adj[u] & adj[v])
                    valid_edges.append((cost, u, v))
                    
            if not valid_edges:
                break
                
            # Find minimum cost
            # To add some diversity, we can pick randomly from the set of edges
            # that have the minimum cost.
            min_cost_val = min(c for c, _, _ in valid_edges)
            best_candidates = [(u, v) for c, u, v in valid_edges if c == min_cost_val]
            
            # Pick one
            u, v = random.choice(best_candidates)
            
            # Add to graph
            adj[u, v] = True
            adj[v, u] = True
            edges.append([u, v])
            
            # Remove from remaining
            # Since we are iterating over a list, removing is O(N).
            # 190 items, so it's cheap.
            remaining_edges.remove((u, v))
            
        return edges, adj

    def local_search(adj, edges, time_limit):
        """
        Improve the graph by trying to add missing edges and performing
        remove-refill moves.
        """
        best_edges = edges
        best_adj = adj
        best_count = len(best_edges)
        
        current_adj = best_adj.copy()
        current_edges = list(best_edges)
        
        # Phase 1: Greedy Fill
        # Try to add any missing edge that is valid.
        all_possible = [(i, j) for i in range(n) for j in range(i + 1, n) if not current_adj[i, j]]
        random.shuffle(all_possible)
        for u, v in all_possible:
            if time.time() > time_limit:
                break
            if is_valid_add(current_adj, u, v):
                current_adj[u, v] = True
                current_adj[v, u] = True
                current_edges.append([u, v])
                
        if len(current_edges) > best_count:
            best_edges = current_edges
            best_adj = current_adj
            best_count = len(best_edges)
            
        # Phase 2: Remove and Refill
        # If we are stuck, removing an edge might allow adding two others.
        max_ls_iters = 100  # We have time, NumPy is fast
        for _ in range(max_ls_iters):
            if time.time() > time_limit:
                break
            
            # Pick a random edge to remove
            if not current_edges:
                break
            rem_idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[rem_idx]
            
            # Remove it
            current_adj[u, v] = False
            current_adj[v, u] = False
            current_edges.pop(rem_idx)
            
            # Try to greedily add back any valid edges, including the removed one?
            # No, we want to see if we can add *different* edges.
            # We will try to add ANY valid missing edge.
            all_missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not current_adj[i, j]]
            random.shuffle(all_missing)
            
            for x, y in all_missing:
                if time.time() > time_limit:
                    break
                if is_valid_add(current_adj, x, y):
                    current_adj[x, y] = True
                    current_adj[y, x] = True
                    current_edges.append([x, y])
                    
            # Check if we improved
            if len(current_edges) > best_count:
                best_edges = current_edges
                best_adj = current_adj
                best_count = len(best_edges)
            else:
                # If we didn't improve, revert to the state before removal
                # to avoid drifting into worse states
                current_adj = best_adj.copy()
                current_edges = list(best_edges)
                
        return best_edges, best_adj

    # Main Logic
    best_overall_edges = []
    best_overall_count = 0
    
    # We will run the greedy construction a few times with different random seeds
    # (implicitly via random.choice in tie-breaking) and then apply local search.
    
    num_greedy_restarts = 5
    
    for r in range(num_greedy_restarts):
        if time.time() > deadline - 10:
            break
            
        adj = np.zeros((n, n), dtype=bool)
        edges, adj = greedy_min_common(adj, deadline - 10)
        
        if len(edges) > best_overall_count:
            best_overall_count = len(edges)
            best_overall_edges = edges
            
        # Apply local search to this result
        if time.time() > deadline - 5:
            break
            
        improved_edges, improved_adj = local_search(adj, edges, deadline - 2)
        
        if len(improved_edges) > best_overall_count:
            best_overall_count = len(improved_edges)
            best_overall_edges = improved_edges
            
    return best_overall_edges
# EVOLVE-BLOCK-END
