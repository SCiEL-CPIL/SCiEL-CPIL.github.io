import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute neighbors for each edge to speed up common neighbor checks?
    # Actually, we need to check common neighbors for ALL pairs (x,y) when we flip an edge (u,v).
    # The pairs affected are those involving u or v.
    # Specifically, if we flip (u,v):
    # 1. The pair (u,v) itself: common neighbors change? No, N(u) and N(v) change, but intersection is defined by other nodes.
    #    Wait, |N(u) ∩ N(v)| is the number of w such that w~u and w~v.
    #    Flipping (u,v) does NOT change whether w~u or w~v for w != u,v.
    #    So |N(u) ∩ N(v)| is UNCHANGED by flipping (u,v).
    # 2. Pairs (u, w) for w != u, v:
    #    N(u) changes (v added/removed).
    #    Intersection N(u) ∩ N(w) changes if v is in N(w).
    #    If v~w, then v is a common neighbor of u and w.
    #    So if we ADD (u,v), and v~w, then |N(u) ∩ N(w)| increases by 1.
    #    If we REMOVE (u,v), and v~w, then |N(u) ∩ N(w)| decreases by 1.
    # 3. Pairs (v, w) for w != u, v:
    #    Similar logic.
    
    # To make SA efficient, we maintain:
    # - adj: 20x20 matrix
    # - cn[u][v]: number of common neighbors for pair (u,v). Symmetric.
    # - m: number of edges
    # - penalty: sum over all pairs of max(0, cn[u][v] - 2)
    
    adj = [[0]*n for _ in range(n)]
    cn = [[0]*n for _ in range(n)]
    
    def update_cn_after_flip(u, v, adding):
        # Flip edge (u,v)
        # 1. Update adj
        adj[u][v] = 1 if adding else 0
        adj[v][u] = 1 if adding else 0
        
        # 2. Update cn for affected pairs
        # Affected pairs: (u, w) and (v, w) for all w != u, v
        # Also, does cn[u][v] change? No, as reasoned above.
        
        delta = 1 if adding else -1
        
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Pair (u, w)
            # v is a common neighbor of u and w if adj[v][w] is 1.
            # Note: adj[v][w] is current state.
            # If we are ADDING (u,v):
            #   If v~w, then v becomes a common neighbor of u and w. So cn[u][w] += 1.
            #   If not v~w, no change.
            # If we are REMOVING (u,v):
            #   If v~w, then v ceases to be a common neighbor of u and w. So cn[u][w] -= 1.
            #   If not v~w, no change.
            
            if adj[v][w]:
                cn[u][w] += delta
                cn[w][u] += delta
                
            # Pair (v, w)
            # u is a common neighbor of v and w if adj[u][w] is 1.
            if adj[u][w]:
                cn[v][w] += delta
                cn[w][v] += delta

    def get_penalty():
        # Sum of max(0, cn[u][v] - 2) for all u < v
        pen = 0
        for u in range(n):
            for v in range(u + 1, n):
                if cn[u][v] > 2:
                    pen += cn[u][v] - 2
        return pen

    # Initialize with a random sparse graph
    # Start with ~50% density? Or lower to be valid?
    # Let's start with a random graph with p=0.4
    # Then repair or let SA fix it.
    
    best_edges = []
    best_score = -1
    best_adj = None
    
    # Run multiple restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Reset
        adj = [[0]*n for _ in range(n)]
        cn = [[0]*n for _ in range(n)]
        
        # Initialize random graph
        for u, v in all_pairs:
            if random.random() < 0.5:
                adj[u][v] = 1
                adj[v][u] = 1
        
        # Calculate initial cn
        for u in range(n):
            for v in range(u + 1, n):
                cnt = 0
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        cnt += 1
                cn[u][v] = cnt
                cn[v][u] = cnt
        
        m = sum(adj[u][v] for u in range(n) for v in range(u+1, n))
        penalty = get_penalty()
        
        # SA Parameters
        T = 10.0
        cooling = 0.99
        min_T = 0.01
        steps_per_T = 100 # Adjust based on speed
        
        current_best_in_run = m - 10 * penalty
        best_edges_in_run = [[u, v] for u in range(n) for v in range(u+1, n) if adj[u][v]]
        best_adj_in_run = [row[:] for row in adj]
        best_cn_in_run = [row[:] for row in cn]
        
        while T > min_T and time.time() < deadline:
            for _ in range(steps_per_T):
                if time.time() > deadline:
                    break
                    
                # Pick a random edge to flip
                u, v = random.choice(all_pairs)
                
                # Calculate delta in penalty
                # Flipping (u,v) affects cn[u][w] and cn[v][w]
                # We need to calculate change in penalty for affected pairs.
                
                delta_pen = 0
                affected_pairs = set()
                
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    # Pair (u, w)
                    if adj[v][w]: # v is common neighbor
                        old_cn_uw = cn[u][w]
                        new_cn_uw = old_cn_uw + (1 if adj[u][v]==0 else -1) # if adding, +1; if removing, -1
                        # Wait, adj[u][v] is current state.
                        # If adj[u][v]==0, we are adding. new = old + 1.
                        # If adj[u][v]==1, we are removing. new = old - 1.
                        old_pen = max(0, old_cn_uw - 2)
                        new_pen = max(0, new_cn_uw - 2)
                        delta_pen += new_pen - old_pen
                        affected_pairs.add((u, w) if u < w else (w, u))
                        
                    # Pair (v, w)
                    if adj[u][w]: # u is common neighbor
                        old_cn_vw = cn[v][w]
                        new_cn_vw = old_cn_vw + (1 if adj[u][v]==0 else -1)
                        old_pen = max(0, old_cn_vw - 2)
                        new_pen = max(0, new_cn_vw - 2)
                        delta_pen += new_pen - old_pen
                        affected_pairs.add((v, w) if v < w else (w, v))
                        
                # Change in m
                delta_m = 1 if adj[u][v] == 0 else -1
                delta_score = delta_m - 10 * delta_pen
                
                # Acceptance probability
                if delta_score > 0:
                    accept = True
                else:
                    if T == 0:
                        accept = False
                    else:
                        accept = random.random() < math.exp(delta_score / T)
                        
                if accept:
                    # Perform flip
                    update_cn_after_flip(u, v, adding=(adj[u][v] == 0))
                    m += delta_m
                    penalty += delta_pen
                    
                    current_score = m - 10 * penalty
                    if current_score > current_best_in_run:
                        current_best_in_run = current_score
                        best_edges_in_run = [[u, v] for u in range(n) for v in range(u+1, n) if adj[u][v]]
                        best_adj_in_run = [row[:] for row in adj]
                        best_cn_in_run = [row[:] for row in cn]
                        
            T *= cooling
            
        # After SA, do a greedy fill to ensure maximality
        # This is crucial because SA might stop with valid but non-maximal graphs.
        # We use the best_adj_in_run as starting point.
        
        if time.time() < deadline:
            # Greedy fill on best_adj_in_run
            # We need to update cn if we add edges.
            # Let's just re-run a fast greedy fill on the final best graph from this restart.
            # Actually, we can just take the best graph found and try to add edges.
            
            fill_adj = [row[:] for row in best_adj_in_run]
            fill_cn = [row[:] for row in best_cn_in_run]
            
            # Helper to check if edge (u,v) can be added
            def can_add(u, v, f_adj, f_cn):
                # Check if adding (u,v) violates constraint
                # It increases cn[u][w] by 1 if v~w, and cn[v][w] by 1 if u~w.
                # We need new cn <= 2.
                # So if v~w, we need f_cn[u][w] < 2.
                # If u~w, we need f_cn[v][w] < 2.
                
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if f_adj[v][w] and f_cn[u][w] >= 2:
                        return False
                    if f_adj[u][w] and f_cn[v][w] >= 2:
                        return False
                return True

            # Try to add all possible edges
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for u, v in all_pairs:
                    if fill_adj[u][v]:
                        continue
                    if can_add(u, v, fill_adj, fill_cn):
                        # Add it
                        fill_adj[u][v] = 1
                        fill_adj[v][u] = 1
                        # Update cn
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if fill_adj[v][w]:
                                fill_cn[u][w] += 1
                                fill_cn[w][u] += 1
                            if fill_adj[u][w]:
                                fill_cn[v][w] += 1
                                fill_cn[w][v] += 1
                        improved = True
                        
            # Update global best
            final_edges = [[u, v] for u in range(n) for v in range(u+1, n) if fill_adj[u][v]]
            final_m = len(final_edges)
            
            # Verify validity just in case
            valid = True
            for u in range(n):
                for v in range(u+1, n):
                    if fill_cn[u][v] > 2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and final_m > best_score:
                best_score = final_m
                best_edges = final_edges
                
    # Fallback if no valid graph found (should not happen)
    if not best_edges:
        # Return empty graph
        return []
        
    return best_edges
