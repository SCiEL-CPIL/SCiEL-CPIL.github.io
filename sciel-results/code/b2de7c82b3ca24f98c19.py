import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = 20
    
    def popcount(x):
        return x.bit_count()
        
    def init_adj():
        # Start with a random graph, maybe slightly sparse to start
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def calc_metrics(adj):
        """Calculate total edges and total violations."""
        edges = 0
        violations = 0
        for u in range(n):
            edges += popcount(adj[u])
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if popcount(common) > 2:
                    violations += 1
        return edges // 2, violations

    def update_metrics(adj, u, v, edges, violations, common_counts):
        """
        Toggle edge (u,v) in adj.
        Update edges, violations, and common_counts.
        Returns new edges, violations.
        Assumes u < v for common_counts indexing convenience, but handles any order.
        """
        # Determine if edge is being added or removed
        is_present = (adj[u] >> v) & 1
        if is_present:
            # Removing edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            delta_edges = -1
        else:
            # Adding edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            delta_edges = 1
            
        # Update common counts for pairs involving u or v
        # Pairs (u, w) and (v, w) for w != u, v
        # And pair (u, v)
        
        # Helper to get/set common count
        def get_cc(a, b):
            if a > b: a, b = b, a
            return common_counts[a][b]
            
        def set_cc(a, b, val):
            if a > b: a, b = b, a
            common_counts[a][b] = val

        # 1. Update pair (u, v)
        old_uv = get_cc(u, v)
        new_uv = popcount(adj[u] & adj[v])
        if old_uv > 2 and new_uv <= 2:
            violations -= 1
        elif old_uv <= 2 and new_uv > 2:
            violations += 1
        set_cc(u, v, new_uv)

        # 2. Update pairs (u, w) and (v, w)
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Pair (u, w)
            old_uw = get_cc(u, w)
            new_uw = popcount(adj[u] & adj[w])
            if old_uw > 2 and new_uw <= 2:
                violations -= 1
            elif old_uw <= 2 and new_uw > 2:
                violations += 1
            set_cc(u, w, new_uw)
            
            # Pair (v, w)
            old_vw = get_cc(v, w)
            new_vw = popcount(adj[v] & adj[w])
            if old_vw > 2 and new_vw <= 2:
                violations -= 1
            elif old_vw <= 2 and new_vw > 2:
                violations += 1
            set_cc(v, w, new_vw)

        edges += delta_edges
        return edges, violations

    def run_sa(adj_init, time_limit):
        adj = list(adj_init)
        edges, violations = calc_metrics(adj)
        
        # Initialize common_counts
        common_counts = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                common_counts[u][v] = popcount(adj[u] & adj[v])
        
        best_edges = 0
        best_adj = None
        if violations == 0:
            best_edges = edges
            best_adj = list(adj)
            
        T = 5.0
        T_min = 0.01
        cooling = 0.9999
        penalty = 1000.0
        
        while time.time() < time_limit:
            # Pick random edge to toggle
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            # Calculate delta
            # We need to know if the move is valid in terms of cost
            # Cost = -Edges + Penalty * Violations
            
            # Perform toggle to see new state? 
            # To avoid modifying state if rejected, we can calculate delta analytically 
            # or use a copy. Given N=20, copying is cheap.
            adj_copy = list(adj)
            new_edges, new_violations = update_metrics(adj_copy, u, v, edges, violations, 
                                                        [[0]*n for _ in range(n)]) # This is wrong, we need to pass the counts or recalc
            # The update_metrics function above modifies the common_counts passed to it.
            # We need to be careful.
            
            # Let's re-structure: 
            # 1. Save old state of affected common counts.
            # 2. Toggle.
            # 3. Recalc affected common counts.
            # 4. Calculate delta.
            # 5. If rejected, revert.
            
            # This is getting messy. Let's use the "calculate from scratch for affected" approach 
            # without maintaining a global common_counts matrix in the main loop state, 
            # but rather just computing the delta violations directly.
            
            # Actually, maintaining the matrix is faster.
            # Let's refactor run_sa to maintain the matrix properly.
            
            pass # Placeholder, will implement below

    # Refactored SA
    def run_sa_optimized(adj_init, time_limit):
        adj = list(adj_init)
        edges, violations = calc_metrics(adj)
        
        # common_counts[u][v] for u < v
        cc = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                cc[u][v] = popcount(adj[u] & adj[v])
        
        best_edges = 0
        best_adj = None
        if violations == 0:
            best_edges = edges
            best_adj = list(adj)
            
        T = 10.0
        T_min = 0.01
        cooling = 0.99995
        penalty = 500.0
        
        # Precompute all pairs for iteration? No, random is fine.
        
        while time.time() < time_limit:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            # Calculate Delta
            # Toggle edge (u,v)
            is_present = (adj[u] >> v) & 1
            
            # Affected pairs: (u,v), (u,w) for w!=u,v, (v,w) for w!=u,v
            # We need old and new common counts for these pairs.
            
            delta_edges = 1 if not is_present else -1
            
            # Calculate new common counts for affected pairs
            # Note: adj is not modified yet. We calculate what it WOULD be.
            new_adj_u = adj[u] | (1 << v) if not is_present else (adj[u] & ~(1 << v))
            new_adj_v = adj[v] | (1 << u) if not is_present else (adj[v] & ~(1 << u))
            
            delta_violations = 0
            
            # Pair (u, v)
            old_uv = cc[u][v]
            new_uv = popcount(new_adj_u & new_adj_v)
            if old_uv > 2 and new_uv <= 2: delta_violations -= 1
            elif old_uv <= 2 and new_uv > 2: delta_violations += 1
            
            # Pairs (u, w) and (v, w)
            for w in range(n):
                if w == u or w == v:
                    continue
                
                # Pair (u, w)
                if u < w:
                    old_uw = cc[u][w]
                else:
                    old_uw = cc[w][u]
                new_uw = popcount(new_adj_u & adj[w])
                if old_uw > 2 and new_uw <= 2: delta_violations -= 1
                elif old_uw <= 2 and new_uw > 2: delta_violations += 1
                
                # Pair (v, w)
                if v < w:
                    old_vw = cc[v][w]
                else:
                    old_vw = cc[w][v]
                new_vw = popcount(new_adj_v & adj[w])
                if old_vw > 2 and new_vw <= 2: delta_violations -= 1
                elif old_vw <= 2 and new_vw > 2: delta_violations += 1
                
            delta_cost = -delta_edges + penalty * delta_violations
            
            # Acceptance
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Accept move
                if not is_present:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                edges += delta_edges
                violations += delta_violations
                
                # Update cc matrix
                cc[u][v] = new_uv
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if u < w:
                        cc[u][w] = new_uw
                    else:
                        cc[w][u] = new_uw
                    if v < w:
                        cc[v][w] = new_vw
                    else:
                        cc[w][v] = new_vw
                        
                if violations == 0 and edges > best_edges:
                    best_edges = edges
                    best_adj = list(adj)
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
        return best_adj, best_edges

    # Multi-start
    best_adj_overall = None
    best_edges_overall = 0
    
    # Try a few initial configurations
    starts = 10
    for i in range(starts):
        if time.time() > deadline - 10:
            break
            
        # Random init
        adj_init = init_adj()
        
        # Run SA
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
            
        adj_res, edges_res = run_sa_optimized(adj_init, time.time() + remaining)
        
        if edges_res > best_edges_overall:
            best_edges_overall = edges_res
            best_adj_overall = adj_res
            
    # If we didn't find any valid graph (unlikely), return empty or best effort
    if best_adj_overall is None:
        return []
        
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj_overall[u] >> v) & 1:
                edges.append([u, v])
                
    return edges

