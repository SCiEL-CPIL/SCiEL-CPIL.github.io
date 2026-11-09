import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def count_edges(bs, adj_list):
        """Count edges in the subgraph induced by bitset bs."""
        count = 0
        temp = bs
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            count += (adj_list[v] & bs).bit_count()
            temp -= lsb
        return count // 2

    def try_solve(n):
        """
        Attempt to find a valid graph on n vertices using Simulated Annealing.
        Returns (n, edges) if found, else None.
        """
        if n < 1:
            return None
        
        # Initialize
        adj = [0] * n
        mask = (1 << n) - 1
        comp_adj = [0] * n
        
        # Start with random graph p=0.5
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Compute complement
        for i in range(n):
            comp_adj[i] = mask ^ adj[i] ^ (1 << i)
            
        # Initial Cost
        k4_count = 0
        # Count K4s: sum over edges (u,v) of edges in G[N(u) & N(v)] / 6
        # Or sum over vertices of triangles in G[N(u)] / 4
        # Let's use the edge method for consistency with delta updates
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    w = adj[u] & adj[v]
                    k4_count += count_edges(w, adj)
        k4_count //= 6
        
        # Count Indep6s: sum over edges in comp_adj (u,v) of edges in CompG[N_comp(u) & N_comp(v)] / 6
        # Wait, Indep6 in G is K6 in CompG.
        # Number of K6s in CompG.
        # Sum over edges (u,v) in CompG of edges in CompG[N_comp(u) & N_comp(v)] / 6
        indep6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if comp_adj[u] & (1 << v):
                    w = comp_adj[u] & comp_adj[v]
                    indep6_count += count_edges(w, comp_adj)
        indep6_count //= 6
        
        current_cost = k4_count + indep6_count
        best_cost = current_cost
        best_adj = adj[:]
        
        if current_cost == 0:
            return n, adj_to_edges(adj)
            
        T = 1000.0
        cooling = 0.9999
        min_T = 1e-6
        
        steps = 0
        while time.time() < deadline:
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Ensure u < v for consistency, though bitsets handle it
            if u > v:
                u, v = v, u
                
            is_edge = bool(adj[u] & (1 << v))
            
            # Calculate Delta
            # W = N(u) & N(v) in G
            w_g = adj[u] & adj[v]
            # Z = N_comp(u) & N_comp(v) in G (which is N(u) & N(v) in CompG)
            z_g = comp_adj[u] & comp_adj[v]
            
            # Note: w_g and z_g do not include u or v because adj[u] doesn't have u.
            # However, when we flip (u,v), N(u) and N(v) change.
            # The formula for delta assumes we count K4s *containing* (u,v).
            # If we ADD (u,v):
            #   New K4s = Edges in G[W].
            #   New Indep6s = - Edges in CompG[Z] (which are Indep4 in G[Z]).
            #   Wait, Indep6 containing (u,v) in G requires (u,v) to be non-adjacent in G.
            #   So if we ADD (u,v), we DESTROY all Indep6s that contain (u,v).
            #   The number of such Indep6s is the number of K6s in CompG containing (u,v).
            #   K6 in CompG containing (u,v) corresponds to edges in CompG[N_comp(u) & N_comp(v)].
            #   So Delta Indep6 = - count_edges(z_g, comp_adj).
            
            # If we REMOVE (u,v):
            #   Remove K4s containing (u,v) = - count_edges(w_g, adj).
            #   Add Indep6s containing (u,v) = + count_edges(z_g, comp_adj).
            
            if is_edge:
                # Removing
                delta_k4 = -count_edges(w_g, adj)
                delta_indep = count_edges(z_g, comp_adj)
            else:
                # Adding
                delta_k4 = count_edges(w_g, adj)
                delta_indep = -count_edges(z_g, comp_adj)
                
            delta_cost = delta_k4 + delta_indep
            
            # Acceptance
            if delta_cost < 0 or random.random() < pow(2.718281828, -delta_cost / T):
                # Apply flip
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                comp_adj[u] ^= (1 << v)
                comp_adj[v] ^= (1 << u)
                
                current_cost += delta_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj[:]
                    if best_cost == 0:
                        return n, adj_to_edges(adj)
                
            T *= cooling
            if T < min_T:
                T = min_T
                
            steps += 1
            if steps % 10000 == 0 and steps > 0:
                 # Optional: restart if stuck?
                 pass
        
        return None

    def adj_to_edges(adj):
        edges = []
        n = len(adj)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Try n=35 first
    for n in [35, 30, 25, 20, 17]:
        if time.time() > deadline:
            break
        res = try_solve(n)
        if res:
            n_res, edges_res = res
            return {'n': n_res, 'edges': edges_res}
            
    # Fallback: Return the Paley graph on 17
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
