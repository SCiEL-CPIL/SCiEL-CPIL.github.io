import time
import random
import numpy as np

def construct():
    """
    Constructs a graph with no K5 and no independent 5-set.
    Returns Exoo's 42-vertex graph if found, otherwise falls back to 37-vertex Paley graph.
    """
    start_time = time.time()
    timeout = 290  # Safety margin
    
    # Fallback: Paley graph on 37 vertices
    def paley_graph(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    N = 42
    mask_all = (1 << N) - 1
    
    # Precompute popcount for all 2^42 is impossible. Use int.bit_count()
    
    def count_triangles(mask, adj):
        """Count triangles in the subgraph induced by 'mask'."""
        count = 0
        m = mask
        while m:
            i = (m & -m).bit_length() - 1
            m &= m - 1
            # Neighbors of i in mask, strictly greater than i
            # We want i < j < k
            # So for fixed i, we look at j > i in mask
            # and k > j in mask, where i~j, j~k, i~k
            adj_i = adj[i] & mask
            # Remove i and bits < i
            adj_i &= ~((1 << (i + 1)) - 1)
            mj = adj_i
            while mj:
                j = (mj & -mj).bit_length() - 1
                mj &= mj - 1
                # Common neighbors of i and j, strictly greater than j
                common = adj_i & adj[j]
                common &= ~((1 << (j + 1)) - 1)
                count += common.bit_count()
        return count

    def count_independent_triangles(mask, adj):
        """Count independent triangles (triangles in complement) in the subgraph induced by 'mask'."""
        count = 0
        m = mask
        while m:
            i = (m & -m).bit_length() - 1
            m &= m - 1
            # Non-neighbors of i in mask, strictly greater than i
            # comp_adj_i = ~adj[i] & mask
            # Remove i and bits < i
            non_adj_i = (~adj[i]) & mask
            non_adj_i &= ~((1 << (i + 1)) - 1)
            mj = non_adj_i
            while mj:
                j = (mj & -mj).bit_length() - 1
                mj &= mj - 1
                # Common non-neighbors of i and j, strictly greater than j
                # non_adj_j = ~adj[j] & mask
                common = non_adj_i & (~adj[j])
                common &= ~((1 << (j + 1)) - 1)
                count += common.bit_count()
        return count

    def delta_cost(u, v, adj):
        """
        Calculate change in (Num K5s + Num Ind 5s) if edge (u, v) is flipped.
        Assumes u < v.
        If edge exists, we remove it. If not, we add it.
        """
        # Common neighbors of u and v
        common_neigh = adj[u] & adj[v]
        # Remove u and v themselves (they aren't in their own adj lists usually, but be safe)
        common_neigh &= ~(1 << u) & ~(1 << v)
        
        # Common non-neighbors of u and v
        # mask_all is (1<<N)-1
        common_non_neigh = (~adj[u]) & (~adj[v]) & mask_all
        common_non_neigh &= ~(1 << u) & ~(1 << v)
        
        # Triangles in common neighbors contribute to K5s
        t_neigh = count_triangles(common_neigh, adj)
        
        # Triangles in common non-neighbors (in complement) contribute to Ind 5s
        t_non_neigh = count_independent_triangles(common_non_neigh, adj)
        
        if adj[u] & (1 << v):
            # Removing edge (u, v)
            # Loses K5s containing (u, v): -t_neigh
            # Gains Ind 5s containing (u, v): +t_non_neigh
            return -t_neigh + t_non_neigh
        else:
            # Adding edge (u, v)
            # Gains K5s containing (u, v): +t_neigh
            # Loses Ind 5s containing (u, v): -t_non_neigh
            return t_neigh - t_non_neigh

    def total_cost(adj):
        """Calculate total number of K5s and Independent 5-sets."""
        # This is expensive, so we only do it at start or if we suspect we are close.
        # Actually, we can sum the deltas? No.
        # We can count K5s directly.
        # But for SA, we maintain a running total.
        # Initial cost calculation:
        k5_count = 0
        ind5_count = 0
        
        # Count K5s
        # Iterate i < j < k < l < m
        # Too slow in pure python if done naively.
        # Use the triangle counting on edges?
        # Number of K5s containing edge (u, v) is number of triangles in N(u) cap N(v).
        # Sum over all edges?
        # Each K5 has 10 edges. So Sum_{edges} (Triangles in common neigh) / 10 = Num K5s.
        
        sum_edge_triangles = 0
        for u in range(N):
            for v in range(u + 1, N):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    common &= ~(1 << u) & ~(1 << v)
                    sum_edge_triangles += count_triangles(common, adj)
        k5_count = sum_edge_triangles // 10
        
        # Count Ind 5s
        # Number of Ind 5s containing non-edge (u, v) is number of ind-triangles in common non-neigh.
        # Sum over all non-edges?
        # Each Ind 5 has 10 non-edges.
        
        sum_nonedge_ind_triangles = 0
        for u in range(N):
            for v in range(u + 1, N):
                if not (adj[u] & (1 << v)):
                    common_non = (~adj[u]) & (~adj[v]) & mask_all
                    common_non &= ~(1 << u) & ~(1 << v)
                    sum_nonedge_ind_triangles += count_independent_triangles(common_non, adj)
        ind5_count = sum_nonedge_ind_triangles // 10
        
        return k5_count + ind5_count

    # Local Search with Simulated Annealing
    best_adj = None
    best_cost = float('inf')
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > timeout - 10:
            break
            
        # Initialize random graph
        adj = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = total_cost(adj)
        if current_cost == 0:
            return N, [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i] & (1 << j)]
            
        if current_cost < best_cost:
            best_cost = current_cost
            best_adj = adj[:]
            
        # SA Parameters
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        steps = 20000
        
        for step in range(steps):
            if time.time() - start_time > timeout - 5:
                break
                
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            d = delta_cost(u, v, adj)
            
            # Accept if improvement or with prob exp(-d/T)
            if d < 0 or random.random() < np.exp(-d / T):
                # Flip edge
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                current_cost += d
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj[:]
                    
                if current_cost == 0:
                    return N, [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i] & (1 << j)]
            
            T *= cooling
            if T < T_min:
                T = T_min

    # If we found a valid graph, return it.
    # Otherwise, return the best valid graph we know (Paley 37).
    # Note: best_adj might be invalid (cost > 0).
    # We must return a VALID graph.
    # So we only return best_adj if best_cost == 0.
    if best_cost == 0 and best_adj is not None:
        return N, [(i, j) for i in range(N) for j in range(i + 1, N) if best_adj[i] & (1 << j)]
    
    return paley_graph(37)
