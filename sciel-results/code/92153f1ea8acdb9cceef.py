import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin before 120s/150s kill

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    def greedy_partition_bitmask(n, edges, adj, rng, deadline):
        """
        Performs a single greedy clique partition using bitmasks.
        Returns list of cliques (lists of vertices).
        """
        # Initialize remaining adjacency
        rem_adj = adj[:]
        remaining_edges_count = len(edges)
        cliques = []
        
        # Precompute list of vertices for faster iteration if needed, but bitmask is key
        vertices = list(range(n))
        
        while remaining_edges_count > 0:
            if time.time() > deadline:
                # Fallback: add remaining edges as individual cliques
                # This ensures validity if we run out of time mid-partition
                # We need to reconstruct remaining edges from rem_adj
                for u in range(n):
                    mask = rem_adj[u] & ((1 << u) - 1) # Only look at lower vertices to avoid double count
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        cliques.append([u, v])
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
                        mask ^= lsb
                return cliques

            # 1. Find vertex with max degree in remaining graph
            max_deg = -1
            start_u = -1
            # To add diversity, we can randomly perturb the choice among top-degree vertices
            # or just pick max. Let's pick max, but use rng for tie-breaking.
            
            # Optimization: Instead of scanning all n every time, we could maintain a heap,
            # but n is small (32, 64), so O(n) scan is fine.
            for u in vertices:
                d = popcount(rem_adj[u])
                if d > max_deg:
                    max_deg = d
                    start_u = u
                elif d == max_deg and d > 0:
                    # Random tie-breaking
                    if rng.random() < 0.5:
                        start_u = u

            if start_u == -1 or max_deg == 0:
                break

            # 2. Build clique greedily starting from start_u
            clique = [start_u]
            # Candidates P: neighbors of start_u in remaining graph
            P = rem_adj[start_u]
            
            while P:
                # Pick v in P with max degree in P
                max_v_deg = -1
                best_v = -1
                
                # Iterate through set bits in P
                P_temp = P
                while P_temp:
                    lsb = P_temp & (-P_temp)
                    v = lsb.bit_length() - 1
                    P_temp ^= lsb
                    
                    # Degree of v in P
                    deg_in_P = popcount(rem_adj[v] & P)
                    
                    if deg_in_P > max_v_deg:
                        max_v_deg = deg_in_P
                        best_v = v
                    elif deg_in_P == max_v_deg and deg_in_P > 0:
                        # Random tie-breaking for diversity
                        if rng.random() < 0.5:
                            best_v = v
                            
                if best_v == -1:
                    break
                    
                clique.append(best_v)
                # Update P: intersect with neighbors of best_v
                P &= rem_adj[best_v]
            
            # 3. Remove edges of clique from rem_adj
            edges_removed = 0
            for i in range(len(clique)):
                u_c = clique[i]
                # Only remove edges to subsequent vertices to avoid double counting/removal
                for j in range(i + 1, len(clique)):
                    v_c = clique[j]
                    if (rem_adj[u_c] >> v_c) & 1:
                        rem_adj[u_c] &= ~(1 << v_c)
                        rem_adj[v_c] &= ~(1 << u_c)
                        edges_removed += 1
            
            if edges_removed == 0:
                # Safety break to avoid infinite loop
                break
                
            cliques.append(clique)
            remaining_edges_count -= edges_removed

        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        if not edges:
            return []
            
        adj = build_adj(n, edges)
        
        # Baseline: each edge as its own clique
        baseline_cliques = [[u, v] for u, v in edges]
        best_cliques = baseline_cliques
        best_count = len(baseline_cliques)
        
        start = time.time()
        attempt = 0
        end_time = start + time_limit
        
        # Run many fast greedy restarts
        while time.time() < end_time:
            # Use different seeds for RNG to get diverse partitions
            rng = random.Random(seed * 1000 + attempt)
            
            # We pass a deadline slightly before the actual limit to allow for return processing
            # But greedy_partition_bitmask checks its own deadline for fallback
            result = greedy_partition_bitmask(n, edges, adj, rng, end_time)
            
            if result is not None:
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
            
            attempt += 1
            
        return best_cliques

    # Solve both instances
    # Allocate time based on problem size
    # n=32 is much faster, so it can afford more restarts or we just split time.
    # Let's give n=32 40s and n=64 70s. Total 110s.
    time_budget_1 = 40.0
    time_budget_2 = 70.0
    
    # Ensure we don't exceed total deadline
    current_time = time.time()
    remaining = DEADLINE - current_time
    if time_budget_1 + time_budget_2 > remaining:
        scale = remaining / (time_budget_1 + time_budget_2)
        time_budget_1 *= scale
        time_budget_2 *= scale

    result1 = solve_instance(32, 0.6, 601, time_budget_1)
    
    if time.time() > DEADLINE:
        # Fallback: return baseline for second
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        # Adjust budget for second instance based on actual remaining time
        actual_budget_2 = DEADLINE - time.time()
        result2 = solve_instance(64, 0.55, 602, actual_budget_2)
    
    return [result1, result2]
