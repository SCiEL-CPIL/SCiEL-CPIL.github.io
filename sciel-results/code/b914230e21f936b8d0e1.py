import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s kill

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def popcount(x):
        return bin(x).count('1')

    def greedy_clique_cover_large(n, edges, adj, rng, deadline):
        """
        Greedy: repeatedly find a LARGE clique among remaining edges.
        Strategy: pick a vertex with high degree in remaining graph,
        expand to a large clique using max-degree-in-common heuristic.
        """
        remaining = set(edges)
        if not remaining:
            return []
        
        # Bitmask adjacency for remaining edges
        rem_adj = [0] * n
        for u, v in remaining:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        # Track degrees in remaining graph
        rem_deg = [popcount(rem_adj[i]) for i in range(n)]
        
        cliques = []
        
        while remaining:
            if time.time() > deadline:
                return None
            
            # Find vertex with highest degree in remaining graph
            # Sample if graph is large
            active_vertices = [v for v in range(n) if rem_deg[v] > 0]
            if not active_vertices:
                break
            
            if len(active_vertices) > 50:
                sample_size = min(50, len(active_vertices))
                sample = rng.sample(active_vertices, sample_size)
            else:
                sample = active_vertices
            
            best_vertex = max(sample, key=lambda v: rem_deg[v])
            
            # Start clique from best_vertex
            clique = [best_vertex]
            # Common neighbors of all vertices in clique (initially all neighbors of best_vertex)
            common = rem_adj[best_vertex]
            
            # Expand clique: greedily add vertex with max degree in common
            added = True
            while added:
                added = False
                if time.time() > deadline:
                    return None
                
                # Find candidate in common with max rem_deg
                best_w = -1
                best_w_deg = -1
                cm = common
                while cm:
                    w = (cm & -cm).bit_length() - 1
                    cm &= cm - 1
                    if w in clique:
                        continue
                    if rem_deg[w] > best_w_deg:
                        best_w_deg = rem_deg[w]
                        best_w = w
                
                if best_w < 0:
                    break
                
                clique.append(best_w)
                common &= rem_adj[best_w]
                added = True
            
            # Remove all edges within clique from remaining
            clique_set = set(clique)
            new_remaining = set()
            for u2, v2 in remaining:
                if u2 in clique_set and v2 in clique_set:
                    rem_adj[u2] &= ~(1 << v2)
                    rem_adj[v2] &= ~(1 << u2)
                    rem_deg[u2] -= 1
                    rem_deg[v2] -= 1
                else:
                    new_remaining.add((u2, v2))
            
            cliques.append(clique)
            remaining = new_remaining
        
        return cliques

    def greedy_clique_cover_min_common(n, edges, adj, rng, deadline):
        """
        Greedy: repeatedly find a clique starting from edge with min common neighbors.
        """
        remaining = set(edges)
        if not remaining:
            return []
        
        rem_adj = [0] * n
        for u, v in remaining:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        cliques = []
        
        while remaining:
            if time.time() > deadline:
                return None
            
            edge_list = list(remaining)
            if len(edge_list) > 100:
                sample_size = min(100, len(edge_list))
                sample = rng.sample(edge_list, sample_size)
            else:
                sample = edge_list
            
            best_edge = None
            best_common_mask = None
            best_count = float('inf')
            for u, v in sample:
                common = rem_adj[u] & rem_adj[v]
                count = popcount(common)
                if count < best_count:
                    best_count = count
                    best_edge = (u, v)
                    best_common_mask = common
            
            u, v = best_edge
            clique = [u, v]
            common = best_common_mask
            
            added = True
            while added:
                added = False
                if time.time() > deadline:
                    return None
                
                best_w = -1
                best_w_count = -1
                cm = common
                while cm:
                    w = (cm & -cm).bit_length() - 1
                    cm &= cm - 1
                    if w in clique:
                        continue
                    w_count = popcount(common & rem_adj[w])
                    if w_count > best_w_count:
                        best_w_count = w_count
                        best_w = w
                
                if best_w < 0:
                    break
                
                clique.append(best_w)
                common &= rem_adj[best_w]
                added = True
            
            clique_set = set(clique)
            new_remaining = set()
            for u2, v2 in remaining:
                if u2 in clique_set and v2 in clique_set:
                    rem_adj[u2] &= ~(1 << v2)
                    rem_adj[v2] &= ~(1 << u2)
                else:
                    new_remaining.add((u2, v2))
            
            cliques.append(clique)
            remaining = new_remaining
        
        return cliques

    def solve_instance(n, p, seed, deadline):
        adj, edges = gen_graph(n, p, seed)
        
        if not edges:
            return []
        
        best_cliques = [[u, v] for u, v in edges]
        best_count = len(best_cliques)
        
        rng_master = random.Random(12345)
        
        elapsed = time.time() - start_time
        remaining_time = deadline - start_time - elapsed
        if n <= 32:
            num_restarts = min(500, max(10, int(remaining_time / 0.03)))
        else:
            num_restarts = min(300, max(5, int(remaining_time / 0.08)))
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            rng = random.Random(rng_master.randint(0, 2**31))
            
            # Alternate between strategies
            if restart % 2 == 0:
                result = greedy_clique_cover_large(n, edges, adj, rng, deadline)
            else:
                result = greedy_clique_cover_min_common(n, edges, adj, rng, deadline)
            
            if result is not None:
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
        
        return best_cliques

    # Solve both instances
    result1 = solve_instance(32, 0.6, 601, DEADLINE)
    
    if time.time() > DEADLINE:
        adj2, edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, DEADLINE)
    
    return [result1, result2]
