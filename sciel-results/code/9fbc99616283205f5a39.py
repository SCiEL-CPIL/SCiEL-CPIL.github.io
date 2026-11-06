import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def _has_c4(adj, n):
        """Check if graph has a C4 using common neighbor count."""
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 1:
                    return True
        return False

    def _try_greedy(n, target_deg, seed, deadline):
        """Try to build C4-free graph on n vertices with min_degree >= target_deg."""
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        
        # Precompute all edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        def is_safe(u, v):
            """Check if adding edge (u,v) keeps graph C4-free."""
            # Check (u,v) common neighbors
            if len(adj[u] & adj[v]) > 1:
                return False
            
            # Check pairs (u, z) for z in N(v) where z != u
            for z in adj[v]:
                if z == u:
                    continue
                if len(adj[u] & adj[z]) >= 1:
                    return False
            
            # Check pairs (v, z) for z in N(u) where z != v
            for z in adj[u]:
                if z == v:
                    continue
                if len(adj[v] & adj[z]) >= 1:
                    return False
            
            return True
        
        # Multi-pass greedy: in each pass, find the best edge to add
        for pass_num in range(n * 3):
            if time.time() > deadline:
                break
            
            best_edge = None
            best_key = None
            
            for u, v in all_edges:
                if v in adj[u]:
                    continue
                
                deg_u = len(adj[u])
                deg_v = len(adj[v])
                
                # Only consider if it helps at least one under-degree vertex
                if deg_u >= target_deg and deg_v >= target_deg:
                    continue
                
                if not is_safe(u, v):
                    continue
                
                # Key: prioritize edges that help the most under-degree vertex
                key = (max(deg_u, deg_v), min(deg_u, deg_v))
                
                if best_key is None or key < best_key:
                    best_key = key
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            
            # Early exit if target reached
            if all(len(adj[i]) >= target_deg for i in range(n)):
                break
        
        # Final verification
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= target_deg:
            if not _has_c4(adj, n):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges
        return None

    def _paley_graph(n):
        """Construct Paley graph on n vertices (n must be prime congruent to 1 mod 4)."""
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                # Check if (v-u) is a quadratic residue mod n
                diff = (v - u) % n
                # Compute Legendre symbol
                ls = pow(diff, (n - 1) // 2, n)
                if ls == 1:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def _try_paley_subset(n_target, deadline):
        """
        Try to find a C4-free induced subgraph of Paley graph with high min degree.
        Strategy: Start with all vertices, iteratively remove vertices that
        minimize the impact on minimum degree.
        """
        if n_target > 47:
            return None
            
        # Use Paley graph on 47 vertices (47 is prime, 47 % 4 == 3, so not Paley)
        # 47 is not a Paley prime (needs n ≡ 1 mod 4)
        # Let's use the next suitable prime: 49 is not prime, 53 is prime and 53 % 4 == 1
        # But we need n_target <= 47. Let's just use the full Paley graph on 53 and subset.
        
        # Actually, let's try a different approach: use the known structure.
        # For n=47, target_deg = 8.
        # Let's try to construct a graph directly using greedy on a random initial set.
        
        # Alternative: Use the polarity graph approach from inspiration.
        # But let's stick to the directive: "Use a deterministic greedy algorithm on the vertices of the Paley graph P_47..."
        # Wait, P_47 is not a Paley graph because 47 ≡ 3 mod 4.
        # The directive says "Paley graph P_47", which is technically incorrect but let's assume it means
        # a graph on 47 vertices with similar properties, or we use a different prime.
        # Let's use prime 53 (53 ≡ 1 mod 4) and select 47 vertices.
        
        p = 53
        paley_adj = _paley_graph(p)
        
        # We want to select a subset S of size n_target from [0, p-1]
        # such that the induced subgraph has min degree >= n_target - 39.
        
        # Greedy selection:
        # Start with all vertices.
        # While len(S) > n_target:
        #   Find vertex v in S whose removal minimizes the drop in min_degree of S\{v}.
        #   Remove v.
        
        S = list(range(p))
        target_deg = n_target - 39
        
        while len(S) > n_target:
            if time.time() > deadline:
                return None
                
            best_v = None
            best_min_deg = -1
            
            # For each candidate v to remove, compute the min degree of the remaining induced subgraph
            # This is expensive. Let's use a heuristic: remove the vertex with the lowest degree in S.
            
            # Compute degrees in current induced subgraph
            degs = {}
            for u in S:
                degs[u] = len([v for v in S if v != u and v in paley_adj[u]])
            
            # Find vertex with minimum degree
            min_deg_val = min(degs.values())
            candidates = [u for u in S if degs[u] == min_deg_val]
            
            # If multiple, pick one randomly or deterministically
            v = candidates[0]
            
            S.remove(v)
        
        # Verify
        if len(S) != n_target:
            return None
            
        # Check min degree
        min_deg = 0
        for u in S:
            deg = len([v for v in S if v != u and v in paley_adj[u]])
            if deg < min_deg:
                min_deg = deg
        
        if min_deg < target_deg:
            return None
            
        # Check C4-free (Paley graphs are C4-free for prime p? No, they can have C4s.
        # Actually, Paley graphs are strongly regular and can contain C4s.
        # We need to verify.)
        
        # Build adj for S
        S_set = set(S)
        adj = [set() for _ in range(n_target)]
        S_to_idx = {v: i for i, v in enumerate(S)}
        
        for u in S:
            for v in paley_adj[u]:
                if v in S_set:
                    i = S_to_idx[u]
                    j = S_to_idx[v]
                    adj[i].add(j)
                    adj[j].add(i)
        
        if _has_c4(adj, n_target):
            return None
            
        edges = []
        for u in range(n_target):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    best_n = 39
    best_edges = []
    
    # Try from highest n down
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
        
        target_deg = max(1, n - 39)
        
        # Try Paley subset first (deterministic)
        if time.time() < deadline - 10:
            edges = _try_paley_subset(n, deadline)
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                break
        
        # Try multiple seeds for greedy
        max_seeds = 50 if n >= 46 else 100
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            edges = _try_greedy(n, target_deg, seed, deadline)
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                break  # Found valid for this n, try higher n
        
        if best_n >= n:
            # Already found something at this level or better
            if best_n > n:
                break
            # Found at this n, try higher
            continue
    
    if best_n <= 39:
        return {'n': 39, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
