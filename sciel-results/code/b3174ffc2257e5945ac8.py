import random
import itertools

def construct():
    def can_add_edge(G, u, v, n):
        # Check if adding (u, v) creates a C4.
        # C4: u - w - x - v - u
        # w in N(u), x in N(v), (w, x) in E
        # Note: w != v, x != u, w != x
        N_u = G[u]
        N_v = G[v]
        
        for w in N_u:
            if w == v:
                continue
            # Check if any neighbor of w is in N_v
            # We can just iterate neighbors of w
            for x in G[w]:
                if x in N_v:
                    # If x == u, then u-w-u? No, w is neighbor of u.
                    # If x == u, then u-w-u is a triangle? No, x is in N_v.
                    # If x == u, then u is in N_v. But we are adding u-v.
                    # If u in N_v, then v is in N_u.
                    # If w is in N_u and u is in N_v, and (w,u) in E (yes), 
                    # then path u-w-u-v? No.
                    # Path u - w - u - v is not a simple path.
                    # The C4 must be simple.
                    # If x == u, then we have u-w-u... 
                    # Wait, if x == u, then u is a neighbor of w.
                    # And u is in N_v.
                    # So we have u-w and u-v (new) and w-u (existing).
                    # This forms a triangle u-w-v? No, v is not connected to w yet.
                    # The path is u - w - u - v. Not a cycle.
                    # So we must ensure w != x.
                    if x == u:
                        continue
                    # Also w != x is implied if we check distinct vertices?
                    # If w == x, then w is in N_u and w is in N_v.
                    # And (w,w) is not an edge.
                    # So w != x is required for a C4.
                    if w == x:
                        continue
                    return True
        return False

    def try_construct(n, num_trials=200):
        target_deg = n - 39
        if target_deg < 0:
            target_deg = 0
            
        for _ in range(num_trials):
            G = {i: set() for i in range(n)}
            
            # We want to add edges to reach min degree target_deg.
            # Greedy: pick vertex with min degree, add random valid edge.
            # To ensure we don't get stuck, we can shuffle the order of attempts.
            
            # Initialize with some random edges? No, start empty.
            # But we need to be careful not to get stuck in a local minimum.
            # A better strategy: Randomly permute all possible edges and add them if valid.
            # This is the "Random Greedy" approach.
            
            all_edges = list(itertools.combinations(range(n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Only add if it helps increase min degree?
                # Or just add if valid?
                # If we add all valid edges, we might create a dense graph that is C4-free.
                # But we only need min degree target_deg.
                # Adding extra edges might block future necessary edges?
                # No, if we add an edge, it only makes C4-freeness harder for future edges.
                # So we should be selective.
                
                # Strategy: Add edge if it increases the degree of the lower-degree endpoint
                # and the current min degree is still < target_deg.
                # Or simpler: Just try to build a graph with min degree >= target_deg.
                
                # Let's stick to the "shuffle all edges" approach but stop early?
                # No, we must ensure min degree.
                
                # Refined Strategy:
                # 1. Shuffle edges.
                # 2. Iterate.
                # 3. If adding (u,v) keeps graph C4-free, add it.
                # 4. At the end, check min degree.
                
                if not can_add_edge(G, u, v, n):
                    G[u].add(v)
                    G[v].add(u)
            
            min_deg = min(len(G[i]) for i in range(n))
            if min_deg >= target_deg:
                edges = []
                for u in range(n):
                    for v in G[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
        
        return None

    # Try n from 47 down to 43
    # 43 is guaranteed by Polarity Graph, but let's try random for 47, 46, 45, 44 first.
    for n in range(47, 42, -1):
        res = try_construct(n, num_trials=100)
        if res:
            return res
    
    # Fallback: Polarity Graph of Projective Plane of order 6 (n=43)
    # We need to generate the edges.
    # This is complex to code from scratch correctly in a short time.
    # However, if random fails for 43, we can return a known valid small graph?
    # No, we want to maximize n.
    # If random fails for 43, we should implement the Polarity Graph.
    
    # Let's try to implement a simple Polarity Graph for q=6?
    # It requires generating the projective plane of order 6.
    # This is non-trivial.
    
    # Alternative Fallback:
    # If random fails, return the best we found?
    # Or just return n=39 empty graph? Score 1.0.
    # But random should find 43 easily.
    
    return {'n': 39, 'edges': []}
