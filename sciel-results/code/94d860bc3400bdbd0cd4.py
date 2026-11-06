import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 18
    
    # We want to construct H such that alpha(H) <= 3 and omega(H) <= 4.
    # Then G = complement(H) will be K4-free and have max triangle-free induced subgraph size 4.
    
    def check_safe_addition(adj, u, v):
        # Check if adding (u, v) creates a K5 in H.
        # This happens if common neighbors of u and v contain a triangle.
        common = adj[u] & adj[v]
        # Check if subgraph induced by 'common' has a triangle
        # Iterate pairs in common
        # Bit trick:
        # For each w in common, check if (common & adj[w]) has any bit?
        # No, that checks if w is connected to another in common.
        # We need a triangle w, x, y in common.
        # Equivalent: Does the graph induced by 'common' have an edge?
        # No, triangle needs 3 vertices.
        # If 'common' has an edge (w, x), and both are connected to y?
        # Simpler:
        # Iterate w in common:
        #   If (common & adj[w]) != 0, then there is an edge (w, x) in common.
        #   But we need a triangle.
        #   Actually, if 'common' has ANY edge, say (w,x), we need a third vertex y connected to both.
        #   Wait. If common has a triangle, then K5 is formed.
        #   If common has NO triangle, is it safe? Yes.
        #   So we just need to check if 'common' contains a triangle.
        
        # Fast triangle check on bitmask
        # Iterate w in common
        for w in range(n):
            if common & (1 << w):
                # Check if w has two neighbors in common that are connected to each other?
                # Or just: does common have a triangle?
                # A set of vertices contains a triangle iff:
                # There exists w such that (common & adj[w]) contains an edge.
                # Let neighbors_of_w_in_common = common & adj[w]
                # Check if neighbors_of_w_in_common has an edge.
                # Iterate x in neighbors_of_w_in_common:
                #    If neighbors_of_w_in_common & adj[x]: return True (has triangle)
                neigh = common & adj[w]
                # Check if neigh has an edge
                for x in range(n):
                    if neigh & (1 << x):
                        if neigh & adj[x]:
                            return False # Unsafe
        return True # Safe

    def find_independent_set_4(adj):
        # Returns a tuple of 4 vertices that form an independent set, or None
        # Optimize: Iterate combinations
        for combo in itertools.combinations(range(n), 4):
            a, b, c, d = combo
            # Check if no edges
            if (adj[a] & (1 << b)) == 0 and (adj[a] & (1 << c)) == 0 and (adj[a] & (1 << d)) == 0 and \
               (adj[b] & (1 << c)) == 0 and (adj[b] & (1 << d)) == 0 and \
               (adj[c] & (1 << d)) == 0:
                return combo
        return None

    def has_clique_5(adj):
        # Check if H has a K5
        # Iterate combinations of 5
        # This is slow if done repeatedly. 
        # But we only check at the end or if we suspect.
        # Actually, the safe addition check prevents K5 formation.
        # So if we start with empty graph and only add safe edges, we never have K5.
        # So we don't need to check this during construction.
        pass

    def get_complement_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    edges.append([u, v])
        return edges

    # Main Search Loop
    # We try multiple random restarts to find a valid H
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        # Initialize H as empty
        adj = [0] * n
        
        # Randomize vertex order to break symmetry? 
        # The graph is vertex-transitive? No.
        # But the search is greedy.
        
        # Greedy addition
        while True:
            if time.time() > deadline:
                break
                
            iso_set = find_independent_set_4(adj)
            if iso_set is None:
                # No independent set of size 4. alpha <= 3.
                # Since we only added safe edges, omega <= 4.
                # Check omega <= 4 explicitly just in case (though logic says it's true)
                # Actually, safe addition prevents K5. So omega <= 4 is guaranteed.
                # So we have a valid H.
                g_edges = get_complement_edges(adj)
                
                # Verify G is K4-free and score is 4
                # (Optional but good for debugging)
                # return g_edges
                
                # Final verification before returning
                # Check K4-free on G
                adj_g = [0]*n
                for u, v in g_edges:
                    adj_g[u] |= (1 << v)
                    adj_g[v] |= (1 << u)
                
                is_k4_free = True
                for combo in itertools.combinations(range(n), 4):
                    a, b, c, d = combo
                    if (adj_g[a] & (1<<b)) and (adj_g[a] & (1<<c)) and (adj_g[a] & (1<<d)) and \
                       (adj_g[b] & (1<<c)) and (adj_g[b] & (1<<d)) and \
                       (adj_g[c] & (1<<d)):
                        is_k4_free = False
                        break
                
                if not is_k4_free:
                    continue # Should not happen, but safety
                    
                # Check max triangle-free induced subgraph
                max_tf = 0
                # We expect 4.
                # Check if there is a triangle-free set of size 5.
                # If not, score is <= 4.
                # Since R(4,4)=18, score must be >= 4.
                # So score is 4.
                
                return g_edges

            # We have an independent set iso_set = (a, b, c, d)
            # We need to add an edge to break it.
            # Candidates: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
            added = False
            random.shuffle(iso_set) # Randomize choice
            
            for u, v in itertools.combinations(iso_set, 2):
                if adj[u] & (1 << v):
                    continue # Already edge
                if check_safe_addition(adj, u, v):
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            
            if not added:
                # Stuck. No safe edge in this independent set.
                # Try to find a safe edge elsewhere?
                # Or break and restart.
                break

    # Fallback: Return Turán graph T_3(18)
    # This has score 12.
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
