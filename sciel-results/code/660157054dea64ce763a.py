import random

def construct():
    n = 20
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(pairs)
    
    # Fixed seed for reproducibility
    random.seed(42)
    
    best_edges_count = 0
    best_graph = []
    
    # Number of restarts
    # 190 edges, check is fast. We can do many restarts.
    num_restarts = 2000
    
    for _ in range(num_restarts):
        # Create a random permutation of edges
        edges_order = pairs[:]
        random.shuffle(edges_order)
        
        adj = [0] * n
        edges = []
        
        for u, v in edges_order:
            # Check if adding (u, v) is valid
            # 1. Common neighbors of u and v must be <= 2
            common_uv = adj[u] & adj[v]
            if common_uv.bit_count() > 2:
                continue
            
            # 2. For each neighbor x of v, common neighbors of u and x must be <= 1
            #    (Because if |N(u) \cap N(x)| >= 2, say {a, b}, then {u, x} and {a, b, v?} 
            #     Wait, K2,3 is two vertices with 3 common neighbors.
            #     If we add (u,v), we need to ensure no pair of vertices has 3 common neighbors.
            #     Pairs involving u: (u, x) for x in N(u) U {v}.
            #     Pairs involving v: (v, y) for y in N(v) U {u}.
            #     Pairs not involving u or v are unchanged.
            
            valid = True
            
            # Check pairs (u, x) for x in N(v)
            # If we add (u,v), N(u) gains v, N(v) gains u.
            # For a neighbor x of v (x != u), the pair (u, x) now has common neighbors:
            # N(u) \cap N(x). Since v is in N(x) and we are adding u to N(v)?? No.
            # Let's re-evaluate the condition.
            
            # A graph is K2,3-free if for every pair of distinct vertices {a, b}, |N(a) \cap N(b)| <= 2.
            
            # When adding edge (u, v):
            # 1. The pair {u, v} itself: |N(u) \cap N(v)| must be <= 2.
            #    Current N(u) and N(v) do not include each other yet.
            #    So we check adj[u] & adj[v]. If > 2, invalid.
            
            # 2. Pairs {u, x} where x is a neighbor of v.
            #    If x is in N(v), then v is in N(x).
            #    After adding (u,v), u is in N(v) and v is in N(u).
            #    The common neighbors of u and x are N(u) \cap N(x).
            #    Note that v is in N(x) (since x in N(v) implies v in N(x)).
            #    Is v in N(u)? Yes, after adding.
            #    So v is a common neighbor of u and x?
            #    N(u) contains v. N(x) contains v. So v is in N(u) \cap N(x).
            #    Also, any other common neighbors?
            #    Before adding (u,v), N(u) did not contain v, and N(x) contained v.
            #    So v was NOT in N(u) \cap N(x) before.
            #    After adding, v IS in N(u) \cap N(x).
            #    So the new common neighbors are (Old N(u) \cap N(x)) \cup {v}.
            #    We need |New Common| <= 2.
            #    So |Old N(u) \cap N(x)| + 1 <= 2  => |Old N(u) \cap N(x)| <= 1.
            #    Old N(u) is adj[u]. Old N(x) is adj[x].
            #    So we need (adj[u] & adj[x]).bit_count() <= 1.
            
            mask_v = adj[v]
            while mask_v:
                lsb = mask_v & -mask_v
                x = lsb.bit_length() - 1
                mask_v ^= lsb
                # x is a neighbor of v
                # Check pair (u, x)
                if (adj[u] & adj[x]).bit_count() > 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # 3. Pairs {v, y} where y is a neighbor of u.
            #    Symmetric to above.
            #    Need (adj[v] & adj[y]).bit_count() <= 1.
            
            mask_u = adj[u]
            while mask_u:
                lsb = mask_u & -mask_u
                y = lsb.bit_length() - 1
                mask_u ^= lsb
                # y is a neighbor of u
                # Check pair (v, y)
                if (adj[v] & adj[y]).bit_count() > 1:
                    valid = False
                    break
            
            if valid:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        if len(edges) > best_edges_count:
            best_edges_count = len(edges)
            best_graph = edges[:]
            
            # Early termination if we hit a known high bound or max possible?
            # Max edges for K2,3-free on 20 vertices is around 40-50?
            # If we get 50, it's likely optimal or near.
            if best_edges_count >= 50:
                break
                
    return best_graph
