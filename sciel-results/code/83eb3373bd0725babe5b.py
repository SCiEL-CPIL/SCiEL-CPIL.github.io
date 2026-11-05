import random
import numpy as np

def construct():
    n = 40
    target_edges = 401
    
    # Bitmask adjacency
    # adj[i] is an integer where bit j is 1 if (i,j) is an edge
    
    def get_common_count(adj, u, v):
        # Count common neighbors of u and v
        # adj[u] & adj[v] gives bitmask of common neighbors
        # Need to exclude u and v themselves? 
        # In simple graph, u is not neighbor of u.
        # adj[u] does not have bit u set.
        # So adj[u] & adj[v] contains only w != u, v.
        return bin(adj[u] & adj[v]).count('1')

    def build_graph(max_book_limit, seed):
        random.seed(seed)
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        adj = [0] * n
        edge_count = 0
        edges_list = []
        
        for u, v in all_edges:
            if edge_count == target_edges:
                break
            
            # Check if adding (u,v) violates max book limit
            # Current common neighbors
            common = bin(adj[u] & adj[v]).count('1')
            
            # If we add (u,v), the book size of (u,v) becomes 'common'.
            # Also, adding (u,v) increases the book size of existing edges (u,w) and (v,w)
            # where w is a common neighbor.
            # For each w in common_neighbors(u,v):
            #   book(u,w) increases by 1 (because v becomes a common neighbor of u and w? No.)
            #   Let's check: Edge (u,w). Common neighbors are N(u) \cap N(w).
            #   Before adding (u,v): v is not in N(u).
            #   After adding (u,v): v is in N(u).
            #   Is v in N(w)?
            #   w is a common neighbor of u and v, so (v,w) is an edge. Yes, v is in N(w).
            #   So v becomes a new common neighbor of (u,w).
            #   So book(u,w) increases by 1.
            #   Similarly book(v,w) increases by 1.
            
            # We need to ensure that for all existing edges (u,w) and (v,w) where w is common neighbor,
            # their new book size does not exceed max_book_limit.
            
            # Let's identify the common neighbors
            common_mask = adj[u] & adj[v]
            
            # Check if any existing edge involving u or v and a common neighbor would exceed the limit
            # Edges (u,w) for w in common_mask
            # Current book size of (u,w) is popcount(adj[u] & adj[w])
            # New book size will be current + 1
            # We need current + 1 <= max_book_limit => current <= max_book_limit - 1
            
            # Similarly for (v,w)
            
            # If max_book_limit is small, this check is cheap.
            
            valid = True
            # Iterate through bits in common_mask
            m = common_mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                w = lsb.bit_length() - 1
                m ^= lsb
                
                # Check edge (u, w)
                # Is (u,w) an edge? Yes, w is neighbor of u.
                # Current book size of (u,w)
                bw_uw = bin(adj[u] & adj[w]).count('1')
                if bw_uw + 1 > max_book_limit:
                    valid = False
                    break
                
                # Check edge (v, w)
                bw_vw = bin(adj[v] & adj[w]).count('1')
                if bw_vw + 1 > max_book_limit:
                    valid = False
                    break
            
            if valid:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges_list.append((u, v))
                edge_count += 1
        
        return edges_list if edge_count == target_edges else None

    # Try to build with max_book_limit = 7
    # Theorem says min is 7.
    # If we can't find one, try 8, etc.
    
    best_edges = None
    for limit in range(7, 15):
        for seed in range(100): # Try 100 random orders
            edges = build_graph(limit, seed)
            if edges:
                # Verify max book size
                adj = [0] * n
                for u, v in edges:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                max_book = 0
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u] & (1 << v):
                            b = bin(adj[u] & adj[v]).count('1')
                            if b > max_book:
                                max_book = b
                            
                # If max_book <= limit, it's a valid solution
                # We want the smallest max_book
                if max_book <= limit:
                    # Since we iterate limit from 7 up, the first one found with limit=7 is optimal
                    # But we should check if we can get lower? No, lower bound is 7.
                    # So if we find one with limit 7, we are done.
                    return [list(e) for e in edges]
                    
        # If no solution found for this limit, try next
    
    # Fallback: Return the best found or random
    # This should not happen if limit is high enough
    # Let's just return a random graph if all fails (should be very unlikely)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    return [list(e) for e in all_edges[:target_edges]]
