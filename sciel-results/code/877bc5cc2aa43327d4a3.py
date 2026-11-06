import time
import random
import sys

def construct():
    deadline = time.time() + 110
    n = 40
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_pairs)
    
    # Fast C6 check using bitset operations
    def has_c6_fast(adj, u, v):
        """Check if adding edge (u,v) creates a C6.
        A C6 through (u,v) has form: u-a-b-c-d-v with all vertices distinct.
        We check: exists a in N(u), d in N(v), b in N(a), c in N(d) 
        with b-c an edge, all 6 vertices distinct.
        """
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        # For each a in N(u) \ {v}
        a_iter = nu
        while a_iter:
            a_bit = a_iter & -a_iter
            a = a_bit.bit_length() - 1
            a_iter ^= a_bit
            
            # For each d in N(v) \ {u, a}
            d_iter = nv & ~(1 << a)
            while d_iter:
                d_bit = d_iter & -d_iter
                d = d_bit.bit_length() - 1
                d_iter ^= d_bit
                
                # b in N(a) \ {u, v, d}, c in N(d) \ {u, v, a}
                # Need b-c to be an edge
                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << d)
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # Check if any b in b_mask is connected to any c in c_mask
                # Use bitset intersection trick
                b_iter = b_mask
                while b_iter:
                    b_bit = b_iter & -b_iter
                    b = b_bit.bit_length() - 1
                    b_iter ^= b_bit
                    if (adj[b] & c_mask) != 0:
                        return True
        
        return False
    
    def get_edges(adj):
        """Get list of edges from adjacency bitsets."""
        edges = []
        for u in range(n):
            v_iter = adj[u] & (~((1 << (u+1)) - 1))
            while v_iter:
                v_bit = v_iter & -v_iter
                v = v_bit.bit_length() - 1
                v_iter ^= v_bit
                edges.append((u, v))
        return edges
    
    def greedy_fill(adj, deadline):
        """Greedily add edges in random order until no more can be added."""
        added = 0
        # Try multiple random orderings
        while time.time() < deadline:
            order = all_pairs[:]
            random.shuffle(order)
            progress = False
            for u, v in order:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if not has_c6_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added += 1
                    progress = True
            if not progress:
                break
        return added
    
    def local_search(adj, deadline):
        """Local search: try to add edges, and if stuck, remove one and try again."""
        edge_count = sum(bin(x).count('1') for x in adj) // 2
        best_adj = adj[:]
        best_count = edge_count
        no_improve = 0
        
        while time.time() < deadline:
            # Try to add any edge
            improved = False
            order = all_pairs[:]
            random.shuffle(order)
            for u, v in order:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if not has_c6_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1
                    improved = True
                    if edge_count > best_count:
                        best_count = edge_count
                        best_adj = adj[:]
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 5:
                    break
            
            # If we couldn't add, try removing a random edge and see if we can add more
            if time.time() > deadline - 2:
                break
            
            edges = get_edges(adj)
            if not edges:
                break
            
            ru, rv = random.choice(edges)
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
            edge_count -= 1
            
            # Try to add edges after removal
            added_after_removal = 0
            order2 = all_pairs[:]
            random.shuffle(order2)
            for u, v in order2:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if not has_c6_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1
                    added_after_removal += 1
            
            if edge_count > best_count:
                best_count = edge_count
                best_adj = adj[:]
        
        return best_adj, best_count
    
    # Start with 8 disjoint K5s as base
    base_adj = [0] * n
    for k in range(8):
        start = k * 5
        for a in range(5):
            for b in range(a + 1, 5):
                u = start + a
                v = start + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
    
    best_adj = base_adj[:]
    best_count = sum(bin(x).count('1') for x in best_adj) // 2
    
    random.seed(42)
    
    # Phase 1: Multiple greedy builds from base
    while time.time() < deadline - 30:
        adj = base_adj[:]
        greedy_fill(adj, deadline - 30)
        current_count = sum(bin(x).count('1') for x in adj) // 2
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
    
    # Phase 2: Local search from best
    if time.time() < deadline - 5:
        best_adj, best_count = local_search(best_adj, deadline - 2)
    
    # Phase 3: Final greedy polish
    if time.time() < deadline - 1:
        adj = best_adj[:]
        greedy_fill(adj, deadline - 0.5)
        current_count = sum(bin(x).count('1') for x in adj) // 2
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
    
    # Convert to edge list
    edges = []
    for u in range(n):
        v_iter = best_adj[u] & (~((1 << (u+1)) - 1))
        while v_iter:
            v_bit = v_iter & -v_iter
            v = v_bit.bit_length() - 1
            v_iter ^= v_bit
            edges.append([u, v])
    
    return edges
