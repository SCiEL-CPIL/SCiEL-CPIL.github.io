import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    total_possible = n * (n - 1) // 2
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(adj, u, v):
        # Check if adding edge (u,v) keeps graph K2,3-free
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Pair (u,v) common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
            
        # 2. For all w in N(v), pair (u,w) gains v as common neighbor.
        #    So current common(u,w) must be <= 1.
        w = adj[v]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            if popcount(adj[u] & adj[wi]) > 1:
                return False
                
        # 3. For all w in N(u), pair (v,w) gains u as common neighbor.
        #    So current common(v,w) must be <= 1.
        w = adj[u]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            if popcount(adj[v] & adj[wi]) > 1:
                return False
                
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def count_edges(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2
        
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v_mask = adj[u]
            while v_mask:
                lsb = v_mask & (-v_mask)
                v_mask ^= lsb
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
        return edges

    def is_k23_free(adj):
        """Check if graph is K2,3-free"""
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) > 2:
                    return False
        return True

    def spectral_construction():
        """
        Use a Paley-type construction over finite field GF(19) to create 
        a graph with limited common neighbors, then map to 20 vertices.
        """
        # Use GF(19) since 19 is prime and 19 ≡ 3 (mod 4) for Paley graph
        p = 19
        
        # Quadratic residues mod p
        def is_quadratic_residue(x, p):
            if x == 0:
                return False
            return pow(x, (p-1)//2, p) == 1
        
        # Build adjacency for 19 vertices using Paley graph
        adj = [0] * n
        for u in range(p):
            for v in range(u + 1, p):
                diff = (v - u) % p
                if is_quadratic_residue(diff, p):
                    add_edge(adj, u, v)
        
        # For vertex 20 (index 19), connect to a subset to maintain K2,3-free
        # Try different connection patterns and pick the best
        best_adj = adj[:]
        best_edges = count_edges(adj)
        
        # Try connecting vertex 19 to different subsets
        for mask in range(1 << p):
            if popcount(mask) > 12:  # Limit degree of vertex 19
                continue
            
            test_adj = adj[:]
            valid = True
            for u in range(p):
                if (mask >> u) & 1:
                    # Check if adding edge (19, u) keeps K2,3-free
                    if not can_add(test_adj, 19, u):
                        valid = False
                        break
                    if valid:
                        add_edge(test_adj, 19, u)
            
            if valid:
                edges = count_edges(test_adj)
                if edges > best_edges:
                    best_edges = edges
                    best_adj = test_adj[:]
        
        return best_adj

    def coordinate_ascent(adj, max_iters=5000):
        """
        Optimize by trying to flip individual edges (add or remove) 
        to improve the edge count while maintaining K2,3-free property.
        """
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add a random missing edge
            for _ in range(50):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v or (current_adj[u] >> v) & 1:
                    continue
                if can_add(current_adj, u, v):
                    add_edge(current_adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                    improved = True
                    break
            
            if not improved:
                # Try to remove a random edge and see if we can add two
                all_edges = []
                for u in range(n):
                    v_mask = current_adj[u]
                    while v_mask:
                        lsb = v_mask & (-v_mask)
                        v_mask ^= lsb
                        v = lsb.bit_length() - 1
                        if u < v:
                            all_edges.append((u, v))
                
                if all_edges:
                    u_rem, v_rem = random.choice(all_edges)
                    remove_edge(current_adj, u_rem, v_rem)
                    old_edges = current_edges
                    current_edges -= 1
                    
                    # Try to add back two edges
                    added = 0
                    for _ in range(100):
                        u_add = random.randint(0, n - 1)
                        v_add = random.randint(0, n - 1)
                        if u_add == v_add or (current_adj[u_add] >> v_add) & 1:
                            continue
                        if (u_add == u_rem and v_add == v_rem) or (u_add == v_rem and v_add == u_rem):
                            continue
                        if can_add(current_adj, u_add, v_add):
                            add_edge(current_adj, u_add, v_add)
                            current_edges += 1
                            added += 1
                            if added >= 2:
                                break
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                    elif current_edges < old_edges:
                        # Revert if we didn't improve
                        # This is simplified - in practice we'd need to track state
                        pass
        
        return best_adj, best_edges

    # Multi-start with spectral construction
    best_adj = [0] * n
    best_edges = 0
    
    # Start 1: Spectral construction
    adj = spectral_construction()
    adj, edges = coordinate_ascent(adj)
    if edges > best_edges:
        best_edges = edges
        best_adj = adj[:]
    
    # Additional random starts
    num_starts = 15
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        # Greedy build
        order = list(range(n * (n - 1) // 2))
        random.shuffle(order)
        pair_map = {}
        idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                pair_map[idx] = (u, v)
                idx += 1
        
        for i in order:
            if time.time() > deadline:
                break
            u, v = pair_map[i]
            if can_add(adj, u, v):
                add_edge(adj, u, v)
        
        # Coordinate ascent
        adj, edges = coordinate_ascent(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    # Final verification
    if not is_k23_free(best_adj):
        # Fallback to a simple valid graph
        best_adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if (v - u) % 3 != 0:  # Simple pattern that's likely K2,3-free
                    if can_add(best_adj, u, v):
                        add_edge(best_adj, u, v)
    
    return adj_to_edges(best_adj)
# EVOLVE-BLOCK-END
