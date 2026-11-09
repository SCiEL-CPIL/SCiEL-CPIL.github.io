import random
import time

def construct():
    n = 20
    
    start_time = time.time()
    deadline = start_time + 110
    
    def count_bits(x):
        return bin(x).count('1')
    
    def is_valid_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        if u > v:
            u, v = v, u
        if (adj[u] >> v) & 1:
            return True
        
        # The new edge (u,v) can only be in a K3,3 if u and v are in different parts.
        # Case 1: u in A-part, v in B-part
        # Need: 2 more A-vertices a1,a2 such that:
        #   - a1, a2 connect to v (they're in A, v in B)
        #   - u connects to a1, a2
        #   - There exist 2 more B-vertices b1,b2 such that:
        #     - b1, b2 connect to u, a1, a2
        #     - v connects to a1, a2 (already checked)
        
        nv = adj[v]  # neighbors of v
        nu = adj[u]  # neighbors of u
        
        # A-candidates: vertices connected to both u and v
        # Actually, a1 must connect to v (since a1 in A, v in B) and to u (since u in A)
        # So a1 in N(u) ∩ N(v)
        a_candidates = nu & nv
        # Remove u and v themselves
        a_candidates &= ~(1 << u)
        a_candidates &= ~(1 << v)
        
        # Convert to list
        a_list = []
        c = a_candidates
        while c:
            lsb = c & (-c)
            idx = lsb.bit_length() - 1
            a_list.append(idx)
            c ^= lsb
        
        if len(a_list) < 2:
            return True
        
        # For each pair (a1, a2) from a_list
        for i in range(len(a_list)):
            a1 = a_list[i]
            for j in range(i + 1, len(a_list)):
                a2 = a_list[j]
                # B-candidates: vertices connected to u, a1, a2, and v
                # b must connect to u, a1, a2 (since b in B, u,a1,a2 in A)
                # and v must connect to a1, a2 (already true since a1,a2 in N(v))
                b_candidates = nu & adj[a1] & adj[a2]
                # Remove v (v is already in B)
                b_candidates &= ~(1 << v)
                # Remove u, a1, a2 (they're in A)
                b_candidates &= ~(1 << u)
                b_candidates &= ~(1 << a1)
                b_candidates &= ~(1 << a2)
                
                if count_bits(b_candidates) >= 2:
                    return False
        
        # Case 2: v in A-part, u in B-part (symmetric)
        # By symmetry, this is the same check with u and v swapped
        # But since we already checked u in A, v in B, and the K3,3 condition is symmetric,
        # we don't need to check again. The edge (u,v) is undirected.
        
        return True
    
    def greedy_construct():
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj
    
    def count_edges(adj):
        total = 0
        for i in range(n):
            total += count_bits(adj[i])
        return total // 2
    
    def local_search(adj, time_limit):
        best_adj = adj[:]
        best_count = count_edges(adj)
        
        while time.time() < time_limit:
            improved = False
            
            # Try adding edges
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    if is_valid_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
            
            if improved:
                new_count = count_edges(adj)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
                continue
            
            # No more edges can be added
            break
        
        return best_adj, best_count
    
    # Run multiple rounds
    best_adj = [0] * n
    best_count = 0
    
    while time.time() < deadline:
        adj = greedy_construct()
        adj, count = local_search(adj, deadline - 2)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        if time.time() > deadline - 5:
            break
    
    # Convert to edge list
    edges = []
    for u in range(n):
        c = best_adj[u]
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            c ^= lsb
    
    return edges
