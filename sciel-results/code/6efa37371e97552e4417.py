import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s safety margin

    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(all_edges)  # 190 edges

    # Build adjacency: adj[i][j] = True if edge (i,j) exists
    # We'll work with a bitset representation for speed
    # adj[u] is an integer where bit v is set if (u,v) is an edge

    # Initial graph: 4 disjoint K5s
    adj = [0] * n
    for i in range(4):
        base = i * 5
        for u in range(5):
            for v in range(u+1, 5):
                adj[base+u] |= (1 << (base+v))
                adj[base+v] |= (1 << (base+u))

    def edge_count():
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    def common_neighbors(u, v):
        """Return bitmask of common neighbors of u and v"""
        return adj[u] & adj[v]

    def is_valid_add(u, v):
        """Check if adding edge (u,v) creates a K3,3"""
        nu = adj[u]
        nv = adj[v]
        
        # Count common neighbors
        cn = nu & nv
        cn_count = bin(cn).count('1')
        
        # For a K3,3 to form with u-v as a cross edge:
        # We need 2 common neighbors of u and v (call them a1, a2)
        # and 2 vertices b1, b2 such that:
        #   b1, b2 are neighbors of both a1 and a2
        #   b1, b2 are NOT neighbors of u or v (or they are, but we need the full K3,3)
        # Actually, the K3,3 has parts {u, a1, a2} and {v, b1, b2}
        # Edges needed: u-v (being added), u-b1, u-b2, v-a1, v-a2, a1-b1, a1-b2, a2-b1, a2-b2
        # So b1, b2 must be common neighbors of a1 and a2, AND neighbors of u, AND not neighbors of v
        # Wait, b1, b2 are in the part with v, so they must be neighbors of v? No.
        # In K3,3 with parts A={u,a1,a2} and B={v,b1,b2}:
        # All edges between A and B must exist.
        # u-v: being added
        # u-b1, u-b2: b1,b2 must be neighbors of u
        # v-a1, v-a2: a1,a2 must be neighbors of v
        # a1-b1, a1-b2, a2-b1, a2-b2: b1,b2 must be common neighbors of a1,a2
        
        # So: a1, a2 in N(u) ∩ N(v) = common neighbors
        #     b1, b2 in N(u) ∩ N(a1) ∩ N(a2)
        #     But also b1, b2 should not be in A (i.e., not u, a1, a2)
        
        if cn_count < 2:
            return True
            
        # Get list of common neighbors
        cn_list = []
        for x in range(n):
            if cn & (1 << x):
                cn_list.append(x)
                
        for a1, a2 in itertools.combinations(cn_list, 2):
            # Find b's that are neighbors of u, a1, and a2
            common_ua1a2 = adj[u] & adj[a1] & adj[a2]
            # Exclude u, a1, a2 themselves
            common_ua1a2 &= ~(1 << u) & ~(1 << a1) & ~(1 << a2)
            b_count = bin(common_ua1a2).count('1')
            if b_count >= 2:
                return False
                
        return True

    def try_add(u, v):
        if not (adj[u] & (1 << v)):
            if is_valid_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                return True
        return False

    # Phase 1: Greedy add all possible edges
    for i in range(n):
        for j in range(i+1, n):
            try_add(i, j)

    best_edges = edge_count()
    best_adj = adj[:]

    # Phase 2: Branch and Bound for Maximum Independent Set on conflict hypergraph
    # A K3,3 violation is a set of 9 edges forming the K3,3
    # We want to find the maximum set of edges with no K3,3 subset
    
    # For branch and bound, we need to identify which edges conflict (can't both be in a K3,3-free graph)
    # Two edges conflict if they are part of some K3,3 together with other edges
    
    # Simpler approach: use local search with better moves
    # Phase 2: 2-swap local search with fast validity check
    
    def can_add_edge_fast(u, v):
        """Fast check: only need to verify if adding (u,v) creates a NEW K3,3"""
        nu = adj[u]
        nv = adj[v]
        cn = nu & nv
        cn_count = bin(cn).count('1')
        
        if cn_count < 2:
            return True
            
        cn_list = []
        for x in range(n):
            if cn & (1 << x):
                cn_list.append(x)
                
        for a1, a2 in itertools.combinations(cn_list, 2):
            common_ua1a2 = adj[u] & adj[a1] & adj[a2]
            common_ua1a2 &= ~(1 << u) & ~(1 << a1) & ~(1 << a2)
            b_count = bin(common_ua1a2).count('1')
            if b_count >= 2:
                return False
        return True

    random.seed(42)
    
    # Multiple restarts with perturbation
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Start from best or random perturbation
        if restart == 0:
            adj = best_adj[:]
        else:
            adj = best_adj[:]
            # Perturb: remove 5-10 random edges
            num_removals = random.randint(5, 10)
            current_edges_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        current_edges_list.append((i, j))
            for _ in range(min(num_removals, len(current_edges_list))):
                idx = random.randint(0, len(current_edges_list) - 1)
                u, v = current_edges_list[idx]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges_list[idx] = current_edges_list[-1]
                current_edges_list.pop()
        
        # Local search: try all single additions, then 2-swaps
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Single additions
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] & (1 << j)):
                        if can_add_edge_fast(i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            improved = True
            
            # 2-swaps: remove edge (i,j), add edge (k,l)
            current_edges_list = []
            non_edges_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        current_edges_list.append((i, j))
                    else:
                        non_edges_list.append((i, j))
                        
            if not current_edges_list or not non_edges_list:
                break
                
            # Try random 2-swaps
            for _ in range(min(200, len(current_edges_list) * len(non_edges_list) // 10)):
                if time.time() > deadline - 5:
                    break
                    
                rm_idx = random.randint(0, len(current_edges_list) - 1)
                add_idx = random.randint(0, len(non_edges_list) - 1)
                
                u1, v1 = current_edges_list[rm_idx]
                u2, v2 = non_edges_list[add_idx]
                
                # Remove (u1,v1)
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                # Try add (u2,v2)
                if can_add_edge_fast(u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    improved = True
                else:
                    # Revert removal
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
        
        # Update best
        cur_edges = edge_count()
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = adj[:]

    # Phase 3: Final greedy pass
    adj = best_adj[:]
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & (1 << j)):
                if can_add_edge_fast(i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

    cur_edges = edge_count()
    if cur_edges > best_edges:
        best_edges = cur_edges
        best_adj = adj[:]

    # Return edges from best_adj
    res = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                res.append([i, j])
    return res
