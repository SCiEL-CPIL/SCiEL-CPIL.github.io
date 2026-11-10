import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 90.0  # Safety margin before 120s kill
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = 0
    best_adj = None
    
    def check_valid(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        S = adj[u] & adj[v]
        # Extract bits of S
        bits = []
        mask = S
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            bits.append(idx)
            mask ^= lsb
            
        # Check Case 1: u in A, v in B
        for i in range(len(bits)):
            a = bits[i]
            for j in range(i + 1, len(bits)):
                b = bits[j]
                # Common neighbors of u, a, b excluding v
                common = (adj[u] & adj[a] & adj[b]) & ~(1 << v)
                if common.bit_count() >= 2:
                    return False
                    
        # Check Case 2: v in A, u in B
        for i in range(len(bits)):
            a = bits[i]
            for j in range(i + 1, len(bits)):
                b = bits[j]
                # Common neighbors of v, a, b excluding u
                common = (adj[v] & adj[a] & adj[b]) & ~(1 << u)
                if common.bit_count() >= 2:
                    return False
                    
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def greedy_build(adj, edge_order):
        for u, v in edge_order:
            if check_valid(adj, u, v):
                add_edge(adj, u, v)
        return adj

    def local_search(adj):
        """
        1. Greedy add missing.
        2. 5 kicks: remove random edge, greedy add missing.
        """
        current_adj = adj[:]
        current_count = count_edges(current_adj)
        
        # Phase 1: Greedy add
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not ((current_adj[i] >> j) & 1):
                    missing.append((i, j))
        random.shuffle(missing)
        for u, v in missing:
            if check_valid(current_adj, u, v):
                add_edge(current_adj, u, v)
                current_count += 1
                
        # Phase 2: Kicks
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
                
            candidate_adj = current_adj[:]
            candidate_count = current_count
            
            # Get current edges
            cur_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (candidate_adj[i] >> j) & 1:
                        cur_edges.append((i, j))
                        
            if not cur_edges:
                break
                
            u, v = random.choice(cur_edges)
            remove_edge(candidate_adj, u, v)
            candidate_count -= 1
            
            # Greedy add missing to candidate
            missing_c = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((candidate_adj[i] >> j) & 1):
                        missing_c.append((i, j))
            random.shuffle(missing_c)
            for eu, ev in missing_c:
                if check_valid(candidate_adj, eu, ev):
                    add_edge(candidate_adj, eu, ev)
                    candidate_count += 1
                    
            if candidate_count > current_count:
                current_adj = candidate_adj
                current_count = candidate_count
                
        return current_adj, current_count

    # Multiple greedy restarts
    while time.time() < deadline - 5.0:
        edge_order = all_edges[:]
        random.shuffle(edge_order)
        
        adj = [0] * n
        greedy_build(adj, edge_order)
        
        count = count_edges(adj)
        
        if time.time() < deadline - 10.0:
            adj, count = local_search(adj)
        
        if count > best_edges:
            best_edges = count
            best_adj = adj[:]
            
    # Reconstruct edge list
    if best_adj is None:
        return []
    
    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                res.append([i, j])
    
    return res
