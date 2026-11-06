import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    n = 40
    m_target = 401
    
    def compute_max_book(adj):
        max_book = 0
        for i in range(n):
            mask_i = adj[i]
            for j in range(i + 1, n):
                if (mask_i >> j) & 1:
                    common = (adj[i] & adj[j]).bit_count()
                    if common > max_book:
                        max_book = common
        return max_book

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            mask &= ~((1 << (i+1)) - 1) 
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                edges.append([i, j])
                mask ^= lsb
        return edges

    def edge_count(adj):
        count = 0
        for i in range(n):
            mask = adj[i]
            mask &= ~((1 << (i+1)) - 1)
            count += mask.bit_count()
        return count

    def make_k20_20_plus_cliques():
        """Create K_{20,20} + two K_{10} cliques.
        
        Partition: A = {0..9}, B = {10..19}, C = {20..29}, D = {30..39}
        Edges:
        - All cross edges between {A,B} and {C,D}: 20*20 = 400 edges
        - All edges within A (K_10): C(10,2) = 45 edges
        - All edges within D (K_10): C(10,2) = 45 edges
        Total: 490 edges. We need 401, so we'll remove 89 edges.
        
        Actually, let's reconsider. The directive says to start with this structure
        and then use local search. But 490 > 401. 
        
        Let me re-read: "partitioning each side into two groups of 10 and adding 
        all edges within each group (creating two disjoint K_{10}s)"
        
        So the base is K_{20,20} (400 edges) + two K_{10}s (90 edges) = 490 edges.
        We need to remove 89 edges to get to 401.
        
        Strategy: Remove edges from the cross-partition to balance degrees and
        reduce common neighbors.
        """
        adj = [0] * n
        
        # Add K_{20,20}: all edges between {0..19} and {20..39}
        for i in range(20):
            for j in range(20, 40):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # Add K_{10} on {0..9}
        for i in range(10):
            for j in range(i+1, 10):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # Add K_{10} on {30..39}
        for i in range(30, 40):
            for j in range(i+1, 40):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # Now we have 490 edges. Need to remove 89 to get 401.
        # Remove random cross-edges to balance.
        # The cross-edges are between {0..19} and {20..39}.
        # There are 400 cross-edges. Remove 89 of them.
        
        cross_edges = []
        for i in range(20):
            for j in range(20, 40):
                cross_edges.append((i, j))
        
        random.shuffle(cross_edges)
        to_remove = cross_edges[:89]
        for u, v in to_remove:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        return adj

    best_adj = None
    best_score = float('inf')
    
    # Simulated Annealing parameters
    T_start = 2.0
    T_end = 0.01
    
    for seed in range(30):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed + 200)
        
        # Initialize with K_{20,20} + two K_{10} cliques, then remove 89 cross-edges
        adj = make_k20_20_plus_cliques()
        
        current_score = compute_max_book(adj)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj.copy()
        
        # Simulated Annealing with edge swaps
        T = T_start
        restart_start = time.time()
        max_steps = 30000
        
        # Precompute degree array for faster evaluation
        deg = [adj[i].bit_count() for i in range(n)]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Choose a random edge (u, v) to remove and non-edge (x, y) to add
            # Sample from existing edges
            attempts = 0
            found_edge = False
            while attempts < 20:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and ((adj[u] >> v) & 1):
                    found_edge = True
                    break
                attempts += 1
            
            if not found_edge:
                continue
            
            attempts = 0
            found_nonedge = False
            while attempts < 20:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y and not ((adj[x] >> y) & 1):
                    found_nonedge = True
                    break
                attempts += 1
            
            if not found_nonedge:
                continue
            
            # Ensure not swapping the same edge
            if (u == x and v == y) or (u == y and v == x):
                continue
            
            # Perform swap
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            
            new_score = compute_max_book(adj)
            
            # Acceptance Criteria
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
            else:
                # Accept worse moves with probability exp(-delta/T)
                if T > T_end:
                    prob = np.exp(-1.0 / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        # Revert
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        adj[x] &= ~(1 << y)
                        adj[y] &= ~(1 << x)
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    adj[x] &= ~(1 << y)
                    adj[y] &= ~(1 << x)
            
            # Cool down
            T = max(T_end, T_start * (0.5 ** ((step + 1) / 3000.0)))
            
            if current_score == 7:
                break
    
    if best_adj is None:
        # Fallback: K_20,20 + 1 edge
        adj = [0] * n
        for i in range(20):
            for j in range(20, 40):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        best_adj = adj

    return get_edge_list(best_adj)
