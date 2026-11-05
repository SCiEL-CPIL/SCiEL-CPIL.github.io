<<<<<<< SEARCH
import random
import time

def construct():
    n = 20
    start_time = time.time()
    max_time = 115.0  # Safety margin for 120s limit
    
    best_edges = []
    best_count = 0
    
    def get_all_pairs(n):
        return [(u, v) for u in range(n) for v in range(u + 1, n)]

    def is_valid_add(u, v, adj_mask, common):
        # Check if edge (u, v) can be added without violating K2,3-free constraint
        # Constraint: No pair of vertices has > 2 common neighbors.
        # Adding (u, v) increases common[u][w] for w in N(v)\{u} and common[v][w] for w in N(u)\{v}.
        # So we need common[u][w] < 2 for all w in N(v)\{u}
        # and common[v][w] < 2 for all w in N(u)\{v}
        
        if (adj_mask[u] >> v) & 1:
            return False
            
        # Check neighbors of v
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u and common[u][w] >= 2:
                return False
            m -= lsb
            
        # Check neighbors of u
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v and common[v][w] >= 2:
                return False
            m -= lsb
            
        return True

    def add_edge(u, v, adj_mask, common, edges):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append([u, v])
        
        # Update common counts
        # For w in N(v) (current, includes u now? No, we update adj first)
        # N(v) now includes u.
        # We need to increment common[u][w] for w in N(v) \ {u}
        # And common[v][w] for w in N(u) \ {v}
        
        # Iterate N(v) excluding u
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            m -= lsb
            
        # Iterate N(u) excluding v
        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            m -= lsb

    def remove_edge(u, v, adj_mask, common, edges):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.remove([u, v])
        
        # Update common counts (decrement)
        # For w in N(v) (current, excludes u now)
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            m -= lsb
            
        # For w in N(u) (current, excludes v now)
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            m -= lsb

    def greedy_fill(adj_mask, common, edges):
        all_pairs = get_all_pairs(n)
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if is_valid_add(u, v, adj_mask, common):
                add_edge(u, v, adj_mask, common, edges)

    def local_search(e1):
        # Initialize state from e1
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        for u, v in e1:
            add_edge(u, v, adj, common, edges)
            
        current_count = len(edges)
        best_local_count = current_count
        best_local_edges = list(edges)
        
        all_pairs = get_all_pairs(n)
        # Precompute list of non-edges for faster access? 
        # Or just shuffle all pairs and check validity.
        
        temperature = 1.0
        cooling = 0.999
        
        for step in range(5000): # Fixed steps per restart, or time-based
            if time.time() - start_time > max_time - 5:
                break
                
            # 1. Try to add a random edge
            u, v = random.choice(all_pairs)
            if is_valid_add(u, v, adj, common):
                add_edge(u, v, adj, common, edges)
                current_count += 1
                if current_count > best_local_count:
                    best_local_count = current_count
                    best_local_edges = list(edges)
                continue
            
            # 2. Try a swap: Remove one edge, add this one
            # Pick a random existing edge to remove
            if edges:
                ru, rv = random.choice(edges)
                # Remove ru, rv
                remove_edge(ru, rv, adj, common, edges)
                current_count -= 1
                
                # Try to add u, v
                if is_valid_add(u, v, adj, common):
                    add_edge(u, v, adj, common, edges)
                    current_count += 1
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_edges = list(edges)
                else:
                    # Revert: Add back ru, rv
                    add_edge(ru, rv, adj, common, edges)
                    current_count += 1
            
            # 3. Random kick: Remove 2 edges, add 1
            if len(edges) >= 2:
                # Pick 2 distinct edges
                e_idx1 = random.randint(0, len(edges)-1)
                e_idx2 = random.randint(0, len(edges)-1)
                if e_idx1 == e_idx2:
                    continue
                e1_remove = edges[e_idx1]
                e2_remove = edges[e_idx2]
                
                remove_edge(e1_remove[0], e1_remove[1], adj, common, edges)
                remove_edge(e2_remove[0], e2_remove[1], adj, common, edges)
                current_count -= 2
                
                # Try to add u, v
                if is_valid_add(u, v, adj, common):
                    add_edge(u, v, adj, common, edges)
                    current_count += 1
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_edges = list(edges)
                else:
                    # Revert
                    add_edge(e1_remove[0], e1_remove[1], adj, common, edges)
                    add_edge(e2_remove[0], e2_remove[1], adj, common, edges)
                    current_count += 2

        return best_local_edges

    # Main loop
    while time.time() - start_time < max_time - 10:
        adj1 = [0] * n
        common1 = [[0] * n for _ in range(n)]
        edges1 = []
        greedy_fill(adj1, common1, edges1)
        
        res_edges = local_search(edges1)
        
        if len(res_edges) > best_count:
            best_count = len(res_edges)
            best_edges = res_edges

    # Final Verification
    if not best_edges:
        best_edges = [[0, 1]]
    else:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        valid = True
        for u in range(n):
            for v in range(u + 1, n):
                if bin(adj[u] & adj[v]).count('1') > 2:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            return [[0, 1]]

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
=======
import random
import time

def construct():
    n = 20
    start_time = time.time()
    max_time = 145.0  # Safety margin for 150s limit
    
    best_edges = []
    best_count = 0
    
    def get_all_pairs(n):
        return [(u, v) for u in range(n) for v in range(u + 1, n)]

    def is_valid_add(u, v, adj_mask, common):
        if (adj_mask[u] >> v) & 1:
            return False
            
        # Check neighbors of v
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u and common[u][w] >= 2:
                return False
            m -= lsb
            
        # Check neighbors of u
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v and common[v][w] >= 2:
                return False
            m -= lsb
            
        return True

    def add_edge(u, v, adj_mask, common, edges):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append([u, v])
        
        # Update common counts
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            m -= lsb
            
        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            m -= lsb

    def remove_edge(u, v, adj_mask, common, edges):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.remove([u, v])
        
        # Update common counts (decrement)
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            m -= lsb
            
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            m -= lsb

    def greedy_fill(adj_mask, common, edges):
        all_pairs = get_all_pairs(n)
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if is_valid_add(u, v, adj_mask, common):
                add_edge(u, v, adj_mask, common, edges)

    def circulant_init():
        """Create a circulant-like graph with structured edges"""
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        
        # Add circulant edges with step sizes that avoid K2,3
        # For n=20, use steps that create a structured but not too dense graph
        steps = [1, 2, 3, 4, 5]  # Try multiple circulant steps
        
        for step in steps:
            for u in range(n):
                v = (u + step) % n
                if u < v:  # Avoid duplicate edges
                    if is_valid_add(u, v, adj, common):
                        add_edge(u, v, adj, common, edges)
        
        # Fill remaining capacity with greedy
        all_pairs = get_all_pairs(n)
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if is_valid_add(u, v, adj, common):
                add_edge(u, v, adj, common, edges)
        
        return edges

    def random_init_density(density):
        """Create random initialization with specified edge density"""
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        
        all_pairs = get_all_pairs(n)
        random.shuffle(all_pairs)
        
        # Add edges up to target density
        target_edges = int(density * len(all_pairs))
        count = 0
        for u, v in all_pairs:
            if count >= target_edges:
                break
            if is_valid_add(u, v, adj, common):
                add_edge(u, v, adj, common, edges)
                count += 1
        
        # Fill remaining with greedy
        for u, v in all_pairs:
            if is_valid_add(u, v, adj, common):
                add_edge(u, v, adj, common, edges)
        
        return edges

    def local_search(e1, max_steps=3000):
        # Initialize state from e1
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        for u, v in e1:
            add_edge(u, v, adj, common, edges)
            
        current_count = len(edges)
        best_local_count = current_count
        best_local_edges = list(edges)
        
        all_pairs = get_all_pairs(n)
        
        for step in range(max_steps):
            if time.time() - start_time > max_time - 5:
                break
                
            # 1. Try to add a random edge
            u, v = random.choice(all_pairs)
            if is_valid_add(u, v, adj, common):
                add_edge(u, v, adj, common, edges)
                current_count += 1
                if current_count > best_local_count:
                    best_local_count = current_count
                    best_local_edges = list(edges)
                continue
            
            # 2. Try a swap: Remove one edge, add this one
            if edges:
                ru, rv = random.choice(edges)
                # Remove ru, rv
                remove_edge(ru, rv, adj, common, edges)
                current_count -= 1
                
                # Try to add u, v
                if is_valid_add(u, v, adj, common):
                    add_edge(u, v, adj, common, edges)
                    current_count += 1
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_edges = list(edges)
                else:
                    # Revert: Add back ru, rv
                    add_edge(ru, rv, adj, common, edges)
                    current_count += 1
            
            # 3. Random kick: Remove 2 edges, add 1
            if len(edges) >= 2:
                e_idx1 = random.randint(0, len(edges)-1)
                e_idx2 = random.randint(0, len(edges)-1)
                if e_idx1 == e_idx2:
                    continue
                e1_remove = edges[e_idx1]
                e2_remove = edges[e_idx2]
                
                remove_edge(e1_remove[0], e1_remove[1], adj, common, edges)
                remove_edge(e2_remove[0], e2_remove[1], adj, common, edges)
                current_count -= 2
                
                # Try to add u, v
                if is_valid_add(u, v, adj, common):
                    add_edge(u, v, adj, common, edges)
                    current_count += 1
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_edges = list(edges)
                else:
                    # Revert
                    add_edge(e1_remove[0], e1_remove[1], adj, common, edges)
                    add_edge(e2_remove[0], e2_remove[1], adj, common, edges)
                    current_count += 2

        return best_local_edges

    # Main loop with diverse initializations
    init_strategies = [
        ("circulant", circulant_init),
        ("random_full", lambda: random_init_density(1.0)),
        ("random_med", lambda: random_init_density(0.7)),
        ("random_sparse", lambda: random_init_density(0.4)),
    ]
    
    restart_count = 0
    while time.time() - start_time < max_time - 10:
        # Cycle through initialization strategies
        strategy_name, init_func = init_strategies[restart_count % len(init_strategies)]
        
        if strategy_name == "circulant":
            edges1 = init_func()
        else:
            edges1 = init_func()
        
        # Local search with adaptive steps based on remaining time
        remaining = max_time - 5 - (time.time() - start_time)
        if remaining < 3:
            break
        steps = min(3000, max(500, int(remaining * 500)))
        
        res_edges = local_search(edges1, steps)
        
        if len(res_edges) > best_count:
            best_count = len(res_edges)
            best_edges = res_edges
        
        restart_count += 1

    # Final Verification
    if not best_edges:
        best_edges = [[0, 1]]
    else:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        valid = True
        for u in range(n):
            for v in range(u + 1, n):
                if bin(adj[u] & adj[v]).count('1') > 2:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            return [[0, 1]]

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
>>>>>>> REPLACE
