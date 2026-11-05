import random
import time

def construct():
    n = 20
    start_time = time.time()
    max_time = 115.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    def get_all_pairs(n):
        return [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_add(u, v, adj_mask, common):
        if (adj_mask[u] >> v) & 1:
            return False
        if common[u][v] > 2:
            return False
        # Check if adding (u,v) would make any (u,w) or (v,w) invalid?
        # Adding (u,v) increases common[u][w] by 1 for all w in N(v).
        # We need common[u][w] < 2 BEFORE adding.
        # Similarly, common[v][w] < 2 for all w in N(u).
        
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u and common[u][w] >= 2:
                return False
            m -= lsb
            
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v and common[v][w] >= 2:
                return False
            m -= lsb
            
        return True

    def add_edge_to_state(u, v, adj_mask, common, edges):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append([u, v])
        
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u:
                common[u][w] += 1
                common[w][u] += 1
            m -= lsb
            
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v:
                common[v][w] += 1
                common[w][v] += 1
            m -= lsb

    def remove_edge_from_state(u, v, adj_mask, common, edges):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.remove([u, v])
        
        # Decrement common
        # For w in CURRENT N(v) (which is OLD N(v) \ {u}), common[u][w] -= 1
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            m -= lsb
            
        # For w in CURRENT N(u) (which is OLD N(u) \ {v}), common[v][w] -= 1
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
                add_edge_to_state(u, v, adj_mask, common, edges)

    def circulant_init():
        """Create a circulant-like structured initialization."""
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        
        # Use circulant graph with steps 1, 2, 3 (but check validity)
        steps = [1, 2, 3]
        for s in steps:
            for u in range(n):
                v = (u + s) % n
                if u < v:
                    if is_valid_add(u, v, adj_mask, common):
                        add_edge_to_state(u, v, adj_mask, common, edges)
        
        # Also try some random additional edges
        all_pairs = get_all_pairs(n)
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if is_valid_add(u, v, adj_mask, common):
                add_edge_to_state(u, v, adj_mask, common, edges)
        
        return edges

    def sparse_random_init():
        """Create a sparse random initialization."""
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        
        all_pairs = get_all_pairs(n)
        random.shuffle(all_pairs)
        # Only take ~40% of possible edges initially
        for u, v in all_pairs[:int(len(all_pairs) * 0.4)]:
            if is_valid_add(u, v, adj_mask, common):
                add_edge_to_state(u, v, adj_mask, common, edges)
        
        return edges

    def local_search_with_kicks(initial_edges, max_kicks=10):
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        
        for u, v in initial_edges:
            add_edge_to_state(u, v, adj_mask, common, edges)
            
        current_count = len(edges)
        best_local_edges = list(edges)
        best_local_count = current_count
        
        for _ in range(200): # Steps per restart
            # Try to add any valid edge
            all_pairs = get_all_pairs(n)
            random.shuffle(all_pairs)
            added_any = False
            for u, v in all_pairs:
                if is_valid_add(u, v, adj_mask, common):
                    add_edge_to_state(u, v, adj_mask, common, edges)
                    added_any = True
                    if len(edges) > best_local_count:
                        best_local_count = len(edges)
                        best_local_edges = list(edges)
                    break # Restart search for valid edge after each add to keep it simple/fast
            
            if not added_any:
                # Local optimum reached
                if not edges:
                    break
                
                # Kick: remove 1 or 2 edges
                kick_size = 1 if random.random() < 0.5 else 2
                
                # Select edges to remove randomly
                to_remove = []
                available = edges[:]
                for _ in range(kick_size):
                    if not available: break
                    idx = random.randint(0, len(available) - 1)
                    to_remove.append(available[idx])
                    available.pop(idx)
                
                for u, v in to_remove:
                    remove_edge_from_state(u, v, adj_mask, common, edges)
                
                # Try to fill again
                for step in range(50):
                    all_pairs = get_all_pairs(n)
                    random.shuffle(all_pairs)
                    added = False
                    for u, v in all_pairs:
                        if is_valid_add(u, v, adj_mask, common):
                            add_edge_to_state(u, v, adj_mask, common, edges)
                            added = True
                            if len(edges) > best_local_count:
                                best_local_count = len(edges)
                                best_local_edges = list(edges)
                            break
                    if not added:
                        break

        return best_local_edges

    # Main loop
    all_possible_edges = get_all_pairs(n)
    
    # Use diverse initialization strategies
    init_strategies = [
        ("random", lambda: None),  # Will use greedy_fill below
        ("circulant", circulant_init),
        ("sparse_random", sparse_random_init),
    ]
    
    strategy_idx = 0
    while time.time() - start_time < max_time:
        # 1. Diverse Initialization
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        
        strategy_name = init_strategies[strategy_idx % len(init_strategies)][0]
        strategy_func = init_strategies[strategy_idx % len(init_strategies)][1]
        
        if strategy_name == "random":
            greedy_fill(adj_mask, common, edges)
        else:
            edges = strategy_func()
        
        strategy_idx += 1
        
        # 2. Local Search
        result_edges = local_search_with_kicks(edges)
        
        if len(result_edges) > best_count:
            best_count = len(result_edges)
            best_edges = result_edges
            
        # Check time
        if time.time() - start_time > max_time - 5:
            break

    # Final Verification
    if not best_edges:
        # Fallback
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
             # This should not happen if state maintenance is correct
             # But for safety, return a small valid graph
             return [[0, 1]]

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
