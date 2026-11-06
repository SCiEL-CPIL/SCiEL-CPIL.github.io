import time
import random

def _is_not_3_colorable(adj, n):
    """
    Check if graph is NOT 3-colorable using bitmask backtracking.
    adj: list of int bitmasks.
    """
    # Order vertices by degree descending
    deg = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    # pos_of[v] is the index of v in the order
    pos_of = [0] * n
    for idx, v in enumerate(order):
        pos_of[v] = idx
        
    # For each vertex v, compute mask of neighbors that appear earlier in the order
    # This allows us to check constraints only against already-colored vertices
    earlier_neigh = [0] * n
    for idx, v in enumerate(order):
        mask = 0
        # Iterate through set bits of adj[v]
        m = adj[v]
        while m:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            if pos_of[u] < idx:
                mask |= (1 << pos_of[u])
            m ^= lsb
        earlier_neigh[v] = mask
        
    # color_masks[c] has bit idx set if order[idx] has color c
    color_masks = [0, 0, 0]
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        em = earlier_neigh[v]
        forbidden = 0
        # Check which colors are used by earlier neighbors
        for c in range(3):
            if em & color_masks[c]:
                forbidden |= (1 << c)
                
        available = 7 & ~forbidden
        if available == 0:
            return False
            
        while available:
            c_bit = available & (-available)
            c = c_bit.bit_length() - 1
            color_masks[c] |= (1 << idx)
            if backtrack(idx + 1):
                return True
            color_masks[c] ^= (1 << idx)
            available ^= c_bit
        return False
        
    return not backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 16
    
    # Grötzsch graph edges (11 vertices: 0..10)
    # Cycle 0-1-2-3-4-0
    # 5 connects to 1, 4
    # 6 connects to 2, 0
    # 7 connects to 3, 1
    # 8 connects to 4, 2
    # 9 connects to 0, 3
    # 10 connects to 5, 6, 7, 8, 9
    core_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 2), (6, 0),
        (7, 3), (7, 1),
        (8, 4), (8, 2),
        (9, 0), (9, 3),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    core_set = set(core_edges)
    non_core_edges = [e for e in all_pairs if e not in core_set]
    
    best_edges_list = []
    best_count = 0
    best_adj = [0] * n
    
    # Initialize best_adj with core
    for u, v in core_edges:
        best_adj[u] |= (1 << v)
        best_adj[v] |= (1 << u)
    best_count = len(core_edges)
    best_edges_list = [[u, v] for u, v in core_edges]
    
    # Run many random restarts of greedy fill
    # The greedy fill is very fast (O(E^2) roughly, but with bitmasks it's O(E) per edge check)
    # The 3-colorability check is the bottleneck.
    
    for seed in range(20000): # High number, will break on time
        if time.time() > deadline:
            break
            
        random.seed(seed)
        adj = [0] * n
        for u, v in core_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        shuffled_non_core = non_core_edges[:]
        random.shuffle(shuffled_non_core)
        
        edge_count = len(core_edges)
        
        for u, v in shuffled_non_core:
            # Check if adding (u,v) creates a triangle
            # Triangle exists if there is a common neighbor w
            # i.e., (adj[u] & adj[v]) != 0
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                
        # If this graph has fewer edges than our current best, skip the expensive check
        if edge_count <= best_count:
            continue
            
        # Check if it is NOT 3-colorable
        if _is_not_3_colorable(adj, n):
            # It is valid and has more edges
            best_count = edge_count
            best_adj = adj[:]
            # Extract edges
            best_edges_list = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        best_edges_list.append([u, v])
                        
    return best_edges_list
