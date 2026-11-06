import time
import random
import sys

DEADLINE_MARGIN = 12.0
HARD_LIMIT = 138.0

def _is_3_colorable(adj_mask, n):
    """
    Check if graph with adjacency bitmasks is 3-colorable using backtracking.
    Returns True if 3-colorable, False otherwise.
    """
    # Order vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda x: -bin(adj_mask[x]).count('1'))
    # Map original vertex to position in order
    pos_in_order = [0] * n
    for i, v in enumerate(order):
        pos_in_order[v] = i
    
    colors = [-1] * n
    
    # Precompute for each vertex in order, which neighbors come before it
    # and their positions
    def backtrack(idx):
        if idx == n:
            return True
        
        v = order[idx]
        # Determine forbidden colors
        forbidden = 0
        mask = adj_mask[v]
        m = mask
        while m:
            lsb = m & -m
            nb = lsb.bit_length() - 1
            if pos_in_order[nb] < idx:  # neighbor already colored
                c = colors[nb]
                if c != -1:
                    forbidden |= (1 << c)
            m ^= lsb
        
        # Try colors 0, 1, 2
        for c in range(3):
            if forbidden & (1 << c):
                continue
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = -1
        
        return False
    
    return backtrack(0)

def _is_triangle_free(adj_mask, n):
    """Check if graph is triangle-free."""
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mask[u] & (1 << v):
                if adj_mask[u] & adj_mask[v]:
                    return False
    return True

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # adj_mask[i] is an integer where bit j is 1 if (i,j) is an edge
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        
    def edge_count():
        return sum(bin(m).count('1') for m in adj_mask) // 2
        
    def can_add_edge(u, v):
        if u == v:
            return False
        if adj_mask[u] & (1 << v):
            return False
        # Check triangle-free: common neighbors must be empty
        if adj_mask[u] & adj_mask[v]:
            return False
        return True

    def get_all_pairs():
        return [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def get_addable_edges():
        cands = []
        for u in range(n):
            for v in range(u + 1, n):
                if can_add_edge(u, v):
                    cands.append((u, v))
        return cands

    def get_current_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mask[u] & (1 << v):
                    edges.append((u, v))
        return edges

    # Initialize with Grötzsch graph on 0-10 (11 vertices, 20 edges)
    # C5: 0-1-2-3-4-0
    for i in range(5):
        add_edge(i, (i + 1) % 5)
    # Mycielski layer 5-9
    for i in range(5):
        add_edge(5 + i, (i + 1) % 5)
        add_edge(5 + i, (i - 1) % 5)
    # Universal 10
    for i in range(5):
        add_edge(10, 5 + i)
        
    # Vertices 11-15 are isolated initially
    
    best_count = edge_count()
    best_adj = adj_mask[:]
    initial_adj = adj_mask[:]
    
    random.seed(42)
    
    # Phase 1: Greedy edge addition
    while time.time() < deadline:
        improved = False
        all_pairs = get_all_pairs()
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if can_add_edge(u, v):
                add_edge(u, v)
                improved = True
        
        current_count = edge_count()
        if current_count > best_count:
            best_count = current_count
            best_adj = adj_mask[:]
            
        if not improved:
            break
    
    # Phase 2: Local search with swaps to escape local optima
    stagnation = 0
    max_stagnation = 50
    
    while time.time() < deadline and stagnation < max_stagnation:
        # Try to add any valid edge first
        improved = False
        all_pairs = get_all_pairs()
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if can_add_edge(u, v):
                add_edge(u, v)
                improved = True
        
        current_count = edge_count()
        if current_count > best_count:
            best_count = current_count
            best_adj = adj_mask[:]
            stagnation = 0
            continue
        
        if not improved:
            # Try swap moves: remove one edge, add two edges
            # This can escape local optima
            edges_in_graph = get_current_edges()
            random.shuffle(edges_in_graph)
            
            swap_found = False
            for u, v in edges_in_graph[:10]:  # Limit swaps per iteration
                if time.time() > deadline:
                    break
                
                remove_edge(u, v)
                
                # Try to add two edges that were previously blocked
                addable = get_addable_edges()
                if len(addable) >= 2:
                    # Try all pairs of addable edges
                    random.shuffle(addable)
                    found_pair = False
                    for i in range(min(len(addable), 20)):
                        for j in range(i + 1, min(len(addable), 20)):
                            a1, b1 = addable[i]
                            a2, b2 = addable[j]
                            # Check if adding both keeps triangle-free
                            # Temporarily add first
                            add_edge(a1, b1)
                            if can_add_edge(a2, b2):
                                add_edge(a2, b2)
                                # Check 3-colorability
                                if not _is_3_colorable(adj_mask, n):
                                    # Good move
                                    current_count = edge_count()
                                    if current_count > best_count:
                                        best_count = current_count
                                        best_adj = adj_mask[:]
                                    swap_found = True
                                    found_pair = True
                                    break
                                else:
                                    remove_edge(a2, b2)
                            remove_edge(a1, b1)
                            if found_pair:
                                break
                        if found_pair:
                            break
                    
                    if found_pair:
                        swap_found = True
                        stagnation = 0
                        break
                
                if not swap_found:
                    add_edge(u, v)  # Revert
            
            if not swap_found:
                stagnation += 1
    
    # Restore best
    adj_mask = best_adj
    
    # Verify the best graph is triangle-free
    if not _is_triangle_free(adj_mask, n):
        adj_mask = initial_adj[:]
    
    # Convert to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mask[u] & (1 << v):
                edges.append([u, v])
                
    return edges
