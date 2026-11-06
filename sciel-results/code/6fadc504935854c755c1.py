import time
import sys
import random

DEADLINE_MARGIN = 15.0
HARD_LIMIT = 135.0

def _has_triangle(adj_mask, n):
    """Check if the graph represented by adj_mask contains a triangle."""
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mask[u] & (1 << v):
                # If u and v are connected, check if they share a common neighbor
                if adj_mask[u] & adj_mask[v]:
                    return True
    return False

def _is_3_colorable_bitmask(adj_mask, n):
    """
    Check if graph with adjacency bitmasks is 3-colorable.
    adj_mask is a list of n integers, where bit i is 1 if vertex i is a neighbor.
    """
    # Order vertices by degree (popcount) descending
    order = sorted(range(n), key=lambda x: -bin(adj_mask[x]).count('1'))
    
    colors = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        
        v = order[idx]
        # Determine forbidden colors from already colored neighbors
        forbidden = 0
        mask = adj_mask[v]
        m = mask
        while m:
            lsb = m & -m
            nb_idx = lsb.bit_length() - 1
            c = colors[nb_idx]
            if c != -1:
                forbidden |= (1 << c)
            m ^= lsb
        
        for c in range(3):
            if forbidden & (1 << c):
                continue
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = -1
        
        return False
    
    return backtrack(0)

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

    # Initialize with Grötzsch graph on 0-10
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
    initial_adj = adj_mask[:] # Store initial state for fallback
    
    # Local Search with improved strategy
    random.seed(42)
    
    def try_add_all_valid_edges():
        """Try to add all currently valid edges in a greedy manner."""
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if can_add_edge(u, v):
                        add_edge(u, v)
                        improved = True
        return edge_count()
    
    def local_optimize():
        """Perform local optimization: add valid edges, then try swaps."""
        # First, greedily add all valid edges
        try_add_all_valid_edges()
        
        current_count = edge_count()
        if current_count > best_count:
            best_count = current_count
            best_adj = adj_mask[:]
        
        # Now try 1-for-1 swaps to escape local optima
        edges_in_graph = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mask[u] & (1 << v):
                    edges_in_graph.append((u, v))
        
        random.shuffle(edges_in_graph)
        for u, v in edges_in_graph[:30]:
            if time.time() > deadline:
                break
                
            remove_edge(u, v)
            
            # Find all addable edges
            addable = []
            for a in range(n):
                for b in range(a + 1, n):
                    if (a, b) != (u, v) and can_add_edge(a, b):
                        addable.append((a, b))
            
            if addable:
                random.shuffle(addable)
                for a, b in addable[:10]:
                    add_edge(a, b)
                    
                    # Check if still not 3-colorable
                    if not _is_3_colorable_bitmask(adj_mask, n):
                        # Good, keep this state
                        new_count = edge_count()
                        if new_count > best_count:
                            best_count = new_count
                            best_adj = adj_mask[:]
                        break
                    else:
                        remove_edge(a, b)
                else:
                    # No valid swap found, restore original edge
                    add_edge(u, v)
            else:
                add_edge(u, v)
    
    # Main search loop with multiple strategies
    while time.time() < deadline:
        # Strategy 1: Local optimization from current state
        local_optimize()
        
        # Strategy 2: Random restart - try different starting points
        if random.random() < 0.3:
            # Save current state
            saved_adj = adj_mask[:]
            
            # Create a new random triangle-free graph
            adj_mask = [0] * n
            random_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(random_pairs)
            
            for u, v in random_pairs:
                if can_add_edge(u, v):
                    add_edge(u, v)
            
            # Check if not 3-colorable
            if not _is_3_colorable_bitmask(adj_mask, n):
                local_optimize()
            else:
                # Restore previous state
                adj_mask = saved_adj
    
    # Restore best
    adj_mask = best_adj
    
    # CRITICAL FIX: Verify the best graph is triangle-free
    if _has_triangle(adj_mask, n):
        # If a bug caused a triangle, fall back to the initial Grötzsch graph
        adj_mask = initial_adj[:]
    
    # Convert to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mask[u] & (1 << v):
                edges.append([u, v])
                
    return edges
