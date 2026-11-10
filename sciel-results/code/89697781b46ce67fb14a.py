import time
import random

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    
    Returns:
        list of edges [u, v]
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    best_edges = None
    best_score = -1
    
    # Grötzsch graph edges (vertices 0-10)
    groetzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [5, 1], [5, 4],
        [6, 0], [6, 2],
        [7, 1], [7, 3],
        [8, 2], [8, 4],
        [9, 3], [9, 0],
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]
    ]
    
    def is_3_colorable_bitmask(adj_masks):
        """
        Returns True if the graph is 3-colorable, False otherwise.
        Uses bitmasks for speed.
        """
        colors = [-1] * n
        
        # Order by degree descending to fail fast
        order = sorted(range(n), key=lambda v: bin(adj_masks[v]).count('1'), reverse=True)
        
        def bt(pos):
            if pos == n:
                return True
            v = order[pos]
            # Check used colors
            used = 0
            mask = adj_masks[v]
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if colors[u] != -1:
                    used |= (1 << colors[u])
                m ^= lsb
                
            # Try colors 0, 1, 2
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if bt(pos + 1):
                        return True
                    colors[v] = -1
            return False
            
        return bt(0)

    def edges_to_masks(edges):
        masks = [0] * n
        for u, v in edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks

    def local_search(masks, edges_list):
        # Greedy add edges
        improved = True
        while improved:
            improved = False
            candidates = []
            for i in range(n):
                for j in range(i+1, n):
                    if not (masks[i] & (1 << j)):
                        candidates.append((i, j))
            
            if not candidates:
                break
                
            random.shuffle(candidates)
            
            for u, v in candidates:
                if masks[u] & (1 << v):
                    continue # Already added
                
                # Check triangle freeness
                if masks[u] & masks[v]:
                    continue
                
                # Tentatively add
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                
                # Check 3-colorability
                if not is_3_colorable_bitmask(masks):
                    # Keep it
                    edges_list.append([u, v])
                    improved = True
                else:
                    # Revert
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    
        return masks, edges_list, improved

    # Initial state: Grötzsch + 5 isolated
    current_edges = groetzsch_edges[:]
    current_masks = edges_to_masks(current_edges)
    
    # Main loop
    while time.time() < deadline:
        current_masks, current_edges, improved = local_search(current_masks, current_edges[:])
        
        count = len(current_edges)
        if count > best_score:
            best_score = count
            best_edges = current_edges[:]
            
        if not improved:
            # Stuck
            if len(current_edges) > 5:
                k = random.randint(2, 4)
                edges_to_remove = random.sample(current_edges, k)
                for u, v in edges_to_remove:
                    current_masks[u] &= ~(1 << v)
                    current_masks[v] &= ~(1 << u)
                    current_edges.remove([u, v])
            else:
                current_edges = groetzsch_edges[:]
                current_masks = edges_to_masks(current_edges)
                
    if best_edges is None:
        return groetzsch_edges
        
    return best_edges
