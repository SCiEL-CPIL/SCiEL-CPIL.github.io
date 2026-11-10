import time
import random

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    
    Uses a combination of spectral heuristic for initial dense core,
    bitmask-based local search, and simulated annealing to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = 16
    
    # Known good starting point: Grötzsch graph (11 vertices, triangle-free, 4-chromatic)
    # Extended with 5 more vertices
    groetzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [5, 1], [5, 4],
        [6, 0], [6, 2],
        [7, 1], [7, 3],
        [8, 2], [8, 4],
        [9, 3], [9, 0],
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]
    ]
    
    def build_masks(edges):
        masks = [0] * n
        for u, v in edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks
    
    def is_triangle_free(masks, u, v):
        """Check if adding edge (u,v) creates a triangle."""
        return (masks[u] & masks[v]) == 0
    
    def is_3_colorable(masks):
        """
        Check if graph is 3-colorable using backtracking with bitmasks.
        Returns True if 3-colorable, False otherwise.
        """
        colors = [-1] * n
        order = sorted(range(n), key=lambda v: masks[v].bit_count(), reverse=True)
        
        def bt(pos):
            if pos == n:
                return True
            v = order[pos]
            used = 0
            m = masks[v]
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if colors[u] != -1:
                    used |= (1 << colors[u])
                m ^= lsb
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if bt(pos + 1):
                        return True
                    colors[v] = -1
            return False
        
        return bt(0)
    
    def count_edges(masks):
        return sum(masks[v].bit_count() for v in range(n)) // 2
    
    def edges_from_masks(masks):
        edges = []
        for u in range(n):
            m = masks[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append([u, v])
                m ^= lsb
        return edges
    
    def greedy_add_edges(masks):
        """
        Greedily add edges that maintain triangle-freeness and 4-chromaticity.
        Returns updated masks and whether any improvement was made.
        """
        improved = False
        # Try all possible edges in random order
        candidates = [(i, j) for i in range(n) for j in range(i + 1, n) 
                      if not (masks[i] & (1 << j))]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if masks[u] & (1 << v):
                continue
            if not is_triangle_free(masks, u, v):
                continue
            
            # Tentatively add
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            
            if not is_3_colorable(masks):
                improved = True
            else:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
        
        return masks, improved
    
    def swap_edges(masks, k_swaps=3):
        """
        Try random edge swaps to escape local optima.
        Remove a random edge, then try to add up to 2 new edges.
        """
        current_edges = edges_from_masks(masks)
        if len(current_edges) < 3:
            return masks, False
        
        improved = False
        for _ in range(k_swaps):
            if time.time() > deadline:
                break
                
            # Pick a random edge to remove
            u, v = random.choice(current_edges)
            if not (masks[u] & (1 << v)):
                continue
                
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            
            # Try to add new edges
            candidates = [(i, j) for i in range(n) for j in range(i + 1, n)
                          if not (masks[i] & (1 << j))]
            random.shuffle(candidates)
            
            added = 0
            for a, b in candidates:
                if added >= 2:
                    break
                if masks[a] & (1 << b):
                    continue
                if not is_triangle_free(masks, a, b):
                    continue
                
                masks[a] |= (1 << b)
                masks[b] |= (1 << a)
                
                if not is_3_colorable(masks):
                    added += 1
                else:
                    masks[a] &= ~(1 << b)
                    masks[b] &= ~(1 << a)
            
            if added > 0:
                improved = True
        
        return masks, improved
    
    def simulated_annealing(masks, iterations=200):
        """
        Simulated annealing to find better edge configurations.
        """
        current = masks[:]
        current_score = count_edges(current)
        best = current[:]
        best_score = current_score
        temperature = 1.0
        cooling = 0.99
        
        for it in range(iterations):
            if time.time() > deadline:
                break
                
            # Generate neighbor: random edge add/remove
            candidates = [(i, j) for i in range(n) for j in range(i + 1, n)]
            u, v = random.choice(candidates)
            
            new_masks = current[:]
            if new_masks[u] & (1 << v):
                new_masks[u] &= ~(1 << v)
                new_masks[v] &= ~(1 << u)
                action = 'remove'
            else:
                if is_triangle_free(new_masks, u, v):
                    new_masks[u] |= (1 << v)
                    new_masks[v] |= (1 << u)
                    action = 'add'
                else:
                    temperature *= cooling
                    continue
            
            # Check validity
            if not is_3_colorable(new_masks):
                new_score = count_edges(new_masks)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < (1 - temperature):
                    current = new_masks
                    current_score = new_score
                    if new_score > best_score:
                        best = new_masks[:]
                        best_score = new_score
            
            temperature *= cooling
        
        return best, best_score
    
    # Main search
    best_edges = None
    best_score = -1
    
    # Start from Grötzsch graph
    current_masks = build_masks(groetzsch_edges)
    current_score = count_edges(current_masks)
    
    if not is_3_colorable(current_masks):
        best_edges = edges_from_masks(current_masks)
        best_score = current_score
    
    # Phase 1: Greedy edge addition
    while time.time() < deadline - 20:
        current_masks, improved = greedy_add_edges(current_masks)
        current_score = count_edges(current_masks)
        
        if current_score > best_score:
            best_score = current_score
            best_edges = edges_from_masks(current_masks)
        
        if not improved:
            # Try swaps
            current_masks, swap_improved = swap_edges(current_masks, k_swaps=5)
            current_score = count_edges(current_masks)
            
            if current_score > best_score:
                best_score = current_score
                best_edges = edges_from_masks(current_masks)
            
            if not swap_improved:
                # Try simulated annealing from current state
                sa_masks, sa_score = simulated_annealing(current_masks, iterations=100)
                if sa_score > best_score:
                    best_score = sa_score
                    best_edges = edges_from_masks(sa_masks)
                    current_masks = sa_masks
                break
    
    # Phase 2: Final polish - try more aggressive search
    if best_edges is not None and time.time() < deadline - 5:
        # Try from the best state found
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            polished, p_score = simulated_annealing(
                build_masks(best_edges), iterations=50
            )
            if p_score > best_score:
                best_score = p_score
                best_edges = edges_from_masks(polished)
    
    if best_edges is None:
        return groetzsch_edges
    
    return best_edges
