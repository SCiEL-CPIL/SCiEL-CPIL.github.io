import time
import random
import sys

_colorable_cache = {}

def _is_3_colorable(adj_mask, n):
    """
    Check if graph with adjacency bitmasks is 3-colorable.
    Returns True if 3-colorable, False otherwise.
    Uses efficient backtracking with bitmasks and caching.
    """
    key = tuple(adj_mask)
    if key in _colorable_cache:
        return _colorable_cache[key]
    
    color = [-1] * n
    color_mask = [0, 0, 0]
    
    # Precompute degrees for heuristic
    degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
    
    def backtrack(uncolored_count):
        if uncolored_count == 0:
            return True
        
        # Find the best uncolored vertex to color next
        # Heuristic: Fewest available colors first (MRV), then highest degree
        best_v = -1
        best_deg = -1
        best_avail_count = 4
        best_avail_mask = 0
        
        for v in range(n):
            if color[v] != -1:
                continue
            
            # Compute available colors
            avail = 0
            if not (adj_mask[v] & color_mask[0]):
                avail |= 1
            if not (adj_mask[v] & color_mask[1]):
                avail |= 2
            if not (adj_mask[v] & color_mask[2]):
                avail |= 4
            
            if avail == 0:
                return False
            
            avail_count = bin(avail).count('1')
            
            # Prefer fewer available colors (MRV), then higher degree
            if avail_count < best_avail_count or (avail_count == best_avail_count and degrees[v] > best_deg):
                best_v = v
                best_deg = degrees[v]
                best_avail_count = avail_count
                best_avail_mask = avail
        
        if best_v == -1:
            return True
        
        v = best_v
        
        # Try colors in order of least impact
        for c in range(3):
            if not (best_avail_mask & (1 << c)):
                continue
            
            color[v] = c
            color_mask[c] |= (1 << v)
            
            if backtrack(uncolored_count - 1):
                return True
            
            color_mask[c] &= ~(1 << v)
            color[v] = -1
        
        return False
    
    result = backtrack(n)
    _colorable_cache[key] = result
    return result

def _build_adj_mask(edges, n=16):
    """Build adjacency bitmask list from edge list."""
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj_mask

def _is_triangle_free(adj_mask, n):
    """Check if graph is triangle-free using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_mask[u] >> v) & 1:
                # Check if u and v have a common neighbor
                common = adj_mask[u] & adj_mask[v]
                if common != 0:
                    return False
    return True

def _can_add_edge(u, v, adj_mask):
    """Check if edge (u,v) can be added without creating a triangle."""
    if u == v:
        return False
    if (adj_mask[u] >> v) & 1:
        return False
    # Check for common neighbors
    common = adj_mask[u] & adj_mask[v]
    if common != 0:
        return False
    return True

def _groetzsch_edges():
    """Return edges of the Grötzsch graph (Mycielski of C5) on vertices 0-10."""
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    return edges

def _try_add_edge(u, v, adj_mask, edges, n):
    """
    Try to add edge (u,v) to the graph.
    Returns (new_adj_mask, new_edges) if the edge can be added and the graph
    remains not 3-colorable, else None.
    """
    if not _can_add_edge(u, v, adj_mask):
        return None
    
    new_adj_mask = list(adj_mask)
    new_adj_mask[u] |= (1 << v)
    new_adj_mask[v] |= (1 << u)
    
    # Check if still not 3-colorable
    if _is_3_colorable(new_adj_mask, n):
        return None
    
    new_edges = edges + [[u, v]]
    return new_adj_mask, new_edges

def construct():
    start = time.time()
    deadline = start + 115.0  # Leave 5s safety margin for 120s limit
    n = 16
    
    best_edges = []
    best_count = -1
    
    def elapsed_ok():
        return time.time() < deadline
    
    def run_greedy(seed_order=None):
        """
        Run greedy edge addition starting from Grötzsch graph.
        Uses randomization to escape local optima.
        """
        base_edges = _groetzsch_edges()
        adj_mask = _build_adj_mask(base_edges, n)
        edges = [list(e) for e in base_edges]
        
        # Verify base is valid
        if not _is_triangle_free(adj_mask, n) or _is_3_colorable(adj_mask, n):
            return adj_mask, edges, len(edges)
        
        # All possible pairs
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # Shuffle pairs based on seed
        if seed_order is not None:
            rng = random.Random(seed_order)
            rng.shuffle(all_pairs)
        else:
            all_pairs = list(all_pairs)
        
        # Phase 1: Greedy addition - try to add edges in shuffled order
        changed = True
        while changed and elapsed_ok():
            changed = False
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if (adj_mask[u] >> v) & 1:
                    continue
                res = _try_add_edge(u, v, adj_mask, edges, n)
                if res is not None:
                    adj_mask, edges = res
                    changed = True
        
        # Phase 2: Intensive local search with multiple swap attempts
        improved = True
        swap_attempts = 0
        while improved and elapsed_ok() and swap_attempts < 5000:
            improved = False
            swap_attempts += 1
            
            # Try to add any edge
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if (adj_mask[u] >> v) & 1:
                    continue
                res = _try_add_edge(u, v, adj_mask, edges, n)
                if res is not None:
                    adj_mask, edges = res
                    improved = True
                    break
            
            if improved or not elapsed_ok():
                continue
            
            # Try 1-1 swap: remove one edge, try to add another
            if edges:
                # Try multiple random edges to remove
                tried = 0
                while tried < 5 and elapsed_ok():
                    tried += 1
                    idx = random.randint(0, len(edges) - 1)
                    u_rem, v_rem = edges[idx]
                    
                    # Remove edge
                    cand_adj_mask = list(adj_mask)
                    cand_adj_mask[u_rem] &= ~(1 << v_rem)
                    cand_adj_mask[v_rem] &= ~(1 << u_rem)
                    cand_edges = edges[:idx] + edges[idx+1:]
                    
                    # Try to add a different edge
                    pairs_to_try = [(u, v) for u in range(n) for v in range(u+1, n) 
                                   if not ((cand_adj_mask[u] >> v) & 1)]
                    random.shuffle(pairs_to_try)
                    
                    swapped = False
                    for u_add, v_add in pairs_to_try[:200]:
                        if not elapsed_ok():
                            break
                        if not _can_add_edge(u_add, v_add, cand_adj_mask):
                            continue
                        
                        tmp_adj_mask = list(cand_adj_mask)
                        tmp_adj_mask[u_add] |= (1 << v_add)
                        tmp_adj_mask[v_add] |= (1 << u_add)
                        
                        if _is_3_colorable(tmp_adj_mask, n):
                            continue
                        
                        # Valid swap
                        adj_mask = tmp_adj_mask
                        edges = cand_edges + [[u_add, v_add]]
                        swapped = True
                        break
                    
                    if swapped:
                        improved = True
                        break
        
        return adj_mask, edges, len(edges)
    
    # Run multiple instances with different seeds
    max_restarts = 200
    for i in range(max_restarts):
        if not elapsed_ok():
            break
        
        seed = i * 1000 + 42
        adj_mask, edges, count = run_greedy(seed)
        
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edges]
    
    # Final validation
    if best_edges:
        final_adj_mask = _build_adj_mask(best_edges, n)
        assert _is_triangle_free(final_adj_mask, n), "Graph is not triangle-free"
        assert not _is_3_colorable(final_adj_mask, n), "Graph is 3-colorable"
        return best_edges
    else:
        # Fallback to Grötzsch
        base_edges = _groetzsch_edges()
        return [list(e) for e in base_edges]
