"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""
import time
import random
import itertools

_DEADLINE = 150.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START) - 5.0  # 5s safety margin


def _is_mono(color_mask, edge_mask):
    """Check if edge is monochromatic under coloring color_mask."""
    return (color_mask & edge_mask) == 0 or (color_mask & edge_mask) == edge_mask


def _greedy_set_cover(n, deadline):
    """
    Greedy set cover over the space of colorings.
    Each edge 'covers' (kills) the set of colorings that make it monochromatic.
    We want to cover all colorings (excluding trivial all-0 and all-1).
    Returns list of edge tuples or None.
    """
    vertices = range(n)
    all_edges = list(itertools.combinations(vertices, 5))
    
    # Precompute edge masks
    edge_masks = []
    for e in all_edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Valid colorings: exclude all-0 and all-1
    # For n up to ~20, 2^n is manageable. For larger n, this approach breaks.
    # But the task says n <= 26. 2^26 is 67M, too big for a list.
    # However, we only need to find a small set of edges.
    # If n is large, the number of colorings is huge, but maybe we can use a different representation?
    # Actually, for n > 15, 2^n is > 32k.
    # Let's stick to n <= 16 for the list-based approach.
    # For n > 16, we might need a different strategy or just skip.
    # The best known construction has 51 edges.
    # Typically these are on n ~ 10-15 vertices.
    
    if n > 18:
        return None  # Too many colorings for list-based set cover

    # Initialize set of uncovered colorings
    # Represent as a set of integers
    all_colorings = set(range(1, (1 << n) - 1))
    
    selected_edges = []
    selected_masks = []
    
    # Precompute for each edge the set of colorings it kills?
    # That would be len(all_edges) * len(all_colorings) booleans.
    # For n=15, len(all_edges)=3003, len(all_colorings)=32766. ~100M entries. Too much memory.
    # Instead, compute on the fly or use bitsets if possible.
    # Python sets are fast enough for small n.
    
    while all_colorings:
        if time.time() > deadline - 2:
            return None
            
        best_edge_idx = -1
        best_kill_count = 0
        
        # Iterate through all edges
        # Optimization: skip edges already selected
        selected_set = set(selected_edges)
        
        for i, e in enumerate(all_edges):
            if e in selected_set:
                continue
                
            em = edge_masks[i]
            kill_count = 0
            # Count how many uncovered colorings are killed by this edge
            for c in all_colorings:
                if _is_mono(c, em):
                    kill_count += 1
                    
            if kill_count > best_kill_count:
                best_kill_count = kill_count
                best_edge_idx = i
                
        if best_edge_idx == -1 or best_kill_count == 0:
            break
            
        # Add the best edge
        best_edge = all_edges[best_edge_idx]
        best_mask = edge_masks[best_edge_idx]
        selected_edges.append(best_edge)
        selected_masks.append(best_mask)
        
        # Remove killed colorings
        to_remove = set()
        for c in all_colorings:
            if _is_mono(c, best_mask):
                to_remove.add(c)
        all_colorings -= to_remove
        
        if len(selected_edges) > 200:
            return None
            
    if not all_colorings:
        return selected_edges
    else:
        return None


def _local_search_reduce(edges, n, deadline):
    """
    Try to reduce the number of edges by removing one and checking if still non-2-colorable.
    Uses the set-cover check implicitly: if removing an edge leaves some colorings uncovered,
    then it was necessary. If all colorings are still covered, it was redundant.
    """
    current = list(edges)
    edge_masks = []
    for e in current:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Get the set of colorings covered by the current set of edges
    # Actually, we just need to check if the union of killed sets is all colorings.
    
    def is_non_colorable(masks):
        if n > 18:
            # Fallback to backtracking for large n? Or just assume True if we got here?
            # For now, if n > 18, we can't do this check efficiently.
            return True
            
        all_colorings = set(range(1, (1 << n) - 1))
        for em in masks:
            to_remove = set()
            for c in all_colorings:
                if _is_mono(c, em):
                    to_remove.add(c)
            all_colorings -= to_remove
            if not all_colorings:
                return True
        return len(all_colorings) == 0

    # Check current
    if not is_non_colorable(edge_masks):
        return current  # Should not happen

    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            if time.time() > deadline - 2:
                return current
            # Try removing edge i
            candidate_masks = edge_masks[:i] + edge_masks[i+1:]
            if is_non_colorable(candidate_masks):
                # Edge i is redundant
                current.pop(i)
                edge_masks.pop(i)
                improved = True
                break
                
    return current


def _random_search(n, target_edges, deadline):
    """
    Randomly sample sets of edges of size target_edges and check if non-2-colorable.
    """
    vertices = range(n)
    all_edges = list(itertools.combinations(vertices, 5))
    all_masks = []
    for e in all_edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        all_masks.append(mask)
        
    for _ in range(1000):
        if time.time() > deadline - 2:
            return None
            
        indices = random.sample(range(len(all_edges)), target_edges)
        masks = [all_masks[i] for i in indices]
        
        # Check non-2-colorable
        if n > 18:
            # Use backtracking? Too complex here. Skip.
            continue
            
        all_colorings = set(range(1, (1 << n) - 1))
        for em in masks:
            to_remove = set()
            for c in all_colorings:
                if _is_mono(c, em):
                    to_remove.add(c)
            all_colorings -= to_remove
            if not all_colorings:
                break
                
        if not all_colorings:
            return [all_edges[i] for i in indices]
            
    return None


def construct():
    start = time.time()
    deadline = start + 145.0  # 5s margin
    
    best_edges = None
    best_n = None
    best_m = float('inf')
    
    # Try greedy set cover for various n
    # Start with small n where 2^n is manageable
    for n in range(10, 17):
        if time.time() > deadline - 10:
            break
            
        edges = _greedy_set_cover(n, deadline)
        if edges is not None:
            # Local search to reduce
            edges = _local_search_reduce(edges, n, deadline)
            m = len(edges)
            if m < best_m:
                best_m = m
                best_edges = edges
                best_n = n
                print(f"n={n}, edges={m}")
                
            if m <= 51:
                # Optimal or close, stop
                break
                
    # If greedy didn't find <= 51, try random search for specific n
    if best_m > 51:
        for n in [10, 11, 12, 13]:
            if time.time() > deadline - 5:
                break
            # Try to find a set of 51 edges
            edges = _random_search(n, 51, deadline)
            if edges is not None:
                best_m = 51
                best_edges = edges
                best_n = n
                break
                
    if best_edges is None:
        # Fallback: Complete hypergraph on 9 vertices
        # C(9,5) = 126 edges
        n = 9
        edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_edges = edges
        best_n = n

    # Convert to required format
    edges_list = [list(e) for e in best_edges]
    return {'n': best_n, 'edges': edges_list}


if __name__ == '__main__':
    result = construct()
    print(f"n={result['n']}, edges={len(result['edges'])}")
