"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable_fast(edges, n):
    """Fast check using bitmasks. Returns True if colorable, False if not."""
    # Represent coloring as a bitmask of n bits
    # For each edge, check if it's monochromatic
    # Use backtracking with unit propagation on bitmasks
    
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    def check_coloring(color_mask):
        """Check if a full coloring (color_mask) is valid (no monochromatic edge)."""
        for em in edge_masks:
            # Edge is monochromatic if all vertices have same color
            # Count vertices in edge that are color 0: em & ~color_mask
            # Count vertices in edge that are color 1: em & color_mask
            c0 = bin(em & ~color_mask).count('1')
            c1 = bin(em & color_mask).count('1')
            if c0 == 5 or c1 == 5:
                return False
        return True
    
    # Backtracking with unit propagation
    # State: partial coloring as (assigned_mask, color_mask)
    # assigned_mask: which vertices are assigned
    # color_mask: which assigned vertices are color 1
    
    import sys
    sys.setrecursionlimit(10000)
    
    def propagate(assigned_mask, color_mask):
        """Apply unit propagation. Returns None if contradiction, else (assigned_mask, color_mask)."""
        changed = True
        while changed:
            changed = False
            for em in edge_masks:
                # Get vertices in this edge
                # Count assigned vertices of each color
                assigned_in_edge = em & assigned_mask
                c0 = bin(assigned_in_edge & ~color_mask).count('1')
                c1 = bin(assigned_in_edge & color_mask).count('1')
                unassigned_in_edge = em & ~assigned_mask
                num_unassigned = bin(unassigned_in_edge).count('1')
                
                # Check for violation
                if c0 == 5 or c1 == 5:
                    return None
                
                # Unit propagation
                if c0 == 0 and num_unassigned == 1:
                    # The one unassigned must be color 0
                    v_bit = unassigned_in_edge
                    if color_mask & v_bit:
                        return None  # Already assigned color 1, contradiction
                    assigned_mask |= v_bit
                    changed = True
                    break
                elif c1 == 0 and num_unassigned == 1:
                    # The one unassigned must be color 1
                    v_bit = unassigned_in_edge
                    if not (color_mask & v_bit):
                        return None  # Already assigned color 0, contradiction
                    assigned_mask |= v_bit
                    color_mask |= v_bit
                    changed = True
                    break
                elif c0 == 4 and num_unassigned == 1:
                    # The one unassigned must be color 1
                    v_bit = unassigned_in_edge
                    if not (color_mask & v_bit):
                        return None
                    assigned_mask |= v_bit
                    color_mask |= v_bit
                    changed = True
                    break
                elif c1 == 4 and num_unassigned == 1:
                    # The one unassigned must be color 0
                    v_bit = unassigned_in_edge
                    if color_mask & v_bit:
                        return None
                    assigned_mask |= v_bit
                    changed = True
                    break
        
        return (assigned_mask, color_mask)
    
    def backtrack(assigned_mask, color_mask, unassigned_count):
        """Try to complete. Returns True if colorable."""
        result = propagate(assigned_mask, color_mask)
        if result is None:
            return False
        assigned_mask, color_mask = result
        unassigned_count = n - bin(assigned_mask).count('1')
        
        if unassigned_count == 0:
            return True
        
        # Find an unassigned vertex
        # Heuristic: pick vertex in most constrained edge
        unassigned_mask = ~assigned_mask & ((1 << n) - 1)
        
        # Pick first unassigned vertex for simplicity
        v = 0
        while not (unassigned_mask & (1 << v)):
            v += 1
        
        v_bit = 1 << v
        
        # Try color 0
        new_assigned = assigned_mask | v_bit
        new_color = color_mask
        if backtrack(new_assigned, new_color, unassigned_count - 1):
            return True
        
        # Try color 1
        new_assigned = assigned_mask | v_bit
        new_color = color_mask | v_bit
        if backtrack(new_assigned, new_color, unassigned_count - 1):
            return True
        
        return False
    
    return backtrack(0, 0, n)

def _search_hitting_set(n, time_limit=120):
    """Search for minimum non-2-colorable 5-uniform hypergraph using hitting set approach.
    
    Idea: A hypergraph is non-2-colorable iff it contains, for every 2-coloring,
    at least one monochromatic edge. So we need to find a minimum set of edges
    that "hits" all colorings (i.e., for each coloring, at least one edge is monochromatic).
    
    We'll use a greedy + local search approach:
    1. Start with random edges
    2. Find a coloring that avoids all edges (if exists)
    3. Add an edge that is monochromatic under that coloring
    4. Repeat until no coloring exists
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    all_edges = list(combinations(range(n), 5))
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    best_edges = None
    
    # Try multiple random starting points
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        random.seed(trial * 7919 + 13)
        
        # Start with a random small set of edges
        current_edges = set()
        
        # Add random edges to start
        for _ in range(15):
            e = random.choice(all_edges)
            current_edges.add(e)
        
        # Iteratively find colorings and add edges
        for iteration in range(100):
            if time.time() > deadline - 5:
                break
            
            # Find a 2-coloring that avoids all current edges
            # Use backtracking
            coloring = _find_coloring(list(current_edges), n, timeout=2)
            
            if coloring is None:
                # No coloring exists - we have a non-2-colorable hypergraph!
                if best_edges is None or len(current_edges) < len(best_edges):
                    best_edges = list(current_edges)
                break
            
            # Add an edge that is monochromatic under this coloring
            # Find all monochromatic edges under this coloring
            mono_edges = []
            for e in all_edges:
                # Check if e is monochromatic under coloring
                colors = [coloring[v] for v in e]
                if len(set(colors)) == 1:
                    mono_edges.append(e)
            
            if not mono_edges:
                break
            
            # Add a random monochromatic edge (prefer ones not already in current_edges)
            new_edges = [e for e in mono_edges if e not in current_edges]
            if new_edges:
                e = random.choice(new_edges)
            else:
                e = random.choice(mono_edges)
            current_edges.add(e)
        
        # If we found a valid hypergraph, record it
        if len(current_edges) < 200:
            if best_edges is None or len(current_edges) < len(best_edges):
                best_edges = list(current_edges)
    
    return best_edges

def _find_coloring(edges, n, timeout=5):
    """Find a 2-coloring that avoids all edges (no monochromatic edge).
    Returns None if no such coloring exists, else a list of n colors."""
    start_time = time.time()
    
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    def check_partial(assigned_mask, color_mask):
        """Check if partial assignment is consistent."""
        for em in edge_masks:
            assigned_in_edge = em & assigned_mask
            c0 = bin(assigned_in_edge & ~color_mask).count('1')
            c1 = bin(assigned_in_edge & color_mask).count('1')
            if c0 == 5 or c1 == 5:
                return False
        return True
    
    def backtrack(assigned_mask, color_mask):
        if time.time() - start_time > timeout:
            return None
        
        # Check consistency
        if not check_partial(assigned_mask, color_mask):
            return None
        
        # Check if all assigned
        if assigned_mask == (1 << n) - 1:
            # Build coloring
            coloring = []
            for v in range(n):
                coloring.append(1 if (color_mask & (1 << v)) else 0)
            return coloring
        
        # Find unassigned vertex
        unassigned = ~assigned_mask & ((1 << n) - 1)
        if unassigned == 0:
            return None
        
        # Pick first unassigned
        v = 0
        while not (unassigned & (1 << v)):
            v += 1
        
        v_bit = 1 << v
        
        # Try color 0
        result = backtrack(assigned_mask | v_bit, color_mask)
        if result is not None:
            return result
        
        # Try color 1
        result = backtrack(assigned_mask | v_bit, color_mask | v_bit)
        if result is not None:
            return result
        
        return None
    
    return backtrack(0, 0)

def _local_search_improve(edges, n, time_limit=30):
    """Try to reduce the number of edges while maintaining non-2-colorability."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    current = list(edges)
    
    # Try removing edges one at a time
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(len(current)):
            if time.time() > deadline - 2:
                break
            
            # Try removing edge i
            candidate = current[:i] + current[i+1:]
            
            # Check if still non-2-colorable
            coloring = _find_coloring(candidate, n, timeout=3)
            
            if coloring is None:
                # Still non-2-colorable, remove the edge
                current = candidate
                improved = True
                break
    
    return current

def construct():
    """Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges."""
    start_time = time.time()
    total_deadline = start_time + 140  # Leave 10s margin
    
    best_result = None
    
    # Try different n values
    for n in [9, 10, 11, 12, 13]:
        if time.time() > total_deadline - 20:
            break
        
        # Search for non-2-colorable hypergraph
        result = _search_hitting_set(n, time_limit=30)
        
        if result is not None:
            # Try to improve with local search
            if time.time() < total_deadline - 10:
                improved = _local_search_improve(result, n, time_limit=10)
                result = improved
            
            if best_result is None or len(result) < len(best_result['edges']):
                best_result = {'n': n, 'edges': result}
    
    # Verify the best result
    if best_result is not None:
        if time.time() < total_deadline - 5:
            coloring = _find_coloring(best_result['edges'], best_result['n'], timeout=10)
            if coloring is not None:
                # It's colorable, so invalid
                best_result = None
    
    if best_result is not None:
        return best_result
    
    # Fallback: complete hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
