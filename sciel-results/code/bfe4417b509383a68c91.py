"""Few edges in 5-uniform hypergraphs without Property B (m(5))"""
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills after 150s, keep safety margin
_DEADLINE = 140.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _check_time():
    return _elapsed() > _DEADLINE

def _is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph has a proper 2-coloring
    (no edge is monochromatic) using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    # Pre-process edges for faster access
    # We use a simple list of tuples for speed
    edge_tuples = [tuple(e) for e in edges]
    
    colors = [-1] * n
    
    def propagate():
        """Apply unit propagation. Returns False on contradiction, True otherwise."""
        changed = True
        while changed:
            changed = False
            for e in edge_tuples:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    cv = colors[v]
                    if cv == 0:
                        c0 += 1
                    elif cv == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                # Contradiction: all 5 same color
                if c0 == 5 or c1 == 5:
                    return False
                
                # Unit propagation: 4 of one color forces the 5th
                if c0 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 1
                        changed = True
                    elif colors[v] == 0:
                        return False
                elif c1 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 0
                        changed = True
                    elif colors[v] == 1:
                        return False
        return True
    
    # Variable ordering heuristic: pick the most constrained unassigned vertex
    # But for simplicity and speed in this small n, fixed order or MRV is better.
    # Let's use a simple recursive backtracking with MRV (Minimum Remaining Values)
    # Actually, fixed order is often faster for small n if propagation is strong.
    # Let's stick to fixed order 0..n-1 but with strong propagation.
    
    def backtrack(idx):
        """Try to assign vertex idx and beyond. Returns True if satisfiable."""
        if idx >= n:
            # Check all edges are not monochromatic
            # Propagation should have caught most, but double check
            for e in edge_tuples:
                c0 = 0
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                if c0 == 5 or c0 == 0:
                    return False
            return True
        
        # Try both colors
        for c in (0, 1):
            colors[idx] = c
            saved = colors[idx:]
            if propagate():
                if backtrack(idx + 1):
                    return True
            colors[idx:] = saved
            if _check_time():
                return True  # Assume satisfiable to avoid timeout issues
        
        return False
    
    return backtrack(0)

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-coloring, minimizing edges.
    Returns {'n': n, 'edges': [[v1,...,v5], ...]}
    """
    # Start with a known non-2-colorable hypergraph
    # Complete 5-uniform hypergraph on 9 vertices has 126 edges.
    # We know m(5) <= 51.
    # Let's start with a smaller seed if possible, or just start with n=9 and prune.
    
    n = 9
    # Generate all possible edges
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Randomize order to help escape local optima
    random.shuffle(all_edges)
    
    # Start with a subset that is likely non-2-colorable
    # A random subset of size ~60 might be non-2-colorable.
    # Let's start with all edges and prune.
    current_edges = all_edges[:]
    
    # Verify start is non-2-colorable
    if _is_2_colorable(n, current_edges):
        # If it is (shouldn't be for K_5^9), we need a different seed
        # But K_5^9 is definitely not 2-colorable.
        pass
        
    best_edges = current_edges[:]
    
    # Simulated Annealing / Randomized Greedy Removal
    # We try to remove edges one by one.
    # If removing an edge makes it 2-colorable, we try to swap it with another edge
    # or just keep it.
    
    # To be faster, we can use a "greedy with restarts" or "local search" approach.
    # Let's use a randomized local search:
    # 1. Pick a random edge to remove.
    # 2. If result is non-2-colorable, accept.
    # 3. If result is 2-colorable, reject.
    # 4. Occasionally, add a random edge back (to escape local minima).
    
    current_edges_set = set(map(tuple, current_edges))
    
    # To speed up, we can pre-compute some things? No, n is small.
    
    # Let's try a more aggressive heuristic:
    # Randomized greedy: shuffle edges, try to remove each one.
    # If we get stuck, restart with a different shuffle.
    
    best_size = len(current_edges)
    
    for restart in range(50):
        if _check_time():
            break
            
        # Shuffle edges
        random.shuffle(current_edges)
        current_edges_set = set(map(tuple, current_edges))
        
        # Try to remove edges
        i = 0
        while i < len(current_edges) and not _check_time():
            edge_to_remove = current_edges[i]
            candidate = current_edges[:i] + current_edges[i+1:]
            
            if not _is_2_colorable(n, candidate):
                # Accept removal
                current_edges = candidate
                # Do not increment i, because the next edge is now at index i
            else:
                # Reject removal, try next
                i += 1
                
        if len(current_edges) < best_size:
            best_size = len(current_edges)
            best_edges = current_edges[:]
            
            # If we reach 51, we can try to optimize further or stop?
            # 51 is the known minimum. If we hit 51, we can stop early if we are confident.
            # But maybe we can go lower? No, m(5) is at least 29? 
            # "best known construction has 51 edges and every 5-uniform hypergraph with at most 28 edges is 2-colourable."
            # So 51 is the target. If we find 51, we can return it.
            if best_size <= 51:
                break

    return {'n': n, 'edges': best_edges}
