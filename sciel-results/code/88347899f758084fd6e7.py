"""Few edges in 5-uniform hypergraphs without Property B.

Strategy:
1. Start with the known 51-edge construction (even 5-subsets of 10 vertices).
2. Use local search (edge deletion) to minimize the number of edges while
   maintaining the property that the hypergraph is NOT 2-colorable.
3. Return the best hypergraph found before the time deadline.
"""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = 150.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _edges_10_even():
    """All 5-subsets of {0..9} with an even number of even-indexed vertices."""
    evenset = frozenset((0, 2, 4, 6, 8))
    edges = []
    for c in combinations(range(10), 5):
        if sum(1 for v in c if v in evenset) % 2 == 0:
            edges.append(list(c))
    return edges


def _is_2_colorable_fast(n, edges):
    """
    Fast check for 2-colorability using backtracking with unit propagation.
    Returns True if 2-colorable, False if not.
    Optimized for small n (e.g., 10) and moderate edge counts.
    """
    m = len(edges)
    if m == 0:
        return True
    
    # Precompute incidence: vertex -> list of edge indices
    v_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        for v in e:
            v_edges[v].append(ei)
            
    # State
    color = [-1] * n
    # edge_state[ei] = [count0, count1, uncounted]
    edge_state = [[0, 0, 5] for _ in range(m)]
    
    # Order vertices by degree (descending) for better pruning
    v_order = sorted(range(n), key=lambda v: -len(v_edges[v]))
    
    def propagate():
        """
        Apply unit propagation.
        Returns False if contradiction (monochromatic edge or forced conflict).
        Updates color and edge_state in place.
        """
        changed = True
        while changed:
            changed = False
            # Check for immediate contradictions and force assignments
            for ei in range(m):
                c0, c1, un = edge_state[ei]
                if c0 == 5 or c1 == 5:
                    return False
                if un == 0:
                    # Edge is full, must be bichromatic (checked above)
                    pass
                elif c0 == 4:
                    # Must force last vertex to color 1
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 1
                            edge_state[ei][1] += 1
                            edge_state[ei][2] -= 1
                            # Update other edges containing v
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    edge_state[e2][1] += 1
                                    edge_state[e2][2] -= 1
                            changed = True
                            break
                elif c1 == 4:
                    # Must force last vertex to color 0
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 0
                            edge_state[ei][0] += 1
                            edge_state[ei][2] -= 1
                            # Update other edges containing v
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    edge_state[e2][0] += 1
                                    edge_state[e2][2] -= 1
                            changed = True
                            break
        return True

    def backtrack(idx):
        if idx == n:
            return True
        
        # Find next uncolored vertex in order
        while idx < n and color[v_order[idx]] != -1:
            idx += 1
            
        if idx == n:
            return True
            
        v = v_order[idx]
        
        # Snapshot state for backtracking
        # We need to revert color and edge_state
        # To avoid deep copying, we can use a stack of changes, 
        # but for n=10, simple copying is fast enough.
        color_snap = color[:]
        edge_snap = [list(s) for s in edge_state]
        
        for c in (0, 1):
            color[v] = c
            # Apply color to edges
            valid = True
            for ei in v_edges[v]:
                edge_state[ei][c] += 1
                edge_state[ei][2] -= 1
                if edge_state[ei][0] == 5 or edge_state[ei][1] == 5:
                    valid = False
                    break
            
            if valid and propagate():
                if backtrack(idx + 1):
                    return True
            
            # Revert
            color[:] = color_snap
            for i in range(m):
                edge_state[i] = edge_snap[i]
                
        return False

    return backtrack(0)


def construct():
    if _elapsed() > _DEADLINE - 10:
        # Safety fallback: return the known 51-edge construction
        return {'n': 10, 'edges': _edges_10_even()}

    # Start with the 51-edge construction
    current_edges = _edges_10_even()
    n = 10
    
    # Local search: try to remove edges
    # We iterate, trying to remove one edge at a time if the graph remains non-2-colorable
    # We repeat passes until no more edges can be removed or time runs out
    
    # Set random seed for reproducibility if needed, but random is fine
    rng = random.Random(42)
    
    while _elapsed() < _DEADLINE - 5:
        removed_any = False
        # Shuffle to avoid bias
        indices = list(range(len(current_edges)))
        rng.shuffle(indices)
        
        for idx in indices:
            if _elapsed() > _DEADLINE - 5:
                break
                
            # Try removing edge at idx
            e_to_remove = current_edges[idx]
            new_edges = current_edges[:idx] + current_edges[idx+1:]
            
            # Check if new_edges is still NOT 2-colorable
            # i.e., is_2_colorable should return False
            if not _is_2_colorable_fast(n, new_edges):
                # It is still non-2-colorable, so we can keep it removed
                current_edges = new_edges
                removed_any = True
        
        if not removed_any:
            # No more edges could be removed in this pass
            break

    return {'n': n, 'edges': current_edges}
# EVOLVE-BLOCK-END
