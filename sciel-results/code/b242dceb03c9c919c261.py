"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations, product

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges.
    Uses random search for sparse non-2-colorable hypergraphs followed by greedy edge removal.
    """
    deadline = time.time() + 140
    sys.setrecursionlimit(100000)

    def is_2_colorable(n, edges):
        """
        Check if hypergraph is 2-colorable using backtracking with unit propagation.
        Returns True if 2-colorable, False otherwise.
        """
        if not edges:
            return True
        
        # Preprocess: remove duplicate edges and sort vertices in each edge
        edges = list(set(tuple(sorted(e)) for e in edges))
        if not edges:
            return True
        
        n_edges = len(edges)
        
        # Precompute vertex degrees for ordering
        deg = [0] * n
        for e in edges:
            for v in e:
                deg[v] += 1
        
        # Vertex ordering: highest degree first
        vertex_order = sorted(range(n), key=lambda v: -deg[v])
        
        # For each edge, track state: [count0, count1, uncolored]
        edge_state = [[0, 0, len(e)] for e in edges]
        color = [-1] * n
        
        # Map vertex to its index in vertex_order for faster access? 
        # Or just iterate through vertex_order.
        
        def propagate():
            """
            Unit propagation: 
            1. If an edge is monochromatic (all 0 or all 1), fail.
            2. If an edge has one uncolored vertex and the rest are 0, the last must be 1.
            3. If an edge has one uncolored vertex and the rest are 1, the last must be 0.
            """
            changed = True
            while changed:
                changed = False
                for i, e in enumerate(edges):
                    c0, c1, unc = edge_state[i]
                    if unc == 0:
                        if not (c0 > 0 and c1 > 0):
                            return False
                    elif unc == 1:
                        if c0 > 0 and c1 == 0:
                            # Find the uncolored vertex in this edge
                            for v in e:
                                if color[v] == -1:
                                    if color[v] == 1: # Already assigned? Shouldn't happen if state is consistent
                                        return False
                                    color[v] = 1
                                    edge_state[i] = [c0, c1 + 1, 0]
                                    changed = True
                                    break
                        elif c1 > 0 and c0 == 0:
                            for v in e:
                                if color[v] == -1:
                                    if color[v] == 0:
                                        return False
                                    color[v] = 0
                                    edge_state[i] = [c0 + 1, c1, 0]
                                    changed = True
                                    break
            return True

        def backtrack(pos):
            if pos == len(vertex_order):
                # All vertices assigned. Check if all edges are satisfied.
                # Propagation should have caught failures, but verify.
                for i in range(n_edges):
                    c0, c1, unc = edge_state[i]
                    if unc == 0 and not (c0 > 0 and c1 > 0):
                        return False
                return True
            
            v = vertex_order[pos]
            if color[v] != -1:
                return backtrack(pos + 1)
            
            # Try color 0
            color[v] = 0
            saved_states = []
            for i, e in enumerate(edges):
                if v in e:
                    c0, c1, unc = edge_state[i]
                    saved_states.append((i, (c0, c1, unc)))
                    edge_state[i] = [c0 + 1, c1, unc - 1]
            
            if propagate() and backtrack(pos + 1):
                return True
            
            # Undo color 0
            color[v] = -1
            for i, state in saved_states:
                edge_state[i] = list(state)
            
            # Try color 1
            color[v] = 1
            saved_states = []
            for i, e in enumerate(edges):
                if v in e:
                    c0, c1, unc = edge_state[i]
                    saved_states.append((i, (c0, c1, unc)))
                    edge_state[i] = [c0, c1 + 1, unc - 1]
            
            if propagate() and backtrack(pos + 1):
                return True
            
            # Undo color 1
            color[v] = -1
            for i, state in saved_states:
                edge_state[i] = list(state)
            
            return False
        
        return backtrack(0)

    def minimize_edges(n, edges, rng, deadline):
        """
        Greedy removal of edges that are redundant.
        """
        current_edges = edges[:]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle to avoid bias
            rng.shuffle(current_edges)
            for i in range(len(current_edges)):
                if time.time() > deadline:
                    break
                # Try removing edge i
                trial_edges = current_edges[:i] + current_edges[i+1:]
                if not is_2_colorable(n, trial_edges):
                    current_edges = trial_edges
                    changed = True
                    break # Restart loop with new size
        return current_edges

    best_edges = None
    best_n = None
    rng = random.Random(42)
    
    # Search over n and m
    # n=15..25, m=50..80
    for n in range(15, 26):
        max_m = min(80, len(list(combinations(range(n), 5))))
        if max_m < 50:
            continue
        for m in range(50, max_m + 1):
            if time.time() > deadline:
                break
            
            # Generate random hypergraph
            all_combos = list(combinations(range(n), 5))
            # Sample m edges
            edges = rng.sample(all_combos, m)
            
            # Check if non-2-colorable
            if not is_2_colorable(n, edges):
                # Found one. Minimize.
                min_edges = minimize_edges(n, edges, rng, deadline)
                
                if best_edges is None or len(min_edges) < len(best_edges):
                    best_edges = min_edges
                    best_n = n
                    # If we reach 51, it's very good. We can continue to see if we can go lower, but 51 is likely the floor.
                    # If we find 50 or less, that's a breakthrough.
                    
    # Fallback: if random search failed, try the dense K9 method
    if best_edges is None:
        n = 9
        all_edges = list(combinations(range(n), 5))
        current_edges = all_edges[:]
        if not is_2_colorable(n, current_edges):
            best_edges = minimize_edges(n, current_edges, rng, deadline)
            best_n = n
    
    if best_edges is None:
        # Absolute fallback: return a valid one (K9)
        n = 9
        edges = list(combinations(range(n), 5))
        return {'n': n, 'edges': [list(e) for e in edges]}
        
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}

# EVOLVE-BLOCK-END
