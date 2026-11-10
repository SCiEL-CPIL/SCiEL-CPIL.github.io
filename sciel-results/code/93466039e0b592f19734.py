"""
Strategy: Deterministic minimization of complete hypergraphs K^5_n using a fast bitmask-based
2-colorability checker. This guarantees finding a minimal non-2-colorable hypergraph (with
respect to edge removal) for each n, which is likely to be significantly smaller than the
126-edge fallback or random results, potentially approaching the 51-edge optimum.
"""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _is_2_colorable_fast(n, edges):
    """
    Check if hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if colorable, False otherwise.
    Optimized for small n and moderate edge counts using bitmasks.
    """
    if not edges:
        return True
    
    m = len(edges)
    
    # Precompute edge bitmasks
    # Each edge is a bitmask of its 5 vertices
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # Incidence: for each vertex, which edges contain it
    incidence = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            incidence[v].append(i)
            
    # Order variables by degree (most constrained first)
    var_order = sorted(range(n), key=lambda v: len(incidence[v]), reverse=True)
    
    # State
    assignments = [-1] * n
    edge_0s = [0] * m
    edge_1s = [0] * m
    
    # Unit propagation helper
    def propagate():
        """
        Apply unit propagation.
        Returns False if a contradiction is found.
        Returns True if consistent.
        Modifies assignments, edge_0s, edge_1s in place.
        """
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if assignments[v] != -1:
                    continue
                forced_val = -1
                for ei in incidence[v]:
                    c0 = edge_0s[ei]
                    c1 = edge_1s[ei]
                    if c0 == 4:
                        forced_val = 1
                        break
                    if c1 == 4:
                        forced_val = 0
                        break
                
                if forced_val != -1:
                    assignments[v] = forced_val
                    changed = True
                    for ei in incidence[v]:
                        if forced_val == 0:
                            edge_0s[ei] += 1
                            if edge_0s[ei] == 5:
                                return False
                        else:
                            edge_1s[ei] += 1
                            if edge_1s[ei] == 5:
                                return False
        return True

    def recurse():
        # Find the next unassigned variable in var_order
        v = -1
        for i in range(n):
            if assignments[var_order[i]] == -1:
                v = var_order[i]
                break
        
        if v == -1:
            return True
            
        # Try 0
        assignments[v] = 0
        valid = True
        for ei in incidence[v]:
            edge_0s[ei] += 1
            if edge_0s[ei] == 5:
                valid = False
                break
        if valid:
            if propagate() and recurse():
                return True
        for ei in incidence[v]:
            edge_0s[ei] -= 1
        assignments[v] = -1
        
        # Try 1
        assignments[v] = 1
        valid = True
        for ei in incidence[v]:
            edge_1s[ei] += 1
            if edge_1s[ei] == 5:
                valid = False
                break
        if valid:
            if propagate() and recurse():
                return True
        for ei in incidence[v]:
            edge_1s[ei] -= 1
        assignments[v] = -1
        
        return False
        
    return recurse()

def construct():
    deadline = time.time() + 145
    
    best_count = float('inf')
    best_edges = None
    best_n = 9
    
    # Try n=9 first (most likely to yield small minimal graphs)
    # Then n=10, 11... if time permits
    
    for n in range(9, 13):
        if time.time() > deadline - 10:
            break
            
        all_edges = list(itertools.combinations(range(n), 5))
        
        # Start with the complete hypergraph
        current_edges = all_edges[:]
        
        # Shuffle to randomize removal order
        random.seed(42 + n)
        random.shuffle(current_edges)
        
        # Greedy removal: remove an edge if the graph remains non-colorable
        i = 0
        while i < len(current_edges) and time.time() < deadline - 5:
            test_edges = current_edges[:i] + current_edges[i+1:]
            if _is_2_colorable_fast(n, test_edges):
                # If it becomes colorable, we cannot remove it.
                i += 1
            else:
                # If it remains non-colorable, remove it.
                current_edges.pop(i)
        
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            best_n = n
        
        # If we've already found a very small graph, we might stop early
        # but let's try a few more n values if time permits
        
    if best_edges is None:
        # Fallback: return a known non-colorable graph (K9^5)
        n = 9
        best_edges = list(itertools.combinations(range(n), 5))
        best_n = n
        best_count = len(best_edges)
        
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
