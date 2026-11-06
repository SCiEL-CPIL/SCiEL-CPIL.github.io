"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations, product

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges.
    
    Strategy:
    1. Start with a known small non-2-colorable 5-uniform hypergraph.
    2. Use fast 2-colorability check via backtracking with unit propagation.
    3. Local search: greedy edge removal + edge swaps to escape local minima.
    4. Verify non-2-colorability exactly.
    
    Deadline guard ensures we finish within 150s wall-clock.
    """
    deadline = time.time() + 135  # 15s safety margin
    
    def is_2_colorable_fast(n, edges):
        """
        Fast check if hypergraph is 2-colorable using backtracking with unit propagation.
        Returns True if 2-colorable, False otherwise.
        Uses vertex-based assignment with degree-ordered vertices.
        """
        if not edges:
            return True
        
        # Deduplicate and sort edges
        edges = list(set(tuple(sorted(e)) for e in edges))
        n_edges = len(edges)
        if n_edges == 0:
            return True
        
        # Build incidence: for each vertex, list of edge indices containing it
        vertex_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(i)
        
        # Vertex ordering: highest degree first
        deg = [len(vertex_edges[v]) for v in range(n)]
        vertex_order = sorted(range(n), key=lambda v: -deg[v])
        
        # State: color[v] in {-1, 0, 1}, edge_state[i] = [c0, c1, unc]
        color = [-1] * n
        edge_state = [[0, 0, 5] for _ in range(n_edges)]
        
        sys.setrecursionlimit(100000)
        
        def propagate():
            """Unit propagation: force colors when an edge has 4 of one color."""
            changed = True
            while changed:
                changed = False
                for i in range(n_edges):
                    c0, c1, unc = edge_state[i]
                    if unc == 0:
                        if not (c0 > 0 and c1 > 0):
                            return False  # monochromatic
                    elif unc == 1:
                        if c0 > 0 and c1 == 0:
                            # Last vertex must be color 1
                            e = edges[i]
                            for v in e:
                                if color[v] == -1:
                                    color[v] = 1
                                    edge_state[i] = [c0, 1, 0]
                                    changed = True
                                    break
                        elif c1 > 0 and c0 == 0:
                            # Last vertex must be color 0
                            e = edges[i]
                            for v in e:
                                if color[v] == -1:
                                    color[v] = 0
                                    edge_state[i] = [1, c1, 0]
                                    changed = True
                                    break
            return True
        
        def backtrack(pos):
            if pos == n:
                # All vertices colored; verify all edges are bichromatic
                for i in range(n_edges):
                    c0, c1, unc = edge_state[i]
                    if unc == 0 and not (c0 > 0 and c1 > 0):
                        return False
                return True
            
            v = vertex_order[pos]
            if color[v] != -1:
                return backtrack(pos + 1)
            
            for c in (0, 1):
                color[v] = c
                # Update edge states for edges containing v
                saved = []
                for i in vertex_edges[v]:
                    c0, c1, unc = edge_state[i]
                    saved.append((i, c0, c1, unc))
                    if c == 0:
                        edge_state[i] = [c0 + 1, c1, unc - 1]
                    else:
                        edge_state[i] = [c0, c1 + 1, unc - 1]
                
                if propagate() and backtrack(pos + 1):
                    return True
                
                # Undo
                color[v] = -1
                for i, c0, c1, unc in saved:
                    edge_state[i] = [c0, c1, unc]
            
            return False
        
        return backtrack(0)
    
    # Start with complete 5-uniform hypergraph on 9 vertices
    n = 9
    all_edges = list(combinations(range(n), 5))
    current_edges = list(all_edges)
    
    # Verify initial hypergraph is not 2-colorable
    if is_2_colorable_fast(n, current_edges):
        return {'n': n, 'edges': [list(e) for e in current_edges]}
    
    best_edges = current_edges[:]
    best_count = len(best_edges)
    
    rng = random.Random(42)
    
    # Phase 1: Greedy edge removal
    # Repeatedly try to remove a random edge; keep if still non-2-colorable
    removal_passes = 0
    while time.time() < deadline - 30:  # reserve 30s for phase 2
        progress = False
        for _ in range(200):
            if time.time() > deadline - 30:
                break
            if not current_edges:
                break
            idx = rng.randrange(len(current_edges))
            trial = current_edges[:idx] + current_edges[idx+1:]
            if not is_2_colorable_fast(n, trial):
                current_edges = trial
                progress = True
                if len(current_edges) < best_count:
                    best_edges = current_edges[:]
                    best_count = len(best_edges)
        removal_passes += 1
        if not progress:
            break
    
    # Phase 2: Local search with edge swaps
    # Try to swap an existing edge for a new edge (from the complement) to reduce count
    # or enable further removals
    all_possible = set(all_edges)
    current_set = set(tuple(sorted(e)) for e in current_edges)
    
    swap_passes = 0
    while time.time() < deadline - 5:
        progress = False
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            if not current_edges:
                break
            
            # Pick a random existing edge to potentially remove
            idx = rng.randrange(len(current_edges))
            edge_to_remove = current_edges[idx]
            edge_to_remove_t = tuple(sorted(edge_to_remove))
            
            # Try simple removal first
            trial = current_edges[:idx] + current_edges[idx+1:]
            if not is_2_colorable_fast(n, trial):
                current_edges = trial
                current_set.discard(edge_to_remove_t)
                progress = True
                if len(current_edges) < best_count:
                    best_edges = current_edges[:]
                    best_count = len(best_edges)
                continue
            
            # Try swap: remove this edge, add a random new edge
            # New edge must be from all_possible but not in current_set
            available = all_possible - current_set
            if available:
                new_edge = tuple(sorted(rng.choice(list(available))))
                trial_swapped = current_edges[:idx] + current_edges[idx+1:] + [list(new_edge)]
                if not is_2_colorable_fast(n, trial_swapped):
                    # Accept the swap (same count, but different structure may enable removals)
                    current_edges = trial_swapped
                    current_set.discard(edge_to_remove_t)
                    current_set.add(new_edge)
                    progress = True
            
            # After a swap, try to remove another edge
            if progress and current_edges:
                idx2 = rng.randrange(len(current_edges))
                trial2 = current_edges[:idx2] + current_edges[idx2+1:]
                if not is_2_colorable_fast(n, trial2):
                    current_edges = trial2
                    current_set.discard(tuple(sorted(current_edges[idx2])))
                    if len(current_edges) < best_count:
                        best_edges = current_edges[:]
                        best_count = len(best_edges)
        
        swap_passes += 1
        if not progress:
            # Try restarting from best if stuck
            if swap_passes % 5 == 0:
                current_edges = best_edges[:]
                current_set = set(tuple(sorted(e)) for e in current_edges)
            else:
                break
    
    return {'n': n, 'edges': [list(e) for e in best_edges]}

# EVOLVE-BLOCK-END
