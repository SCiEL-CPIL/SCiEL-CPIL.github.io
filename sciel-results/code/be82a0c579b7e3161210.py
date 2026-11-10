"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_non_2colorable(n, edges):
    """
    Check if the hypergraph is NOT 2-colorable.
    Returns True if no proper 2-coloring exists.
    """
    if not edges:
        return False
        
    edge_set = [tuple(sorted(e)) for e in edges]
    edge_set = list(set(edge_set))
    if not edge_set:
        return False
        
    m = len(edge_set)
    vert_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_set):
        for v in e:
            vert_to_edges[v].append(i)
            
    def solve():
        colors = [-1] * n
        count_0 = [0] * m
        count_1 = [0] * m
        
        def backtrack():
            # Unit Propagation
            changed = True
            while changed:
                changed = False
                for i, e in enumerate(edge_set):
                    c0 = count_0[i]
                    c1 = count_1[i]
                    if c0 == 5 or c1 == 5:
                        return False
                    if c0 == 4 and c1 == 0:
                        # Find unassigned vertex in this edge
                        for v in e:
                            if colors[v] == -1:
                                colors[v] = 1
                                count_1[i] += 1
                                for j in vert_to_edges[v]:
                                    if j != i:
                                        count_1[j] += 1
                                changed = True
                                break
                    elif c1 == 4 and c0 == 0:
                        for v in e:
                            if colors[v] == -1:
                                colors[v] = 0
                                count_0[i] += 1
                                for j in vert_to_edges[v]:
                                    if j != i:
                                        count_0[j] += 1
                                changed = True
                                break
                                
            # Check if all assigned
            if all(c != -1 for c in colors):
                return True # A valid coloring was found
            
            # Branch
            # Heuristic: Pick vertex involved in most "critical" edges (4 assigned)
            unassigned = [v for v in range(n) if colors[v] == -1]
            if not unassigned:
                return True
                
            best_v = -1
            max_crit = 0
            for v in unassigned:
                crit = 0
                for j in vert_to_edges[v]:
                    c0 = count_0[j]
                    c1 = count_1[j]
                    if c0 + c1 == 4:
                        crit += 1
                if crit > max_crit:
                    max_crit = crit
                    best_v = v
                    
            if best_v == -1:
                best_v = unassigned[0]
                
            v = best_v
            
            # Save state
            old_colors = colors[:]
            old_c0 = count_0[:]
            old_c1 = count_1[:]
            
            # Try color 0
            colors[v] = 0
            for j in vert_to_edges[v]:
                count_0[j] += 1
            if backtrack():
                return True
                
            # Restore state
            colors[:] = old_colors
            count_0[:] = old_c0
            count_1[:] = old_c1
            
            # Try color 1
            colors[v] = 1
            for j in vert_to_edges[v]:
                count_1[j] += 1
            if backtrack():
                return True
                
            return False
            
        return backtrack()
        
    return not solve()

def construct():
    """
    Return a non-2-colorable 5-uniform hypergraph with minimal edges.
    """
    deadline = time.time() + 140  # Safety margin
    
    best_m = 127  # Fallback is 126 edges
    best_graph = None
    
    # Fallback: Complete 5-uniform hypergraph on 9 vertices
    fallback_n = 9
    fallback_edges = [list(e) for e in combinations(range(fallback_n), 5)]
    
    # Search for smaller constructions
    # Try n from 10 to 20
    for n in range(10, 21):
        if time.time() > deadline - 10:
            break
            
        all_edges = list(combinations(range(n), 5))
        if len(all_edges) < 10:
            continue
            
        # Try m from 10 to 50
        for m in range(10, 51):
            if m >= best_m:
                break
            if len(all_edges) < m:
                break
                
            # Try random samples
            found = False
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                edges = [list(e) for e in random.sample(all_edges, m)]
                if _is_non_2colorable(n, edges):
                    best_m = m
                    best_graph = {'n': n, 'edges': edges}
                    found = True
                    break
                    
            if found:
                # Continue to see if we can find smaller m for this n or other n
                # Since m is increasing, we found the smallest m for this n in this run
                # But we should check if we can find even smaller m?
                # We iterate m from 10 upwards. So if we found one at m=20,
                # we have already checked m=10..19 and didn't find one (or didn't sample enough).
                # We should probably restart m loop for this n with more samples?
                # Or just accept this and move to next n.
                pass

    if best_graph:
        return best_graph
    else:
        return {'n': fallback_n, 'edges': fallback_edges}
# EVOLVE-BLOCK-END
