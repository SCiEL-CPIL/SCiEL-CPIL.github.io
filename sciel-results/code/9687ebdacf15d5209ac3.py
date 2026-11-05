"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_non_2colorable(n, edges, deadline):
    """Check if hypergraph is NOT 2-colorable using backtracking with unit propagation."""
    edge_set = [set(e) for e in edges]
    color = [-1] * n  # -1 unassigned, 0 or 1
    
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_set):
        for v in e:
            vert_edges[v].append(i)
            
    # Optimization: Precompute edge indices for each vertex
    # Actually, propagate needs to iterate edges.
    
    def propagate():
        """Unit propagation: if 4 of 5 in an edge same color, force 5th."""
        changed = True
        while changed:
            changed = False
            for e in edge_set:
                c0 = c1 = un = 0
                for v in e:
                    if color[v] == 0: c0 += 1
                    elif color[v] == 1: c1 += 1
                    else: un += 1
                if c0 == 4 and un == 1:
                    for v in e:
                        if color[v] == -1:
                            color[v] = 1
                            changed = True
                elif c1 == 4 and un == 1:
                    for v in e:
                        if color[v] == -1:
                            color[v] = 0
                            changed = True
                if c0 == 5 or c1 == 5:
                    return False
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return 'timeout'
        while idx < n and color[idx] != -1:
            idx += 1
        if idx >= n:
            # All assigned
            for e in edge_set:
                c0 = sum(1 for v in e if color[v] == 0)
                if c0 == 5 or c0 == 0:
                    return False
            return True # Found a valid 2-coloring
        
        # Save state
        saved_color = color[:]
        
        # Try both colors
        for c in [0, 1]:
            color[idx] = c
            if propagate():
                result = backtrack(idx + 1)
                if result == 'timeout':
                    color[:] = saved_color
                    return 'timeout'
                if result == True:
                    color[:] = saved_color
                    return True
            color[:] = saved_color
        return False
    
    result = backtrack(0)
    if result == 'timeout':
        return None
    if result == True:
        return False # It IS 2-colorable
    else:
        return True   # It is NOT 2-colorable

def construct():
    deadline = time.time() + 145
    
    # Start with K_9^5
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_n = 9
    
    # Greedy edge removal
    current_edges = best_edges[:]
    current_n = best_n
    removed_count = 0
    
    # Try to remove edges from K_9^5
    # Shuffle to avoid bias
    edge_indices = list(range(len(current_edges)))
    random.seed(42)
    random.shuffle(edge_indices)
    
    for idx in edge_indices:
        if time.time() > deadline:
            break
        if not current_edges:
            break
        # Try removing edge at idx
        edge_to_remove = current_edges[idx]
        temp_edges = current_edges[:idx] + current_edges[idx+1:]
        
        # Check if still non-2-colorable
        # Use a tighter deadline for internal checks?
        # If it's fast, use deadline.
        is_non_2col = _is_non_2colorable(current_n, temp_edges, deadline)
        
        if is_non_2col is True:
            current_edges = temp_edges
            # Update best if better
            if len(current_edges) < len(best_edges):
                best_edges = current_edges[:]
                best_n = current_n
                
    # Also try random constructions on larger n
    for n in range(10, 15):
        if time.time() > deadline:
            break
        all_possible = list(combinations(range(n), 5))
        # Try to find a sparse non-2-colorable one
        # Random sampling is weak, but let's try a few
        for _ in range(10):
            if time.time() > deadline:
                break
            # Sample roughly 1/3 of edges? Or specific counts?
            # Let's try counts around 50-100
            target_e = random.randint(40, 100)
            if target_e > len(all_possible):
                continue
            sample = random.sample(all_possible, target_e)
            sample = [list(e) for e in sample]
            
            is_non_2col = _is_non_2colorable(n, sample, deadline)
            if is_non_2col is True:
                if len(sample) < len(best_edges):
                    best_edges = sample
                    best_n = n
                    # Try to improve this new best by removing edges
                    current_edges = sample[:]
                    current_n = n
                    edge_indices = list(range(len(current_edges)))
                    random.shuffle(edge_indices)
                    for idx in edge_indices:
                        if time.time() > deadline:
                            break
                        edge_to_remove = current_edges[idx]
                        temp_edges = current_edges[:idx] + current_edges[idx+1:]
                        is_non_2col = _is_non_2colorable(current_n, temp_edges, deadline)
                        if is_non_2col is True:
                            current_edges = temp_edges
                            if len(current_edges) < len(best_edges):
                                best_edges = current_edges[:]
                                best_n = current_n

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
