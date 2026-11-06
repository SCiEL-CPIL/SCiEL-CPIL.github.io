import time
import random
from itertools import combinations
import sys

def is_2_colorable_fast(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colorable using DPLL with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Convert edges to bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # For very small n, brute force is often faster due to overhead
    if n <= 10:
        for color_mask in range(1 << n):
            valid = True
            for em in edge_masks:
                if (em & color_mask) == 0 or (em & color_mask) == em:
                    valid = False
                    break
            if valid:
                return True
        return False

    # DPLL for larger n
    # State: 
    # color[v] = -1 (unassigned), 0, 1
    # We maintain counts of 0s and 1s in each edge to detect contradictions and unit clauses.
    
    color = [-1] * n
    c0 = [0] * len(edges)
    c1 = [0] * len(edges)
    
    # Precompute vertex to edges incidence
    vertex_edges = [[] for _ in range(n)]
    for i, em in enumerate(edge_masks):
        # Iterate bits in mask
        m = em
        while m:
            v = (m & -m).bit_length() - 1
            vertex_edges[v].append(i)
            m &= m - 1

    def propagate():
        """
        Unit propagation:
        If an edge has 4 vertices of one color, the 5th must be the other.
        If an edge has 5 vertices of one color, contradiction.
        """
        changed = True
        while changed:
            changed = False
            for i in range(len(edges)):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                if c0[i] == 4 and c1[i] == 0:
                    # Must be 1
                    # Find the unassigned vertex in this edge
                    # We need to find v in edge i where color[v] == -1
                    # Since c0[i]==4, there is exactly one -1
                    m = edge_masks[i]
                    # Iterate vertices in edge i
                    # Optimization: we can just check all vertices in the edge
                    # But we need to be careful with performance.
                    # Since n is small (<=26), and edges are 5-uniform, this is fast.
                    found = False
                    mm = m
                    while mm:
                        v = (mm & -mm).bit_length() - 1
                        if color[v] == -1:
                            color[v] = 1
                            c1[i] += 1
                            # Update other edges containing v
                            for other_e in vertex_edges[v]:
                                c1[other_e] += 1
                                if c1[other_e] == 5:
                                    return False
                            changed = True
                            found = True
                            break
                        mm &= mm - 1
                    if not found:
                        # Should not happen if counts are correct
                        return False
                elif c1[i] == 4 and c0[i] == 0:
                    # Must be 0
                    m = edge_masks[i]
                    found = False
                    mm = m
                    while mm:
                        v = (mm & -mm).bit_length() - 1
                        if color[v] == -1:
                            color[v] = 0
                            c0[i] += 1
                            for other_e in vertex_edges[v]:
                                c0[other_e] += 1
                                if c0[other_e] == 5:
                                    return False
                            changed = True
                            found = True
                            break
                        mm &= mm - 1
                    if not found:
                        return False
        return True

    def backtrack():
        # Check for contradiction
        if not propagate():
            return False
            
        # Find unassigned vertex
        # Heuristic: Pick a vertex in the most constrained edge
        # Or simply pick the first unassigned vertex.
        # Better heuristic: Pick vertex that appears in most edges with high saturation.
        
        # For speed, let's just pick the first unassigned vertex for now, 
        # but maybe prioritize vertices in edges with c0+c1 == 4?
        # Actually, if propagate() returns True, there are no unit clauses left.
        # So all edges have c0+c1 <= 3 or c0=0,c1=0 etc.
        
        # Let's find an unassigned vertex.
        unassigned = [v for v in range(n) if color[v] == -1]
        if not unassigned:
            return True
            
        # Heuristic: Choose vertex v that maximizes the "impact"
        # Impact = sum of (5 - c0[e] - c1[e]) for e in vertex_edges[v]? No, that's unassigned count.
        # Let's choose v that is in the most edges.
        # Or just random? Random helps escape local minima in some cases but is slower.
        # Let's stick to a deterministic heuristic for DPLL: Max Degree.
        
        # Pre-computed degrees?
        # Let's just pick the first one for simplicity, or one with max degree.
        # Calculating degree every time is expensive.
        # Let's just pick unassigned[0].
        
        v = unassigned[0]
        
        # Try 0
        saved_c0 = c0[:]
        saved_c1 = c1[:]
        color[v] = 0
        ok = True
        for e_idx in vertex_edges[v]:
            c0[e_idx] += 1
            if c0[e_idx] == 5:
                ok = False
                break
        if ok and propagate():
            if backtrack():
                return True
        
        # Backtrack
        color[v] = -1
        c0[:] = saved_c0
        c1[:] = saved_c1
        
        # Try 1
        saved_c0 = c0[:]
        saved_c1 = c1[:]
        color[v] = 1
        ok = True
        for e_idx in vertex_edges[v]:
            c1[e_idx] += 1
            if c1[e_idx] == 5:
                ok = False
                break
        if ok and propagate():
            if backtrack():
                return True
        
        # Backtrack
        color[v] = -1
        c0[:] = saved_c0
        c1[:] = saved_c1
        
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    # Prioritize n values likely to yield minimal edges
    # 51 edges is the target. n=11 or 12 are common.
    for n in [11, 12, 10, 13, 9]:
        if time.time() > deadline - 10:
            break
        
        all_possible = list(combinations(range(n), 5))
        
        # Check if complete is non-2-colorable
        if is_2_colorable_fast(n, all_possible):
            continue
        
        # Greedy removal with multiple restarts
        # With faster solver, we can do more restarts
        max_attempts = 300 
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Restart from complete graph
            curr = [list(e) for e in all_possible]
            
            edge_indices = list(range(len(curr)))
            random.shuffle(edge_indices)
            
            removed_any = True
            while removed_any and time.time() < deadline - 5:
                removed_any = False
                # Shuffle indices again to explore different removal orders
                random.shuffle(edge_indices)
                for i in edge_indices:
                    if time.time() > deadline - 5:
                        break
                    if i >= len(curr):
                        continue
                    
                    # Try removing edge i
                    new_edges = curr[:i] + curr[i+1:]
                    if not is_2_colorable_fast(n, new_edges):
                        curr = new_edges
                        # Re-shuffle to break symmetry
                        edge_indices = list(range(len(curr)))
                        removed_any = True
                        break
            
            if len(curr) < best_m:
                best_m = len(curr)
                best_edges = curr[:]
                best_n = n
                if best_m <= 51:
                    break
        
        if best_m <= 51:
            break
    
    if best_edges is None:
        # Fallback to complete 9-vertex
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
    
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
