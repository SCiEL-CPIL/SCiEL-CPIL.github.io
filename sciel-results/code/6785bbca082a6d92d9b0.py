import time
import sys
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _is_2_colorable(n, edges):
    """Exact check (backtracking + unit propagation) for a proper 2-colouring.
    Returns True iff the hypergraph IS 2-colorable."""
    if not edges:
        return True
    
    m = len(edges)
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    all_vars = (1 << n) - 1
    
    def backtrack(assign, unassigned):
        # Unit propagation loop
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                assigned_in_edge = mask & ~unassigned
                unassigned_in_edge = mask & unassigned
                
                if unassigned_in_edge == 0:
                    if (assign & mask) == mask or (assign & mask) == 0:
                        return None
                else:
                    assigned_zero = assigned_in_edge & ~assign
                    assigned_one = assigned_in_edge & assign
                    unassigned_count = bin(unassigned_in_edge).count('1')
                    
                    if assigned_one == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign |= (1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
                    
                    if assigned_zero == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign &= ~(1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
        
        if unassigned == 0:
            for mask in edge_masks:
                if (assign & mask) == mask or (assign & mask) == 0:
                    return None
            return assign
        
        # Branch: pick unassigned variable with most constraints (heuristic)
        # Simple: pick highest-numbered unassigned
        v = unassigned.bit_length() - 1
        bit = 1 << v
        
        res = backtrack(assign | bit, unassigned & ~bit)
        if res is not None:
            return res
        return backtrack(assign & ~bit, unassigned & ~bit)
    
    return backtrack(0, all_vars) is not None

def _compute_edge_masks(n, all_edges):
    """Compute for each edge the bitmask of colorings where it is monochromatic."""
    num_colorings = 1 << n
    masks = []
    for edge in all_edges:
        mask = 0
        edge_set = set(edge)
        
        # Generate all colorings where edge is monochromatic
        # This is equivalent to:
        # 1. All vertices in edge are 0
        # 2. All vertices in edge are 1
        
        # We can iterate through all colorings and check, but that's 2^n per edge.
        # Total work: 2^n * C(n,5). For n=11, 2048 * 462 ~ 1M ops. Very fast.
        # But we can do better.
        
        # Let's just use the direct method for simplicity and speed in Python/Numpy.
        # Actually, constructing the bitset directly is faster.
        
        # Mask for "all 0s on edge":
        # The colorings are determined by the bits outside the edge.
        # There are n-5 free bits.
        # We can enumerate the free bits.
        
        free_verts = [v for v in range(n) if v not in edge_set]
        
        # All 0s
        for free_val in range(1 << len(free_verts)):
            coloring = 0
            for i, v in enumerate(free_verts):
                if free_val & (1 << i):
                    coloring |= (1 << v)
            mask |= (1 << coloring)
            
        # All 1s
        edge_val = 0
        for v in edge:
            edge_val |= (1 << v)
        for free_val in range(1 << len(free_verts)):
            coloring = edge_val
            for i, v in enumerate(free_verts):
                if free_val & (1 << i):
                    coloring |= (1 << v)
            mask |= (1 << coloring)
            
        masks.append(mask)
        
    return masks

def _solve_set_cover(n, all_edges, edge_masks, deadline):
    """Solve the set cover problem using MILP."""
    num_edges = len(all_edges)
    num_colorings = 1 << n
    
    # Objective: minimize sum x_i
    c = np.ones(num_edges)
    
    # Constraints: for each coloring, sum of x_i for edges covering it >= 1
    # A is num_colorings x num_edges
    # A[c, e] = 1 if edge e covers coloring c
    
    # We have edge_masks as list of ints (bitsets).
    # We need to convert this to a dense matrix or sparse.
    # scipy.optimize.milp accepts sparse or dense.
    # Given the size (2048x462), dense is fine.
    
    A = np.zeros((num_colorings, num_edges), dtype=np.int8)
    
    for e_idx, mask in enumerate(edge_masks):
        # Set A[c, e_idx] = 1 for all c where bit c is set in mask
        # Efficient way: iterate through set bits? Or just check?
        # 2048 * 462 checks is ~1M.
        for c_idx in range(num_colorings):
            if (mask >> c_idx) & 1:
                A[c_idx, e_idx] = 1
                
    constraints = [LinearConstraint(A, 1, np.inf)]
    bounds = Bounds(0, 1)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=np.ones(num_edges), bounds=bounds)
        if res.success:
            selected_edges = [i for i in range(num_edges) if res.x[i] > 0.5]
            return selected_edges
    except Exception as e:
        print(f"MILP failed: {e}", file=sys.stderr)
        
    return None

def _greedy_set_cover(n, all_edges, edge_masks, deadline):
    """Greedy set cover as fallback."""
    universe = (1 << (1 << n)) - 1
    covered = 0
    uncovered = universe
    selected = []
    used = [False] * len(all_edges)
    
    indices = list(range(len(all_edges)))
    
    while uncovered != 0:
        if time.time() > deadline:
            return None
        
        best_idx = -1
        best_count = 0
        for idx in indices:
            if used[idx]:
                continue
            coverage = edge_masks[idx] & uncovered
            count = coverage.bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            break
        
        used[best_idx] = True
        selected.append(best_idx)
        covered |= edge_masks[best_idx]
        uncovered = universe & ~covered
    
    if uncovered == 0:
        return selected
    return None

def _local_search_reduce(n, selected_indices, all_edges, edge_masks, deadline):
    """Try to reduce the number of edges."""
    current_selected = list(selected_indices)
    current_set = set(current_selected)
    
    # Precompute coverage
    covered = 0
    for idx in current_selected:
        covered |= edge_masks[idx]
        
    universe = (1 << (1 << n)) - 1
    
    if covered != universe:
        return current_selected
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_selected)):
            if time.time() > deadline:
                break
            
            # Try removing edge i
            test_covered = covered & ~edge_masks[current_selected[i]]
            if test_covered == universe:
                # Can remove
                covered = test_covered
                current_selected.pop(i)
                improved = True
                break
                
    return current_selected

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Fallback: complete 5-graph on 9 vertices
    fallback_n = 9
    fallback_edges = [list(e) for e in itertools.combinations(range(fallback_n), 5)]
    
    best_result = None
    best_count = float('inf')
    
    # Try n=11
    n = 11
    all_edges = list(itertools.combinations(range(n), 5))
    
    print(f"n={n}, edges={len(all_edges)}", file=sys.stderr)
    t0 = time.time()
    edge_masks = _compute_edge_masks(n, all_edges)
    print(f"Computed masks in {time.time()-t0:.2f}s", file=sys.stderr)
    
    # Try MILP
    if time.time() < deadline - 10:
        selected = _solve_set_cover(n, all_edges, edge_masks, deadline - 5)
        if selected is not None:
            edges = [list(all_edges[i]) for i in selected]
            # Verify
            if not _is_2_colorable(n, edges):
                # Local search to ensure minimal
                reduced_indices = _local_search_reduce(n, selected, all_edges, edge_masks, deadline - 5)
                edges = [list(all_edges[i]) for i in reduced_indices]
                if not _is_2_colorable(n, edges):
                    best_result = edges
                    best_count = len(edges)
                    print(f"MILP found {best_count} edges", file=sys.stderr)
    
    # If MILP didn't find optimal or failed, try greedy + local search
    if best_count > 51: # We know 51 is possible
        if time.time() < deadline - 10:
            selected = _greedy_set_cover(n, all_edges, edge_masks, deadline - 5)
            if selected is not None:
                reduced_indices = _local_search_reduce(n, selected, all_edges, edge_masks, deadline - 5)
                edges = [list(all_edges[i]) for i in reduced_indices]
                if not _is_2_colorable(n, edges):
                    if len(edges) < best_count:
                        best_result = edges
                        best_count = len(edges)
                        print(f"Greedy found {best_count} edges", file=sys.stderr)
    
    if best_result is not None:
        return {'n': n, 'edges': best_result}
        
    # Fallback to n=10? Or just return fallback.
    # Actually, if n=11 failed, try n=10.
    n = 10
    all_edges = list(itertools.combinations(range(n), 5))
    edge_masks = _compute_edge_masks(n, all_edges)
    
    if time.time() < deadline - 10:
        selected = _solve_set_cover(n, all_edges, edge_masks, deadline - 5)
        if selected is not None:
            edges = [list(all_edges[i]) for i in selected]
            if not _is_2_colorable(n, edges):
                reduced_indices = _local_search_reduce(n, selected, all_edges, edge_masks, deadline - 5)
                edges = [list(all_edges[i]) for i in reduced_indices]
                if not _is_2_colorable(n, edges):
                    best_result = edges
                    best_count = len(edges)
    
    if best_result is not None:
        return {'n': n, 'edges': best_result}
        
    return {'n': fallback_n, 'edges': fallback_edges}
