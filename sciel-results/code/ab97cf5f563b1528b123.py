import itertools
import time
import random

def _build_masks_fast(n):
    """Build masks for all 5-edges on n vertices.
    masks[i] is a Python int where bit c is 1 iff coloring c (c in 0..2^n-1)
    makes edge i monochromatic.
    """
    all_edges = list(itertools.combinations(range(n), 5))
    num_edges = len(all_edges)
    U_size = 1 << n
    
    # Precompute for each vertex v: for each coloring c, the bit value
    # Instead, for each edge, compute the mask directly using the fact that
    # a coloring c makes edge {v1,v2,v3,v4,v5} monochromatic iff
    # (c & emask) == 0 or (c & emask) == emask where emask has bits at v1..v5
    
    # We can compute this more efficiently by iterating over colorings once
    # and for each coloring, determine which edges it makes monochromatic.
    # But that's O(2^n * num_edges) which is the same.
    
    # Alternative: for each edge, iterate over colorings. With n=11, U_size=2048,
    # num_edges=462. Total ops = 462*2048 = 946176, which is fine.
    
    edge_vmasks = []
    for edge in all_edges:
        em = 0
        for v in edge:
            em |= (1 << v)
        edge_vmasks.append(em)
    
    masks = [0] * num_edges
    for idx in range(num_edges):
        em = edge_vmasks[idx]
        mask = 0
        for c in range(U_size):
            bits = c & em
            if bits == 0 or bits == em:
                mask |= (1 << c)
        masks[idx] = mask
    
    return all_edges, edge_vmasks, masks

def _greedy_cover(masks, U_size, deadline=None):
    """Greedy set cover to find a small set of edges whose masks cover all colorings."""
    if deadline is not None and time.time() > deadline:
        return None
    
    uncovered = (1 << U_size) - 1
    selected = []
    remaining = set(range(len(masks)))
    
    while uncovered:
        best_idx = -1
        best_gain = -1
        
        for idx in remaining:
            gain = (uncovered & masks[idx]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_idx == -1 or best_gain == 0:
            break
            
        selected.append(best_idx)
        remaining.remove(best_idx)
        uncovered &= ~masks[best_idx]
        
        if deadline is not None and time.time() > deadline:
            return None
    
    return selected

def _local_prune(selected, masks, U_size, deadline=None):
    """Try to remove edges from the selected set if coverage is maintained."""
    if deadline is not None and time.time() > deadline:
        return selected
    
    selected = list(selected)
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            if deadline is not None and time.time() > deadline:
                return selected
            # Try removing selected[i]
            candidate = selected[:i] + selected[i+1:]
            # Check if candidate still covers all
            cover = 0
            for idx in candidate:
                cover |= masks[idx]
            if cover == (1 << U_size) - 1:
                selected = candidate
                changed = True
                break  # restart scan
    return selected

def _swap_improve(selected, masks, U_size, all_edges, deadline=None):
    """Try to swap one edge for another to potentially enable further pruning."""
    if deadline is not None and time.time() > deadline:
        return selected
    
    selected = list(selected)
    improved = True
    while improved:
        improved = False
        for i in range(len(selected)):
            if deadline is not None and time.time() > deadline:
                return selected
            for j in range(len(all_edges)):
                if j in selected:
                    continue
                # Try swapping selected[i] with j
                candidate = list(selected)
                candidate[i] = j
                # Check coverage
                cover = 0
                for idx in candidate:
                    cover |= masks[idx]
                if cover == (1 << U_size) - 1:
                    # Now try to prune
                    pruned = _local_prune(candidate, masks, U_size, deadline)
                    if len(pruned) < len(selected):
                        selected = pruned
                        improved = True
                        break
            if improved:
                break
    return selected

def _try_n(n, deadline):
    """Try to find a minimal 2-coloring-failing hypergraph on n vertices."""
    if time.time() > deadline:
        return None
    
    all_edges, edge_vmasks, masks = _build_masks_fast(n)
    U_size = 1 << n
    
    # Greedy set cover
    selected = _greedy_cover(masks, U_size, deadline)
    if selected is None:
        return None
    
    # Local prune
    selected = _local_prune(selected, masks, U_size, deadline)
    
    # Swap improve
    selected = _swap_improve(selected, masks, U_size, all_edges, deadline)
    
    if time.time() > deadline:
        return None
    
    edges = [all_edges[i] for i in selected]
    return {'n': n, 'edges': [list(e) for e in edges]}

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    best_result = None
    best_edge_count = float('inf')
    
    # Try n values in order of likelihood to find good solutions
    for n in [11, 10, 12]:
        if time.time() > deadline:
            break
        
        result = _try_n(n, deadline)
        if result is not None:
            edge_count = len(result['edges'])
            if edge_count < best_edge_count:
                best_edge_count = edge_count
                best_result = result
    
    # Fallback to seed if something went wrong
    if best_result is None:
        return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
    
    return best_result
