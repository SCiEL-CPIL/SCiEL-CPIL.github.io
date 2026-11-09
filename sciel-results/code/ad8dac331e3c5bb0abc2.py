import time
import random
from itertools import combinations

_WALL_LIMIT = 150.0
_MARGIN = 10.0

def _build_edge_data(n):
    """
    Returns:
    edges: list of tuples of 5 vertices
    killed_masks: list of bitmasks, where bit i is set if edge kills coloring i
    killed_lists: list of lists of coloring indices, for faster count updates
    """
    edges = [tuple(e) for e in combinations(range(n), 5)]
    num_colorings = 1 << n
    killed_masks = []
    killed_lists = []
    
    for e in edges:
        vs = list(e)
        vs_set = set(vs)
        rest = [v for v in range(n) if v not in vs_set]
        r = len(rest)
        mask = 0
        lst = []
        # Iterate over all assignments of 'rest'
        # For each assignment, check if edge is monochromatic
        # Edge is monochromatic if all vs are 0 OR all vs are 1.
        # This is independent of 'rest' assignments.
        # So for EVERY assignment of 'rest', the edge is monochromatic?
        # NO. The edge is monochromatic if the colors of v1..v5 are all same.
        # The colors of v1..v5 are FIXED in the mask construction logic of the previous code?
        # Let's re-read the previous code's _edge_masks.
        # It iterates p in range(1 << r).
        # It constructs idx0 and idx1.
        # idx0: vs are all 0, rest are p.
        # idx1: vs are all 1, rest are p.
        # So for each p, there are 2 colorings where the edge is monochromatic.
        # Total 2 * 2^r = 2^(r+1) colorings.
        
        for p in range(1 << r):
            # Construct coloring index for vs=0, rest=p
            idx0 = 0
            idx1 = 0
            for v in vs:
                idx1 |= (1 << v)
            for i, v in enumerate(rest):
                if (p >> i) & 1:
                    idx0 |= (1 << v)
                    idx1 |= (1 << v)
            mask |= (1 << idx0) | (1 << idx1)
            lst.append(idx0)
            lst.append(idx1)
            
        killed_masks.append(mask)
        killed_lists.append(lst)
        
    return edges, killed_masks, killed_lists

def _greedy_set_cover(n, edges, killed_masks, killed_lists, deadline, rng):
    num_colorings = 1 << n
    full_mask = (1 << num_colorings) - 1
    uncovered = full_mask
    chosen = []
    chosen_masks = []
    remaining = list(range(len(edges)))
    
    # Precompute bit_count for speed? Python int.bit_count is fast.
    
    while uncovered:
        best = -1
        best_c = 0
        for ei in remaining:
            c = (uncovered & killed_masks[ei]).bit_count()
            if c > best_c:
                best_c = c
                best = ei
                if best_c == uncovered.bit_count():
                    break
        if best < 0:
            break
        chosen.append(best)
        chosen_masks.append(killed_masks[best])
        uncovered &= ~killed_masks[best]
        remaining.remove(best)
        
        if time.time() > deadline:
            break
            
    if uncovered == 0:
        return chosen, chosen_masks
    return None, None

def _local_prune_with_counts(n, chosen, killed_lists, deadline, rng):
    """
    Prune the chosen set of edges.
    Uses coverage counts to check validity.
    """
    if not chosen:
        return chosen
    
    num_colorings = 1 << n
    cover_count = [0] * num_colorings
    
    # Initialize counts
    for ei in chosen:
        for c in killed_lists[ei]:
            cover_count[c] += 1
            
    chosen_set = set(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle to break symmetry
        order = list(chosen_set)
        rng.shuffle(order)
        for ei in order:
            if ei not in chosen_set:
                continue
            # Check if removing ei leaves any count at 0
            valid = True
            for c in killed_lists[ei]:
                if cover_count[c] == 1:
                    valid = False
                    break
            if valid:
                # Remove ei
                for c in killed_lists[ei]:
                    cover_count[c] -= 1
                chosen_set.remove(ei)
                improved = True
                break # Restart loop with new set
                
    return list(chosen_set)

def _try_n(n, deadline, rng):
    start = time.time()
    edges, killed_masks, killed_lists = _build_edge_data(n)
    
    best = None
    best_size = 1000
    
    # Multiple restarts with different random seeds or perturbations
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # 1. Greedy
        chosen, _ = _greedy_set_cover(n, edges, killed_masks, killed_lists, deadline, rng)
        if chosen is None:
            continue
            
        # 2. Local Prune
        pruned = _local_prune_with_counts(n, chosen, killed_lists, deadline, rng)
        
        if len(pruned) < best_size:
            best_size = len(pruned)
            best = pruned
            
        # 3. Iterated Local Search (optional if time permits)
        # Try to improve further by swapping
        if time.time() < deadline - 5:
            current = set(pruned)
            current_size = len(current)
            no_improve = 0
            while no_improve < 10 and time.time() < deadline - 2:
                # Try to remove one more
                improved = False
                order = list(current)
                rng.shuffle(order)
                for ei in order:
                    if ei not in current: continue
                    # Check validity
                    # We don't have global cover_count here, so we must recompute or use masks
                    # Using masks is faster for small sets?
                    # Let's use masks for the swap step to be safe, or just re-run prune?
                    # Re-running prune is O(|S|^2 * 2^(n-5)).
                    # For |S|=50, n=12, 2500 * 128 = 320k ops. Fast.
                    
                    # Actually, let's just do a "swap" heuristic:
                    # Pick e in current, e' not in current.
                    # If (current - {e} + {e'}) covers all, and allows removal?
                    # This is complex.
                    
                    # Simpler: Randomly remove 2 edges, add 2 random edges, then prune.
                    if len(current) > 2:
                        new_current = set(current)
                        to_remove = rng.sample(list(new_current), 2)
                        for e in to_remove:
                            new_current.remove(e)
                        
                        # Add random edges
                        candidates = [e for e in range(len(edges)) if e not in new_current]
                        if candidates:
                            to_add = rng.sample(candidates, 2)
                            for e in to_add:
                                new_current.add(e)
                        
                        new_pruned = _local_prune_with_counts(n, list(new_current), killed_lists, deadline, rng)
                        if len(new_pruned) < current_size:
                            current = set(new_pruned)
                            current_size = len(current)
                            no_improve = 0
                            improved = True
                        else:
                            no_improve += 1
                            
                if not improved:
                    break
                    
            if current_size < best_size:
                best_size = current_size
                best = list(current)
                
    if best is None:
        return None
        
    # Final verification
    u = 0
    for ei in best:
        u |= killed_masks[ei]
    if u != (1 << (1 << n)) - 1:
        return None
        
    return {'n': n, 'edges': [list(edges[i]) for i in best]}

def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT - _MARGIN
    rng = random.Random(12345)
    
    # Fallback: Complete 5-graph on 9 vertices (126 edges)
    fallback = {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    best = fallback
    
    # Try n=12 first, then 13
    for n in (12, 13):
        if time.time() > deadline:
            break
        try:
            res = _try_n(n, deadline, rng)
        except Exception:
            res = None
        if res is not None and len(res['edges']) < len(best['edges']):
            best = res
            
    return best

def _is_2_colorable(n, edges, deadline):
    """
    Exact backtracking checker.
    """
    edges = [tuple(e) for e in edges]
    color = [-1] * n
    e0 = [0] * len(edges)
    e1 = [0] * len(edges)
    uncol = [list(e) for e in edges]
    
    def propagate():
        queue = list(range(len(edges)))
        while queue:
            changed = False
            q2 = []
            for ei in queue:
                u = uncol[ei]
                if e0[ei] == 5 or e1[ei] == 5:
                    return False
                if len(u) == 1:
                    v = u[0]
                    if e0[ei] == 4:
                        forced = 1
                    elif e1[ei] == 4:
                        forced = 0
                    else:
                        continue
                    if color[v] == -1:
                        color[v] = forced
                        changed = True
                        u.remove(v)
                        for ej in range(len(edges)):
                            if v in uncol[ej]:
                                uncol[ej].remove(v)
                                if forced == 0:
                                    e0[ej] += 1
                                else:
                                    e1[ej] += 1
                                if e0[ej] == 5 or e1[ej] == 5:
                                    return False
                        if e0[ei] == 5 or e1[ei] == 5:
                            return False
                    elif color[v] != forced:
                        return False
            queue = q2
            if not changed:
                break
        return True

    import sys
    sys.setrecursionlimit(100000)

    def rec(k):
        if k >= n:
            return True
        for c in (0, 1):
            color[k] = c
            touched = []
            for ei, e in enumerate(edges):
                if k in e:
                    uncol[ei].remove(k)
                    if c == 0:
                        e0[ei] += 1
                    else:
                        e1[ei] += 1
                    touched.append(ei)
            if propagate():
                if rec(k + 1):
                    return True
            color[k] = -1
            for ei in touched:
                uncol[ei].append(k)
                if c == 0:
                    e0[ei] -= 1
                else:
                    e1[ei] -= 1
        return False

    return rec(0)

if __name__ == '__main__':
    _t0 = time.time()
    _res = construct()
    _ok = _is_2_colorable(_res['n'], _res['edges'], _t0 + 12.0)
    assert not _ok, "constructed hypergraph must NOT be 2-colourable"
    print("edges:", len(_res['edges']), "n:", _res['n'], "2-colourable:", _ok)
