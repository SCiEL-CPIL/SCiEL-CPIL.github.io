import numpy as np
import itertools
import time
import random

D = 11

def _all_candidates(max_norm=3):
    """Generate all nonzero integer vectors in R^D with norm^2 <= max_norm^2.
    We include norms 1, 2, 3 (and optionally 4) to have a rich candidate pool.
    """
    cands = []
    # Norm 1: single coordinate ±1
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            cands.append(tuple(v))
    # Norm 2: two coordinates ±1 (each coordinate can be ±1, so 2*2=4 sign combos per pair)
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    cands.append(tuple(v))
    # Norm 3: three coordinates each ±1 (8 sign combos per triple)
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            cands.append(tuple(v))
    # Norm 4: four coordinates each ±1 (16 sign combos per 4-subset)
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for l in range(k + 1, D):
                    for si in (1, -1):
                        for sj in (1, -1):
                            for sk in (1, -1):
                                for sl in (1, -1):
                                    v = [0] * D
                                    v[i] = si
                                    v[j] = sj
                                    v[k] = sk
                                    v[l] = sl
                                    cands.append(tuple(v))
    return cands


def _norm_sq(v):
    return sum(x * x for x in v)


def _dot(a, b):
    s = 0
    for x, y in zip(a, b):
        s += x * y
    return s


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Build candidate pool with norms 1-4
    cands = _all_candidates(max_norm=4)
    n = len(cands)
    norms = [_norm_sq(v) for v in cands]

    # Precompute dot products? Too large (n ~ 110 + 220 + 330 + 440 = ~1100, n^2 ~ 1.2M, feasible)
    # Actually let's compute on the fly for memory, but precompute is faster.
    # n is around 1100, n^2 = 1.2M, each int — fine.
    # But we only need dot products for conflict checking during search.
    # Let's precompute a conflict matrix: conflict[i][j] = True if vectors i and j
    # cannot both be selected.
    # Two vectors v, w with norms nv, nw: distance^2 = nv + nw - 2*v.w >= max(nv, nw)
    # i.e., nv + nw - 2*v.w >= max(nv, nw)
    # If nv == nw == r: 2r - 2*v.w >= r => v.w <= r/2
    # General: nv + nw - 2*d >= max(nv, nw) => 2*d <= nv + nw - max(nv,nw) = min(nv,nw)
    # So conflict if 2*d > min(nv, nw), i.e., d > min(nv,nw)/2
    # Since d is integer, conflict if d >= floor(min(nv,nw)/2) + 1
    # For norm 2 (ns=2): conflict if d >= 2 (since min=2, d > 1 means d>=2)
    # For norm 4 (ns=4): conflict if d >= 3 (d > 2)
    # For norm 1 (ns=1): conflict if d >= 1 (d > 0.5, so d>=1)
    # For norm 3 (ns=3): conflict if d >= 2 (d > 1.5, so d>=2)

    # Precompute conflict as a list of bitsets or just a list of lists
    # For n ~ 1100, a list of sets is fine.
    # Actually, let's use a list of lists of indices for speed in local search.
    
    # Let's precompute conflicts efficiently
    # Group by norm for faster dot product computation
    # Actually, just compute all pairwise dots for n~1100: 1100*1100/2 = 605K dot products of 11-dim vectors
    # That's ~6.6M multiply-adds, very fast.
    
    conflict = [set() for _ in range(n)]
    for i in range(n):
        vi = cands[i]
        ni = norms[i]
        for j in range(i + 1, n):
            vj = cands[j]
            nj = norms[j]
            d = _dot(vi, vj)
            min_n = ni if ni < nj else nj
            # Conflict if 2*d > min_n
            if 2 * d > min_n:
                conflict[i].add(j)
                conflict[j].add(i)
    
    if elapsed() > deadline:
        # Fallback: return seed (norm-2 vectors)
        seed_vecs = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        seed_vecs.append(v)
        return np.array(seed_vecs, dtype=np.int64)

    # Greedy construction with scoring
    # At each step, among unselected candidates, pick the one with the best score.
    # Score = (number of unselected non-conflicting candidates) - penalty for high norm
    # Actually, to maximize count, we want to pick vectors that conflict with few others.
    # Greedy: repeatedly pick the candidate with the minimum number of conflicts among remaining.
    
    # But we also want to consider that picking a low-norm vector allows more others.
    # Let's use a priority: prefer candidates that are "independent" of many.
    
    # Strategy: 
    # 1. Start with an empty set.
    # 2. Maintain a set of "available" candidates (not yet selected, not conflicting with selected).
    # 3. At each step, pick the available candidate with the fewest conflicts among available.
    #    Tie-break: prefer lower norm (so we can pack more).
    # 4. Repeat until no available candidates.
    
    # This is a greedy maximal independent set on the conflict graph.
    
    # For better results, we can do multiple random restarts with different orderings.
    
    def greedy_construct(order=None):
        """Greedy MIS with a given candidate ordering (or random if None)."""
        selected = set()
        available = set(range(n))
        
        if order is None:
            # Default: sort by (norm, index) to prefer low-norm vectors
            order = sorted(range(n), key=lambda i: (norms[i], i))
        
        # We'll process candidates in the given order, but only select if no conflict with selected.
        # Actually, a better greedy: at each step, among all available, pick the one with min conflicts.
        # But that's O(n^2) per step. For n=1100, that's ~1.2M per step, too slow for many steps.
        
        # Simpler: iterate through order, select if no conflict with already selected.
        # This is a standard greedy MIS.
        for i in order:
            if i not in available:
                continue
            # Check if i conflicts with any selected
            can_add = True
            for j in selected:
                if j in conflict[i]:
                    can_add = False
                    break
            if can_add:
                selected.add(i)
                # Remove all conflicts of i from available
                for j in conflict[i]:
                    if j in available:
                        available.discard(j)
        
        return selected
    
    # Try multiple orderings
    best_selected = set()
    best_size = 0
    
    # Ordering 1: by norm (prefer low norm)
    order1 = sorted(range(n), key=lambda i: (norms[i], i))
    sel1 = greedy_construct(order1)
    if len(sel1) > best_size:
        best_size = len(sel1)
        best_selected = set(sel1)
    
    if elapsed() > deadline:
        return np.array([cands[i] for i in best_selected], dtype=np.int64)
    
    # Ordering 2: random shuffles
    rng = random.Random(42)
    for restart in range(20):
        if elapsed() > deadline - 10:
            break
        order2 = list(range(n))
        rng.shuffle(order2)
        # Bias towards low norm: sort by (norm with random tie-break)
        order2 = sorted(order2, key=lambda i: (norms[i], rng.random()))
        sel2 = greedy_construct(order2)
        if len(sel2) > best_size:
            best_size = len(sel2)
            best_selected = set(sel2)
    
    if elapsed() > deadline:
        return np.array([cands[i] for i in best_selected], dtype=np.int64)
    
    # Local search: try to improve by swaps
    # Move: remove a selected vector and add one or more unselected vectors.
    # Or: swap one selected for one unselected (if it increases count, it doesn't directly help
    # since count stays same). We need to find a way to add vectors.
    
    # Better local search: 
    # 1. Try to add any unselected vector that doesn't conflict with selected. (Shouldn't exist if greedy is maximal)
    # 2. Try to remove one selected vector and add two or more unselected vectors that only conflict with the removed one.
    
    # Let's do a simple local search: 
    # For each selected vector s, consider removing it. Then check if we can add any unselected
    # vectors that only conflict with s (and not with other selected).
    
    improved = True
    max_local_iters = 50
    local_iter = 0
    while improved and local_iter < max_local_iters:
        improved = False
        local_iter += 1
        if elapsed() > deadline - 5:
            break
        
        selected = set(best_selected)
        best_improvement = 0
        best_move = None
        
        # For each selected vector, try removing it
        for s in list(selected):
            if elapsed() > deadline - 5:
                break
            # Remove s, see what becomes available
            new_selected = selected - {s}
            # Find unselected vectors that don't conflict with new_selected
            # and only conflicted with s (among selected)
            candidates_to_add = []
            for u in range(n):
                if u in new_selected:
                    continue
                # Check if u conflicts with any in new_selected
                conflicts_with_new = False
                for j in new_selected:
                    if j in conflict[u]:
                        conflicts_with_new = True
                        break
                if not conflicts_with_new:
                    # u can be added
                    candidates_to_add.append(u)
            
            # Now, from candidates_to_add, we want to add as many as possible
            # such that they don't conflict with each other.
            # Greedy: sort by norm, add if no conflict with already-added.
            added = []
            for u in sorted(candidates_to_add, key=lambda x: (norms[x], x)):
                can_add = True
                for a in added:
                    if a in conflict[u]:
                        can_add = False
                        break
                if can_add:
                    added.append(u)
            
            net_gain = len(added) - 1
            if net_gain > best_improvement:
                best_improvement = net_gain
                best_move = (s, added)
        
        if best_move is not None and best_improvement > 0:
            s, added = best_move
            selected.discard(s)
            selected.update(added)
            best_selected = selected
            best_size = len(selected)
            improved = True
    
    # Final result
    result = [cands[i] for i in best_selected]
    if not result:
        # Fallback to seed
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        result.append(v)
    
    return np.array(result, dtype=np.int64)
