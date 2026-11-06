"""Minimum maximal no-isosceles set: structural redesign via vectorized
conflict-graph construction + simulated-annealing local search."""
import time
import random
import numpy as np

N = 16
DEADLINE = 325.0  # wall-clock safety margin (evaluator kills at 330s)


def _build_conflict_graph():
    """Return a boolean matrix C where C[i,j] is True iff points i and j
    are 'conflicting' in the sense that there exists a third grid point k
    such that {i,j,k} forms an isosceles triangle (with i,j,k distinct).
    
    We also need: for a set S, adding point p is valid iff for all a,b in S,
    {p,a,b} is NOT isosceles. Equivalently, p must not be in the 'forbidden'
    set for any pair (a,b) in S.
    
    For efficiency, we precompute for each pair (a,b) the set of points c
    such that {a,b,c} is isosceles. Then p is forbidden by (a,b) iff p is in
    that set.
    
    We'll store: for each point index, a bitmask of points it conflicts with
    in the sense: point i and point j are 'incompatible as a pair' if there
    exists k such that {i,j,k} is isosceles. But actually the constraint is:
    for the set S to be no-isosceles, for every triple (a,b,c) in S, they
    must not form an isosceles triangle.
    
    So the conflict is on TRIPLES, not pairs. However, we can reformulate:
    when building S incrementally, to add p, we need that for every pair
    (a,b) already in S, the triple (p,a,b) is not isosceles.
    
    Let's precompute: for each pair (a,b), the set of c such that (a,b,c)
    is isosceles. Store as a bitset or boolean array.
    """
    pts = np.array([(x, y) for x in range(N) for y in range(N)])
    npts = N * N
    
    # Precompute all squared distances between pairs
    # dist_sq[i, j] = squared distance between point i and point j
    dx = pts[:, 0:1] - pts[1:0, 0]  # wrong shape, let's do it properly
    # Use broadcasting
    px = pts[:, 0].astype(np.float64)
    py = pts[:, 1].astype(np.float64)
    dx = px[:, None] - px[None, :]
    dy = py[:, None] - py[None, :]
    dist_sq = dx * dx + dy * dy
    
    # For each pair (a, b), find all c such that triangle (a,b,c) is isosceles.
    # Isosceles means: d(a,b)==d(a,c) or d(a,b)==d(b,c) or d(a,c)==d(b,c)
    # i.e., d_ab == d_ac OR d_ab == d_bc OR d_ac == d_bc
    # Note: c must be distinct from a and b.
    
    # conflict[a, b] = boolean array of length npts, True if c makes (a,b,c) isosceles
    # This is npts^2 * npts which is 256^2 * 256 = ~16.7M booleans = ~16.7MB. OK.
    
    # But we can be smarter. For a fixed pair (a,b), the condition is:
    # d_ac == d_ab (c on circle centered at a with radius d_ab)
    # OR d_bc == d_ab (c on circle centered at b with radius d_ab)  
    # OR d_ac == d_bc (c on perpendicular bisector of ab)
    
    # Let's just compute it directly using the distance matrix.
    # For pair (a,b): 
    #   cond1: dist_sq[a, c] == dist_sq[a, b]
    #   cond2: dist_sq[b, c] == dist_sq[a, b]
    #   cond3: dist_sq[a, c] == dist_sq[b, c]
    # And c != a, c != b.
    
    # Precompute for all pairs at once is expensive. Let's do it per-pair but vectorized.
    
    # Actually, let's store for each point p, and for each pair (a,b), whether p is forbidden.
    # forbidden[p, a, b] = True if (p, a, b) is isosceles.
    # That's npts^3 = 256^3 = 16.7M booleans. ~16.7 MB. Fine.
    
    # But we can avoid storing the full 3D array. Instead, for the search, we need:
    # Given current set S, can we add p? Check for all (a,b) in S: not isosceles(p,a,b).
    # And is S maximal? For every p not in S, there exists (a,b) in S such that isosceles(p,a,b).
    
    # Let's precompute a 3D boolean array: iso[p, a, b] = True if p,a,b form isosceles.
    # Symmetries: iso is symmetric in all three arguments (isosceles is symmetric).
    
    # Compute efficiently:
    # For each pair (a, b) with a < b:
    #   Find all c such that (a,b,c) is isosceles.
    #   Set iso[a,b,c] = iso[a,c,b] = iso[b,a,c] = iso[b,c,a] = iso[c,a,b] = iso[c,b,a] = True
    
    # This is O(npts^2 * npts) = O(npts^3) which is 16.7M operations. Fine.
    
    # Let's build it.
    iso = np.zeros((npts, npts, npts), dtype=bool)
    
    for a in range(npts):
        for b in range(a + 1, npts):
            dab = dist_sq[a, b]
            # c such that d_ac == dab
            c1 = np.where(dist_sq[a, :] == dab)[0]
            # c such that d_bc == dab
            c2 = np.where(dist_sq[b, :] == dab)[0]
            # c such that d_ac == d_bc
            c3 = np.where(dist_sq[a, :] == dist_sq[b, :])[0]
            
            c_set = np.union1d(np.union1d(c1, c2), c3)
            # Exclude a and b themselves
            mask = np.ones(npts, dtype=bool)
            mask[a] = False
            mask[b] = False
            c_set = c_set[mask]
            
            for c in c_set:
                # Set all permutations
                iso[a, b, c] = True
                iso[a, c, b] = True
                iso[b, a, c] = True
                iso[b, c, a] = True
                iso[c, a, b] = True
                iso[c, b, a] = True
    
    return pts, dist_sq, iso


def _is_valid_set(S, iso):
    """Check if S is a no-isosceles set."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if iso[S_list[i], S_list[j], S_list[k]]:
                    return False
    return True


def _is_maximal(S, iso, all_points):
    """Check if S is maximal: no point outside S can be added."""
    S_set = set(S)
    for p in all_points:
        if p in S_set:
            continue
        # Can p be added? Check all pairs in S
        can_add = True
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if iso[p, S_list[i], S_list[j]]:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True


def _greedy_construction(order, iso):
    """Greedy construction: process points in given order, add if valid."""
    chosen = []
    chosen_set = set()
    for p in order:
        # Check if p can be added
        ok = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if iso[p, chosen[i], chosen[j]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(p)
            chosen_set.add(p)
    return chosen


def construct():
    start_time = time.time()
    
    # Build the conflict structure
    pts, dist_sq, iso = _build_conflict_graph()
    npts = N * N
    all_points = list(range(npts))
    
    # Phase 1: Multiple greedy restarts with random orders
    best_set = None
    best_size = npts + 1
    
    # Deterministic greedy first (row-major)
    order = list(range(npts))
    S = _greedy_construction(order, iso)
    if _is_maximal(S, iso, all_points):
        best_set = S
        best_size = len(S)
    
    # Random restarts
    rng = np.random.default_rng(42)
    max_restarts = 5000
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 60:
            break
        
        # Random permutation
        order = rng.permutation(npts).tolist()
        S = _greedy_construction(order, iso)
        
        if len(S) >= best_size:
            continue
        
        if _is_maximal(S, iso, all_points):
            best_set = S
            best_size = len(S)
    
    # Phase 2: Simulated annealing / local search to reduce size
    if best_set is not None:
        current_set = list(best_set)
        current_size = len(current_set)
        
        # Try to remove points and repair
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        max_iters = 50000
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            if T < T_min:
                break
            
            # Try removing a random point
            idx = rng.integers(0, len(current_set))
            removed = current_set[idx]
            new_set = current_set[:idx] + current_set[idx+1:]
            
            # Check if new_set is still no-isosceles (it is, since it's a subset)
            # Check if new_set is maximal
            if _is_maximal(new_set, iso, all_points):
                # Accept: we reduced size
                current_set = new_set
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_set = list(current_set)
            else:
                # Try to repair: add back some points to restore maximality
                # Find which points can now be added
                current_set_set = set(current_set)
                addable = []
                for p in all_points:
                    if p in current_set_set:
                        continue
                    ok = True
                    for i in range(len(current_set)):
                        for j in range(i + 1, len(current_set)):
                            if iso[p, current_set[i], current_set[j]]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        addable.append(p)
                
                if not addable:
                    # Can't repair, reject
                    pass
                else:
                    # Add one random addable point
                    add_p = addable[rng.integers(0, len(addable))]
                    new_set = current_set + [add_p]
                    new_size = len(new_set)
                    
                    # Check if new_set is no-isosceles
                    valid = True
                    for i in range(len(new_set)):
                        for j in range(i + 1, len(new_set)):
                            for k in range(j + 1, len(new_set)):
                                if iso[new_set[i], new_set[j], new_set[k]]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        delta_E = new_size - current_size
                        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                            current_set = new_set
                            current_size = new_size
                            if current_size < best_size:
                                best_size = current_size
                                best_set = list(current_set)
            
            T *= cooling
        
        # Further local search: try swapping points
        T = 5.0
        for it in range(20000):
            if time.time() - start_time > DEADLINE - 10:
                break
            if T < T_min:
                break
            
            # Try swapping a point in the set with a point outside
            if len(current_set) == 0:
                break
            
            idx = rng.integers(0, len(current_set))
            old_p = current_set[idx]
            
            # Find a candidate to swap in
            current_set_set = set(current_set)
            candidates = [p for p in all_points if p not in current_set_set]
            if not candidates:
                break
            
            new_p = candidates[rng.integers(0, len(candidates))]
            
            new_set = current_set[:idx] + [new_p] + current_set[idx+1:]
            
            # Check validity
            valid = True
            for i in range(len(new_set)):
                for j in range(i + 1, len(new_set)):
                    for k in range(j + 1, len(new_set)):
                        if iso[new_set[i], new_set[j], new_set[k]]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                T *= 0.999
                continue
            
            # Check maximality
            if _is_maximal(new_set, iso, all_points):
                # Same size, but maybe better structure
                if rng.random() < 0.1:  # Accept with some probability to explore
                    current_set = new_set
                    if current_size < best_size:
                        best_size = current_size
                        best_set = list(current_set)
            
            T *= 0.999
    
    # Final check
    if best_set is None:
        # Fallback: greedy in row-major
        order = list(range(npts))
        best_set = _greedy_construction(order, iso)
    
    # Convert indices back to (x, y) tuples
    result = [(pts[p][0], pts[p][1]) for p in best_set]
    
    return result
