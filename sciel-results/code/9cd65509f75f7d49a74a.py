import numpy as np
import time
from itertools import combinations, product
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # E8 x D3 lattice construction with mixed vectors
    # E8 part: 240 vectors of norm 4 (weight 4, ±1 entries)
    # D3 part: 12 vectors of norm 2 (weight 2, ±1 entries in last 3 dims)
    # Mixed: vectors with norm sqrt(8) that have entries in both E8 and D3 parts
    
    # First, generate the standard E8 x D3 lattice vectors
    # E8: all weight-4 vectors with ±1 entries
    e8_vectors = []
    for support in combinations(range(8), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            e8_vectors.append(v)
    
    # D3: all weight-2 vectors in dimensions 8,9,10
    d3_vectors = []
    for support in combinations(range(8, 11), 2):
        for signs in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            d3_vectors.append(v)
    
    # Mixed vectors: weight-3 vectors with one entry in dims 0-7 and two in dims 8-10
    # OR weight-5 vectors with three in dims 0-7 and two in dims 8-10
    # These have norm sqrt(3) or sqrt(5) which won't work for our constraint
    # Actually, we need vectors of norm 2 (squared norm 4) that are compatible
    # Let me think about this differently.
    
    # The constraint is: min pairwise distance >= max norm
    # If all vectors have the same norm, this becomes: min pairwise distance >= norm
    # For unit vectors, distance^2 = 2 - 2*cos(theta), so we need 2-2*cos >= 1, i.e., cos <= 0.5
    # For integer vectors of norm 2: distance^2 >= 4, so (v1-v2)^2 >= 4
    # v1^2 + v2^2 - 2*v1.v2 >= 4 => 4 + 4 - 2*v1.v2 >= 4 => v1.v2 <= 2
    # So we need |v1.v2| <= 2 for all pairs (with v1 != v2)
    
    # For the E8 vectors (norm 4, squared norm 16):
    # distance^2 >= 16 => 16+16-2*v1.v2 >= 16 => v1.v2 <= 8
    # Two E8 vectors with disjoint supports have dot product 0 ✓
    # Two E8 vectors sharing 1 index: dot = ±1 ✓
    # Two E8 vectors sharing 2 indices: dot = ±1 or ±2 ✓
    # Two E8 vectors sharing 3 indices: dot = ±1, ±3, or ±5 - need to check
    # Actually for weight-4 vectors, max shared indices is 3
    # If they share 3 indices with same signs: dot = 3 ✓
    # If they share 3 indices with 2 same, 1 diff: dot = 1 ✓
    # If they share 3 indices with 1 same, 2 diff: dot = -1 ✓
    # If they share 3 indices with all diff: dot = -3 ✓
    # So any two E8 vectors are compatible!
    
    # For D3 vectors (norm 2, squared norm 4):
    # distance^2 >= 4 => 4+4-2*v1.v2 >= 4 => v1.v2 <= 2
    # Two D3 vectors in dims 8-10: max dot product is 2 (if same support, same signs)
    # But if same support and same signs, they're the same vector!
    # If same support, opposite signs: dot = -2 ✓
    # If different supports (disjoint in 3D is impossible for weight-2): share 1 index
    # dot = ±1 ✓
    # So any two D3 vectors are compatible!
    
    # E8 vs D3: dot product is always 0 (disjoint supports) ✓
    
    # So we can include ALL E8 and D3 vectors: 240 + 12 = 252
    # But we can also add mixed vectors!
    
    # Mixed vectors of norm 2: must have squared norm 4
    # Options: weight 2 (±1,±1) or weight 4 (±1,±1,±1,±1)
    # We already have weight-2 in D3 and weight-4 in E8
    # What about weight-2 with one in 0-7 and one in 8-10?
    mixed_vectors = []
    for i in range(8):
        for j in range(8, 11):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = si
                    v[j] = sj
                    mixed_vectors.append(v)
    
    # Check if mixed vectors are compatible with everything
    # Mixed vs E8: dot product = ±1 (if E8 has that index) or 0
    # Max |dot| = 1 <= 2 ✓ (for E8 norm 4: need dot <= 8, easily satisfied)
    # Mixed vs D3: dot product = ±1 or 0
    # Max |dot| = 1 <= 2 ✓
    # Mixed vs Mixed: two mixed vectors (i,a) and (j,b) where i,j in 0-7, a,b in 8-10
    # If i=j and a=b: same vector
    # If i=j, a!=b: dot = si*sj = ±1 ✓
    # If i!=j, a=b: dot = 0 ✓
    # If i!=j, a!=b: dot = 0 ✓
    # So all mixed vectors are compatible!
    
    # Total: 240 + 12 + 96 = 348 vectors
    all_vectors = e8_vectors + d3_vectors + mixed_vectors
    
    # Verify the constraint
    if len(all_vectors) > 0:
        arr = np.array(all_vectors, dtype=np.int32)
        max_norm_sq = np.max(np.sum(arr**2, axis=1))
        # Compute all pairwise distances squared
        # For efficiency, check min distance
        min_dist_sq = float('inf')
        for i in range(len(arr)):
            if time.time() > deadline - 5.0:
                break
            diffs = arr[i+1:] - arr[i]
            dists_sq = np.sum(diffs**2, axis=1)
            if len(dists_sq) > 0:
                min_dist_sq = min(min_dist_sq, np.min(dists_sq))
        
        if min_dist_sq >= max_norm_sq:
            return np.array(all_vectors, dtype=np.int8)
    
    # Fallback: use MIS-based approach if verification fails
    
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)  # 330
    
    # Precompute support sets for intersection checks
    sup_sets = [set(s) for s in supports]
    
    # Build incompatibility graph: two supports are incompatible if |intersection| >= 3
    incompat = np.zeros((N_sup, N_sup), dtype=np.uint8)
    
    for i in range(N_sup):
        if time.time() > deadline - 30.0:
            break
        for j in range(i + 1, N_sup):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                incompat[i, j] = 1
                incompat[j, i] = 1
    
    # Build neighbor lists for faster access
    neighbors = [set(np.where(incompat[i] == 1)[0]) for i in range(N_sup)]
    # Precompute incompatibility as a list of lists for fast membership tests
    incompat_lists = [list(np.where(incompat[i] == 1)[0]) for i in range(N_sup)]
    # Precompute the set of incompatible indices as a numpy boolean matrix row view
    incompat_bool = incompat.astype(bool)
    
    def greedy_mis_min_degree(available, rng):
        """Greedy MIS using minimum degree heuristic (fast: maintain degree counts)."""
        selected = set()
        avail = set(available)
        # degree[u] = number of neighbors of u that are still in avail
        deg = {}
        for u in avail:
            deg[u] = len(neighbors[u] & avail)
        
        while avail:
            if time.time() > deadline - 15.0:
                break
            
            # Find node with minimum degree
            min_deg = float('inf')
            candidates = []
            for u in avail:
                d = deg[u]
                if d < min_deg:
                    min_deg = d
                    candidates = [u]
                elif d == min_deg:
                    candidates.append(u)
            
            # Pick randomly from minimum-degree candidates
            u = rng.choice(candidates)
            selected.add(u)
            
            # Remove u and its neighbors; update degrees of remaining avail nodes
            to_remove = {u} | (neighbors[u] & avail)
            for v in to_remove:
                avail.discard(v)
            # For each remaining node w, if u was a neighbor, decrement deg[w]
            for w in avail:
                if u in neighbors[w]:
                    deg[w] -= 1
            # Also need to account for removed neighbors other than u
            # Actually, we only removed u and its neighbors. For a remaining node w,
            # its neighbors that were removed are exactly {u} ∩ neighbors[w] plus any
            # neighbor of u that is also a neighbor of w.
            # Let's be more careful: for each removed node r (other than u), if r is
            # a neighbor of w, then deg[w] should be decremented.
            # But we already decremented for u. Let's redo:
            # Reset: for each w in avail, deg[w] = number of neighbors of w in avail
            # That's expensive. Instead, let's just recompute deg for all avail nodes
            # when the set changes significantly. Actually, the current approach of
            # only decrementing for u is incorrect because we also removed neighbors of u.
            
            # Let me use a simpler but still fast approach: recompute degrees only
            # when the set is small, otherwise use the original method.
            # Actually, let's just use the original set-intersection method but with
            # a precomputed incompatibility count via numpy.
            pass
        
        # The above is getting complex. Let me use a cleaner approach:
        # Use numpy to compute degrees in bulk.
        selected = set()
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 15.0:
                break
            
            if len(avail) > 200:
                # Use numpy for bulk degree computation
                avail_arr = np.array(list(avail), dtype=np.int32)
                # degree of each avail node = number of incompatibilities with other avail nodes
                deg_arr = incompat_bool[avail_arr][:, avail_arr].sum(axis=1)
                min_deg = deg_arr.min()
                min_mask = deg_arr == min_deg
                candidates = avail_arr[min_mask]
                u = int(rng.choice(candidates))
            else:
                # Small set: use direct computation
                min_deg = float('inf')
                candidates = []
                for u in avail:
                    d = len(neighbors[u] & avail)
                    if d < min_deg:
                        min_deg = d
                        candidates = [u]
                    elif d == min_deg:
                        candidates.append(u)
                u = rng.choice(candidates)
            
            selected.add(u)
            
            # Remove u and its neighbors
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove
            
        return selected
    
    def local_search_add(selected, rng):
        """Try to add vertices that are compatible with all current selected vertices (vectorized)."""
        selected = set(selected)
        
        if not selected:
            return selected
        
        sel_arr = np.array(list(selected), dtype=np.int32)
        # A node x can be added if it has no incompatibilities with any selected node
        # incompat_bool[x, sel_arr] should be all False
        compatible_mask = ~incompat_bool[:, sel_arr].any(axis=1)
        # Exclude already-selected nodes
        compatible_mask[sel_arr] = False
        
        addable = np.where(compatible_mask)[0]
        for x in addable:
            if time.time() > deadline - 10.0:
                break
            selected.add(int(x))
        
        return selected
    
    def local_search_1_2_swap(selected, rng):
        """Improve MIS using 1-2 swaps: remove one vertex, add two compatible vertices."""
        selected = set(selected)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            for v in list(selected):
                if time.time() > deadline - 10.0:
                    break
                
                # Find all nodes compatible with (selected - {v})
                potential_adds = []
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    # x must be compatible with all y in selected - {v}
                    if not (neighbors[x] & (selected - {v})):
                        potential_adds.append(x)
                
                # Find a pair (u1, u2) in potential_adds that are compatible with each other
                found = False
                n_pot = len(potential_adds)
                for i in range(n_pot):
                    u1 = potential_adds[i]
                    for j in range(i + 1, n_pot):
                        u2 = potential_adds[j]
                        if incompat[u1, u2] == 0:
                            selected.remove(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                
                if improved:
                    break  # Restart the outer loop
        
        return selected
    
    def local_search_2_1_swap(selected, rng):
        """Improve MIS using 2-1 swaps: remove two vertices, add one compatible vertex."""
        selected = set(selected)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            sel_list = list(selected)
            n_sel = len(sel_list)
            
            for i in range(n_sel):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, n_sel):
                    v1 = sel_list[i]
                    v2 = sel_list[j]
                    remaining = selected - {v1, v2}
                    
                    # Find a node x compatible with all in remaining
                    for x in range(N_sup):
                        if x in selected:
                            continue
                        if not (neighbors[x] & remaining):
                            selected.remove(v1)
                            selected.remove(v2)
                            selected.add(x)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return selected
    
    best_iset = set()
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different strategies
    n_restarts = 120
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if r == 0:
            # First run: pure greedy min-degree
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
        elif r < 40:
            # Greedy + local search
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        elif r < 80:
            # Perturbed greedy: start with random subset removed
            avail = set(range(N_sup))
            n_remove = rng.integers(5, 30)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        else:
            # More aggressive perturbation
            avail = set(range(N_sup))
            n_remove = rng.integers(20, 60)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = set(iset)
    
    # Final improvement attempts on best solution
    if time.time() < deadline - 10.0:
        best_iset = local_search_add(best_iset, rng)
        best_iset = local_search_1_2_swap(best_iset, rng)
    
    # One more round of local search if time permits
    if time.time() < deadline - 5.0:
        best_iset = local_search_add(best_iset, rng)
        best_iset = local_search_1_2_swap(best_iset, rng)
    
    # Generate vectors: for each support in the MIS, include all 2^4 = 16 sign patterns
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
    
    if not vectors:
        # Fallback: return a small valid set
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 1
                v[j] = 1
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int8)
