"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    all_points = list(itertools.product(range(N), repeat=2))
    M = len(all_points)
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    
    # Precompute conflict masks
    # conflict_mask[i] is a bitmask of all points j such that {i, j} cannot be in the same no-isosceles set
    # i.e., there exists k such that {i, j, k} is isosceles.
    conflict_mask = [0] * M
    
    # Optimization: 
    # For a fixed pair (i, j), they conflict if there is a k such that d(i,k)==d(j,k) or d(i,j)==d(i,k) or d(i,j)==d(j,k).
    # We can iterate over all triples (i, j, k) but that is O(M^3) = 256^3 ~ 16M operations, which is fine.
    # Or iterate over i, j and check if there's a k.
    
    # Let's use a slightly smarter approach:
    # For each i, j:
    #   Check if there exists k != i, j such that:
    #   1. d(i, k) == d(j, k)
    #   2. d(i, j) == d(i, k)
    #   3. d(i, j) == d(j, k)
    
    # To make this faster, we can precompute distances.
    dist = [[0]*M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            dist[i][j] = _dist2(all_points[i], all_points[j])
            
    for i in range(M):
        for j in range(M):
            if i == j: continue
            # Check condition 2 and 3:
            # d(i, j) == d(i, k) => k is on circle centered at i with radius d(i, j)
            # d(i, j) == d(j, k) => k is on circle centered at j with radius d(i, j)
            # Check condition 1:
            # d(i, k) == d(j, k) => k is on perpendicular bisector of ij
            
            # It is easier to just iterate k for small M? 256*256*256 is 16 million. 
            # In Python, 16M iterations might take 1-2 seconds. Acceptable.
            # But we can do better.
            
            # Let's just do the triple loop but break early if found.
            # Actually, we can just check if ANY k exists.
            # For efficiency, we can precompute for each i, a map from dist to list of points.
            
    # Precompute dist_maps: dist_maps[i][d] = bitmask of points k such that d(i, k) == d
    dist_maps = [{} for _ in range(M)]
    for i in range(M):
        for j in range(M):
            if i == j: continue
            d = dist[i][j]
            if d not in dist_maps[i]:
                dist_maps[i][d] = 0
            dist_maps[i][d] |= (1 << j)

    # Now compute conflict masks
    for i in range(M):
        for j in range(M):
            if i == j: continue
            
            # Condition 2: exists k != i, j such that d(i, k) == d(i, j)
            d_ij = dist[i][j]
            # Points k such that d(i, k) == d_ij
            mask_ci = dist_maps[i].get(d_ij, 0)
            # Exclude i and j
            mask_ci &= ~(1 << i)
            mask_ci &= ~(1 << j)
            
            # Condition 3: exists k != i, j such that d(j, k) == d(i, j)
            mask_cj = dist_maps[j].get(d_ij, 0)
            mask_cj &= ~(1 << i)
            mask_cj &= ~(1 << j)
            
            # Condition 1: exists k != i, j such that d(i, k) == d(j, k)
            # This is harder to precompute directly with simple dist_maps.
            # We can iterate over k? Or precompute perpendicular bisectors?
            # Let's iterate k for condition 1 only if conditions 2 and 3 didn't find a conflict.
            # Actually, we can just check:
            # Is there any k such that d(i, k) == d(j, k)?
            # This is equivalent to: intersection of circles?
            # Let's just do a quick check for condition 1 by iterating k?
            # No, let's optimize:
            # If mask_ci or mask_cj is non-zero, we have a conflict.
            if mask_ci or mask_cj:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                continue
            
            # Check condition 1: d(i, k) == d(j, k)
            # We can iterate k from 0 to M-1
            found = False
            for k in range(M):
                if k == i or k == j: continue
                if dist[i][k] == dist[j][k]:
                    found = True
                    break
            
            if found:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    all_points_mask = (1 << M) - 1

    def is_maximal(S_mask):
        """
        S_mask: bitmask of chosen points.
        Returns True if S is maximal (no point outside S can be added).
        """
        # A point r outside S can be added if it does not conflict with any point in S.
        # i.e., for all p in S, r is NOT in conflict_mask[p].
        # Equivalently, r is not in the union of conflict_mask[p] for p in S.
        
        # Calculate union of conflicts
        union_conflicts = 0
        # Iterate through set bits in S_mask
        temp = S_mask
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            union_conflicts |= conflict_mask[idx]
            temp ^= lsb
            
        # Points that are "blocked" by S are those in union_conflicts.
        # Points that are in S are "occupied".
        # A point r is addable if:
        # 1. r is not in S
        # 2. r is not in union_conflicts
        
        # Addable points = AllPoints & ~S & ~union_conflicts
        addable = all_points_mask & ~S_mask & ~union_conflicts
        
        return addable == 0

    def is_valid(S_mask):
        """
        Check if S_mask represents a no-isosceles set.
        We assume it's built incrementally, so we might not need this if we check during construction.
        But for local search, we might remove points, so we need to be sure.
        Actually, if S is a subset of a valid set, it is valid.
        So if we start with a valid set and only remove points, it remains valid.
        So we only need to check maximality.
        """
        return True

    best_mask = 0
    best_size = float('inf')
    
    # Greedy Construction with Bitmasks
    # We need to check if adding point p to current set S is valid.
    # p is valid to add if p is not in conflict_mask[q] for any q in S.
    # i.e., p is not in union_conflicts(S).
    
    def can_add(p_idx, S_mask):
        # Calculate union of conflicts for S
        # This is O(|S|)
        # Optimization: Keep a running union?
        # For greedy, we can just check against each q in S.
        # p conflicts with q if (conflict_mask[p] >> q) & 1
        # So we need to check if any q in S has bit p set in conflict_mask[q]?
        # No, conflict is symmetric.
        # Check if p is in conflict_mask[q] for any q in S.
        # This is equivalent to: (union_conflicts(S) >> p) & 1
        
        # Let's just iterate q in S for simplicity in greedy, as |S| is small.
        temp = S_mask
        while temp:
            lsb = temp & -temp
            q_idx = lsb.bit_length() - 1
            if (conflict_mask[q_idx] >> p_idx) & 1:
                return False
            temp ^= lsb
        return True

    def greedy_maximal(order, deadline):
        S_mask = 0
        for p_idx in order:
            if time.time() > deadline:
                return None
            if can_add(p_idx, S_mask):
                S_mask |= (1 << p_idx)
        return S_mask

    # Initial deterministic run
    order = list(range(M))
    result_mask = greedy_maximal(order, deadline)
    if result_mask is not None:
        if is_maximal(result_mask):
            best_mask = result_mask
            best_size = bin(result_mask).count('1')
    
    # Local Search / Pruning
    def local_search(S_mask, deadline):
        """
        Try to remove points to minimize size while maintaining maximality.
        """
        S_mask = S_mask
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each point
            temp = S_mask
            while temp:
                lsb = temp & -temp
                p_idx = lsb.bit_length() - 1
                temp ^= lsb
                
                # Try removing p_idx
                new_mask = S_mask & ~lsb
                if is_maximal(new_mask):
                    S_mask = new_mask
                    improved = True
                    break # Restart loop as S_mask changed
            
            # If no single removal worked, try removing pairs?
            # This is expensive. Let's stick to single removals first.
            # If stuck, try randomized restarts.
            
        return S_mask

    # Randomized Restart
    rng = random.Random(42)
    num_trials = 10000
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        order = list(range(M))
        rng.shuffle(order)
        
        # To speed up greedy, we can use a randomized priority or just shuffle.
        # The "can_add" check is the bottleneck.
        # We can optimize can_add by maintaining the union of conflicts?
        # Let's try to optimize the inner loop of greedy.
        
        S_mask = 0
        # We can pre-compute the "conflict union" incrementally?
        # No, because adding a point changes the union.
        # But we can check `can_add` faster if we keep the set of "forbidden" points?
        # Forbidden points = union of conflict_mask[q] for q in S.
        # If p is not in Forbidden, we can add it.
        # Then new Forbidden = Forbidden | conflict_mask[p].
        
        forbidden = 0
        for p_idx in order:
            if time.time() > deadline:
                break
            if not (forbidden >> p_idx) & 1:
                # p_idx is not forbidden
                S_mask |= (1 << p_idx)
                forbidden |= conflict_mask[p_idx]
        
        if is_maximal(S_mask):
            # Try to prune
            S_mask = local_search(S_mask, deadline)
            
            size = bin(S_mask).count('1')
            if size < best_size:
                best_size = size
                best_mask = S_mask
                if best_size <= 8:
                    break

    # Convert best_mask back to list of points
    best_points = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        best_points.append(all_points[idx])
        temp ^= lsb
        
    return best_points
# EVOLVE-BLOCK-END
