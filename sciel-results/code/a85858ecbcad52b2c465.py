import numpy as np
import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    # Precompute squared distances between all pairs of grid points
    # Total points: N*N = 4096
    # We'll use a vectorized approach for checking validity
    
    # For a set of points P, the condition "no three points form an isosceles triangle"
    # means: for any distinct a, b, c in P, the three distances da=|a-b|, db=|b-c|, dc=|c-a|
    # are all distinct. Equivalently, no two of the three distances are equal.
    # 
    # When adding a new point p to an existing set S, we need to check:
    # For every pair (a, b) in S: dist(p,a) != dist(p,b) AND dist(a,b) != dist(p,a) AND dist(a,b) != dist(p,b)
    # Wait, that's not quite right. The triangle is (a, b, p). The three sides are:
    #   d(a,b), d(a,p), d(b,p)
    # None of these three can be equal.
    # So we need: d(a,b) != d(a,p), d(a,b) != d(b,p), d(a,p) != d(b,p)
    #
    # For a fixed p and fixed a, as we vary b in S:
    #   d(a,p) is fixed, so we need d(a,b) != d(a,p) for all b in S, b != a
    #   and d(b,p) != d(a,p) for all b in S, b != a
    #   and d(a,b) != d(b,p) for all b in S, b != a
    #
    # This is complex. Let's think of it differently.
    #
    # Alternative: For a candidate point p and existing set S:
    # For each a in S:
    #   Let d_ap = dist_sq(p, a)
    #   For each b in S, b != a:
    #     Let d_ab = dist_sq(a, b)
    #     Let d_bp = dist_sq(b, p)
    #     Check: d_ab != d_ap, d_ab != d_bp, d_ap != d_bp
    #
    # This is O(|S|^2) per candidate point, which is too slow for large S.
    #
    # Better approach: Precompute for each point p, the set of squared distances to all other grid points.
    # Then for a set S, when adding p:
    #   For each a in S:
    #     d_ap is known
    #     We need: no b in S (b != a) such that d_ab == d_ap OR d_bp == d_ap OR d_ab == d_bp
    #
    # Let's use a different incremental approach:
    # Maintain for each point a in S, the set of squared distances from a to all other points in S.
    # When adding p:
    #   For each a in S:
    #     d_ap = dist_sq(p, a)
    #     Check if d_ap is already in dists_from_a (i.e., there exists b in S, b != a, with dist_sq(a,b) == d_ap)
    #     Also check if there exists b in S, b != a, with dist_sq(b, p) == d_ap
    #     Also check if there exists b in S, b != a, with dist_sq(a, b) == dist_sq(b, p)
    #
    # This is still O(|S|^2) in the worst case.
    #
    # Let's try a different strategy: use simulated annealing with incremental checks.
    # We'll maintain the set of points and for each point, track the multiset of squared distances to other points in the set.
    
    # Precompute squared distances
    xs = np.arange(N)
    ys = np.arange(N)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([XX.ravel(), YY.ravel()], axis=1)
    num_points = N * N
    
    # Squared distance matrix
    # This is 4096 x 4096 = 16 million entries, which is ~128 MB as int32. Acceptable.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dist = np.sum(diff * diff, axis=2)
    # sq_dist[i, j] = squared distance from point i to point j
    
    # For efficiency, let's use a greedy + local search approach.
    # Start with a random valid set, then try to add more points or swap.
    
    # Greedy initialization: pick points one by one, checking validity
    # To speed up the check, we'll use the following:
    # When adding point p to set S:
    #   For each a in S:
    #     d_ap = sq_dist[p, a]
    #     We need to ensure no b in S (b != a) has:
    #       sq_dist[a, b] == d_ap  (isosceles with apex at a)
    #       sq_dist[b, p] == d_ap  (isosceles with apex at p)
    #       sq_dist[a, b] == sq_dist[b, p]  (isosceles with apex at b)
    
    # Let's build a helper that checks if a point can be added
    def can_add(p, S_indices):
        """Check if point p can be added to set S_indices (list of point indices)."""
        S_set = set(S_indices)
        for a in S_indices:
            d_ap = sq_dist[p, a]
            for b in S_indices:
                if b == a:
                    continue
                d_ab = sq_dist[a, b]
                d_bp = sq_dist[b, p]
                if d_ab == d_ap or d_ab == d_bp or d_ap == d_bp:
                    return False
        return True
    
    # This is O(|S|^2) per check. For |S| ~ 20, that's 400 checks per candidate.
    # With 4096 candidates, that's ~1.6 million distance comparisons per greedy step.
    # This might be too slow in pure Python.
    
    # Let's vectorize the check for a given p and set S.
    def can_add_vectorized(p, S_indices):
        """Vectorized check if point p can be added to set S_indices."""
        if len(S_indices) == 0:
            return True
        S_arr = np.array(S_indices)
        # d_ap: distances from p to each point in S
        d_ap = sq_dist[p, S_arr]
        # d_ab: distances between pairs in S
        # We need to check for each a in S:
        #   d_ap[a] not in {d_ab[a, b] for b != a}
        #   d_ap[a] not in {d_bp[b] for b != a} where d_bp = sq_dist[p, S_arr]
        #   d_ab[a, b] != d_bp[b] for all b != a
        
        # d_ap is shape (|S|,)
        # For the first condition: for each a, check if d_ap[a] appears in d_ab[a, :] (excluding b=a)
        # d_ab is |S| x |S|
        d_ab = sq_dist[np.ix_(S_arr, S_arr)]
        d_bp = d_ap  # same as d_ap since distance is symmetric
        
        # Condition 1: d_ap[a] != d_ab[a, b] for all b != a
        # Create mask where b == a
        mask_diag = np.eye(len(S_arr), dtype=bool)
        d_ab_off = d_ab.copy()
        d_ab_off[mask_diag] = -1  # sentinel
        
        # For each a, check if d_ap[a] is in d_ab_off[a, :]
        # This is a bit tricky to vectorize directly.
        # Alternative: for each a, count how many b have d_ab[a, b] == d_ap[a]
        # If count > 0, invalid.
        
        # Let's do it in a loop over a, but vectorize over b
        for i, a in enumerate(S_arr):
            dap = d_ap[i]
            # Check condition 1: any b != a with d_ab[a, b] == dap
            if np.any(d_ab[i, :] == dap):
                # Need to exclude b == a
                if i in S_arr:
                    # d_ab[i, i] = 0, so if dap == 0, this would be a problem, but dap > 0 for p != a
                    pass
                # Check if the match is at b == a
                matches = np.where(d_ab[i, :] == dap)[0]
                valid_matches = [m for m in matches if S_arr[m] != a]
                if valid_matches:
                    return False
            
            # Check condition 2: any b != a with d_bp[b] == dap
            # d_bp = d_ap, so check if any d_ap[j] == dap for j != i
            if i > 0 and np.any(d_ap[:i] == dap):
                return False
            if i < len(d_ap) - 1 and np.any(d_ap[i+1:] == dap):
                return False
            
            # Check condition 3: any b != a with d_ab[a, b] == d_bp[b]
            # d_bp[b] = d_ap[j] where j is index of b in S_arr
            for j, b in enumerate(S_arr):
                if j == i:
                    continue
                if d_ab[i, j] == d_ap[j]:
                    return False
        
        return True
    
    # This is still O(|S|^2) but with numpy operations. Let's try a simpler approach.
    # Given the time constraints, let's use a randomized greedy with limited checks.
    
    # Actually, let's try a different approach: use a known construction.
    # For the no-isosceles problem, one approach is to use points on a parabola or other curves.
    # But let's stick with random + local search.
    
    # Let's use a simpler can_add that's O(|S|^2) but in pure Python, and limit the set size.
    
    def can_add_simple(p, S_indices):
        """Simple O(|S|^2) check."""
        for a in S_indices:
            d_ap = sq_dist[p, a]
            for b in S_indices:
                if b == a:
                    continue
                d_ab = sq_dist[a, b]
                d_bp = sq_dist[b, p]
                if d_ab == d_ap or d_ab == d_bp or d_ap == d_bp:
                    return False
        return True
    
    # Greedy: pick points in random order, add if valid
    random.seed(42)
    S = []
    candidates = list(range(num_points))
    random.shuffle(candidates)
    
    for p in candidates:
        if time.time() > deadline - 60:  # Leave 60s for local search
            break
        if len(S) > 25:  # Limit size for speed
            break
        if can_add_simple(p, S):
            S.append(p)
    
    best_S = S[:]
    best_score = len(S)
    
    # Local search: try to add more points or swap
    # Try adding points
    for p in candidates:
        if time.time() > deadline - 30:
            break
        if p in best_S:
            continue
        if can_add_simple(p, best_S):
            best_S.append(p)
            best_score = len(best_S)
    
    # Try local search: remove a point, try to add two
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        if not best_S:
            break
        # Remove a random point
        idx = random.randint(0, len(best_S) - 1)
        removed = best_S[idx]
        best_S.pop(idx)
        
        # Try to add points
        added_any = False
        for p in candidates:
            if time.time() > deadline - 10:
                break
            if p in best_S:
                continue
            if can_add_simple(p, best_S):
                best_S.append(p)
                added_any = True
                if len(best_S) > best_score:
                    best_score = len(best_S)
                break
        
        if not added_any:
            # Put the removed point back
            best_S.append(removed)
    
    # Convert to (x, y) tuples
    result = [tuple(points[i]) for i in best_S]
    return result
# EVOLVE-BLOCK-END
