def construct():
    import numpy as np
    import random
    from itertools import combinations

    N = 16
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_index = {p: i for i, p in enumerate(all_points)}
    n_pts = len(all_points)
    
    # Precompute squared distances
    # dist[i][j] is squared distance between point i and j
    dist = np.zeros((n_pts, n_pts), dtype=int)
    for i in range(n_pts):
        x1, y1 = all_points[i]
        for j in range(i+1, n_pts):
            x2, y2 = all_points[j]
            d = (x1-x2)**2 + (y1-y2)**2
            dist[i][j] = d
            dist[j][i] = d

    # Precompute isosceles conflicts
    # conflict_mask[i] is a bitset (or list) of points j such that there exists k 
    # where {i, j, k} is isosceles? 
    # Actually, we need to check if adding P to S creates an isosceles triangle.
    # This happens if there exist A, B in S such that {P, A, B} is isosceles.
    # So for a fixed P, and a pair (A, B), we check isosceles.
    
    # Let's precompute for each pair (A, B) the set of points C that form an isosceles triangle with A and B.
    # iso_conflict[A, B] = set of C
    # Since n_pts=256, storing sets for each pair is heavy but manageable.
    # Or, we can just check on the fly with optimized distance checks.
    
    # Given N=16, n_pts=256.
    # Greedy loop:
    # S = []
    # While S is not maximal:
    #   Find all P not in S such that S U {P} is valid.
    #   For each valid P, calculate score (blocking power).
    #   Add P with max score.
    
    # To check validity of P given S:
    # For all A, B in S, check if {P, A, B} is isosceles.
    
    # To calculate blocking power of P:
    # Count Q not in S such that for some A in S, {P, Q, A} is isosceles.
    # Note: Q is blocked if it CANNOT be added to S U {P}.
    # Q cannot be added if there exist A, B in S U {P} such that {Q, A, B} is isosceles.
    # Since S is valid, Q is only blocked by pairs involving P.
    # So Q is blocked if (exists A in S: {Q, P, A} is isosceles) OR (exists B, C in S: {Q, B, C} is isosceles).
    # The second condition is independent of P. Let's call a point Q "already blocked" if it forms an isosceles triangle with any pair in S.
    # If Q is already blocked, adding P doesn't change its status.
    # So we only count Q that are NOT already blocked, but become blocked because of P.
    # i.e., Q is not in S, not already blocked by S, but {Q, P, A} is isosceles for some A in S.

    # Optimization:
    # Precompute for each point P and each other point A, the set of points Q such that {P, A, Q} is isosceles.
    # Let's call this iso_with(P, A).
    # Then if we add P to S, the new blocked points are union of iso_with(P, A) for all A in S.
    
    # Let's precompute iso_with(P, A) as a bitset or set of indices.
    # n_pts = 256. 256 * 256 = 65536 pairs.
    # For each pair (P, A), we need to find Q such that dist(P,Q)==dist(A,Q) or dist(P,Q)==dist(P,A) or dist(A,Q)==dist(P,A).
    
    # This precomputation might take a few seconds.
    # Let's do it.
    
    iso_with = [None] * (n_pts * n_pts)
    # Flatten index: idx = P * n_pts + A
    
    for P in range(n_pts):
        for A in range(P+1, n_pts):
            d_PA = dist[P, A]
            mask = set()
            for Q in range(n_pts):
                if Q == P or Q == A:
                    continue
                d_PQ = dist[P, Q]
                d_AQ = dist[A, Q]
                if d_PQ == d_AQ or d_PQ == d_PA or d_AQ == d_PA:
                    mask.add(Q)
            iso_with[P * n_pts + A] = mask
            iso_with[A * n_pts + P] = mask

    def is_valid_addition(P, S_indices):
        # S_indices is a set of indices
        for A in S_indices:
            for B in S_indices:
                if A == B: continue
                # Check if {P, A, B} is isosceles
                # We can use the precomputed data?
                # iso_with[A, B] contains C such that {A, B, C} is isosceles.
                # So we just check if P is in iso_with[A, B].
                if P in iso_with[A * n_pts + B]:
                    return False
        return True

    def get_newly_blocked(P, S_indices, already_blocked_mask):
        # Returns set of points Q that are newly blocked by adding P
        newly = set()
        for A in S_indices:
            # Get all Q that form isosceles with P and A
            mask_PA = iso_with[P * n_pts + A]
            # Intersect with points not in S and not already blocked
            for Q in mask_PA:
                if Q not in S_indices and Q not in already_blocked_mask:
                    newly.add(Q)
        return newly

    # Greedy Construction
    S_indices = set()
    already_blocked_mask = set()
    
    # To ensure determinism or good results, we can try a few random orderings or just greedy.
    # The greedy choice of max blocking power is deterministic if tie-breaking is fixed.
    
    while True:
        # Find all valid P
        candidates = []
        for P in range(n_pts):
            if P in S_indices:
                continue
            if is_valid_addition(P, S_indices):
                # Calculate blocking power
                newly_blocked = get_newly_blocked(P, S_indices, already_blocked_mask)
                score = len(newly_blocked)
                candidates.append((score, P, newly_blocked))
        
        if not candidates:
            break
        
        # Pick max score
        # Tie-break by index for determinism
        candidates.sort(key=lambda x: (-x[0], x[1]))
        best_score, best_P, best_newly_blocked = candidates[0]
        
        # Add best_P
        S_indices.add(best_P)
        already_blocked_mask.update(best_newly_blocked)
        
        # Check if maximal?
        # If no candidates, it means no point can be added.
        # Is it maximal? Yes, if no point can be added, it is maximal.
        # Wait, the loop breaks if no candidates.
        # So when the loop finishes, S is maximal.

    result = [all_points[i] for i in S_indices]
    return result
