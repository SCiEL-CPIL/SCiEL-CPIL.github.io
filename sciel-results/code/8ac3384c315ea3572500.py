# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses precomputed field tables and vectorized operations for speed.
    """
    deadline = time.time() + 600  # Safety margin

    p = 7
    q = p * p  # 49
    k_field = 3  # alpha^2 = 3

    # --- 1. Precompute Field Tables ---
    # Element x = a + b*alpha, encoded as a + b*p
    # to_pair(x) -> (x % p, x // p)
    
    # Addition table: add_table[x, y] = x + y
    add_table = np.zeros((q, q), dtype=np.int32)
    for x in range(q):
        xa, xb = x % p, x // p
        for y in range(q):
            ya, yb = y % p, y // p
            add_table[x, y] = ((xa + ya) % p) + ((xb + yb) % p) * p

    # Multiplication table: mul_table[x, y] = x * y
    mul_table = np.zeros((q, q), dtype=np.int32)
    for x in range(q):
        xa, xb = x % p, x // p
        for y in range(q):
            ya, yb = y % p, y // p
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            mul_table[x, y] = real + imag * p

    # Inverse table: inv_table[x] = 1/x (0 for x=0)
    inv_table = np.zeros(q, dtype=np.int32)
    for x in range(1, q):
        xa, xb = x % p, x // p
        denom = (xa * xa - xb * xb * k_field) % p
        denom_inv = pow(denom, p - 2, p)
        real = (xa * denom_inv) % p
        imag = (-xb * denom_inv) % p
        inv_table[x] = real + imag * p

    # --- 2. Precompute Lines ---
    # Directions: v = (1, m) for m in 0..48, and v = (0, 1)
    # Total 50 directions.
    n_dirs = q + 1
    n_points = q * q
    
    # lines[dir_idx][point_idx] = array of q-1 point indices on the line through point_idx in direction dir_idx
    # point_idx = x + y*q
    lines = np.zeros((n_dirs, n_points, q - 1), dtype=np.int32)
    
    for d in range(n_dirs):
        if d < q:
            vx, vy = 1, d
        else:
            vx, vy = 0, 1
            
        for x in range(q):
            for y in range(q):
                p_idx = x + y * q
                # t = 1..48
                # point = (x + t*vx, y + t*vy)
                # We need to compute this for all t at once for this (x,y)
                ts = np.arange(1, q, dtype=np.int32)
                txs = (add_table[x, mul_table[vx, ts]])  # x + t*vx
                tys = (add_table[y, mul_table[vy, ts]])  # y + t*vy
                # Convert (tx, ty) to index
                # tx, ty are in 0..48
                # idx = tx + ty*q
                idxs = txs + tys * q
                lines[d, p_idx] = idxs

    # --- 3. Initialize Base Nikodym Set ---
    # Hermitian Unital construction
    # Trace(x) = 2*xa
    # Norm(y) = ya^2 - yb^2*k
    # Unital U: Trace(x) + Norm(y) = 0
    # Excluded S: y in U and y_b is NOT a square? 
    # Wait, the standard construction for small Nikodym sets often involves excluding specific points.
    # Let's use the logic from the seed:
    # y_special: y such that y_a is a square? 
    # Seed logic:
    # y_special = { (u^2, v) }
    # Excluded = { (x, y) in U | y not in y_special }
    # N = All \ Excluded
    
    # Let's stick to the seed's initialization logic but vectorize it.
    in_N = np.ones(n_points, dtype=bool)
    
    # Compute Trace for all x
    traces = np.zeros(q, dtype=np.int32)
    for x in range(q):
        xa = x % p
        traces[x] = (2 * xa) % p
        
    # Compute Norm for all y
    norms = np.zeros(q, dtype=np.int32)
    for y in range(q):
        ya, yb = y % p, y // p
        norms[y] = (ya * ya - yb * yb * k_field) % p
        
    # Determine y_special
    y_special = np.zeros(q, dtype=bool)
    for u in range(p):
        ua_sq = (u * u) % p
        for v in range(p):
            y_idx = ua_sq + v * p
            y_special[y_idx] = True

    # Mark excluded points
    # For each x, y: if (traces[x] + norms[y]) % p == 0 and not y_special[y]: exclude
    for x in range(q):
        t_x = traces[x]
        for y in range(q):
            if (t_x + norms[y]) % p == 0:
                if not y_special[y]:
                    p_idx = x + y * q
                    in_N[p_idx] = False

    # --- 4. Initialize Counts ---
    # not_in_N_count[d, p_idx] = number of points on line(d, p_idx) NOT in N
    not_in_N_count = np.zeros((n_dirs, n_points), dtype=np.int32)
    
    for d in range(n_dirs):
        # For each point p, count how many points on its line are NOT in N
        # lines[d, p] contains q-1 indices
        # We can vectorize this:
        # For a fixed d, we have a matrix of indices L of shape (n_points, q-1)
        # We want for each row i, sum( not in_N[ L[i, :] ] == False )
        # in_N is boolean. not in_N is ~in_N.
        # counts = ~in_N[ L ].sum(axis=1)
        
        # L is lines[d]
        L = lines[d]
        # in_N is 1D array of size n_points
        # We need to index in_N with L.
        # L is int32.
        # is_not_in_N = ~in_N[L]  -> shape (n_points, q-1)
        # counts = is_not_in_N.sum(axis=1)
        
        # This creates a large temporary array (2401 * 48 booleans), which is fine.
        is_not_in_N = ~in_N[L]
        not_in_N_count[d] = is_not_in_N.sum(axis=1)

    # cover_count[p_idx] = number of directions d where not_in_N_count[d, p_idx] == 0
    cover_count = (not_in_N_count == 0).sum(axis=0)

    # --- 5. Greedy Removal ---
    # We want to remove points p from N.
    # Condition for removing p:
    # For all (x, d) such that p is on line(x, d) \ {x}:
    #   After removal, not_in_N_count[d, x] increases by 1.
    #   This decreases cover_count[x] by 1 IF not_in_N_count[d, x] was 0.
    #   So, we must ensure that for all such (x, d) with not_in_N_count[d, x] == 0,
    #   cover_count[x] > 1.
    
    # To check this efficiently:
    # For each point p, we need to know which (x, d) pairs are affected.
    # p is on line(x, d) \ {x} iff x is on line(p, d) \ {p}.
    # So, for a fixed p and d, the affected x's are the points on line(p, d) \ {p}.
    # Let Affected(p) = union over d of { x | x in line(p, d) \ {p} }
    # For each such x, we check if not_in_N_count[d, x] == 0.
    # If yes, we check if cover_count[x] > 1.
    
    # Optimization:
    # Precompute for each p, the list of (x, d) pairs where p is on line(x, d) \ {x}.
    # This is equivalent to: for each d, for each x in line(p, d) \ {p}, add (x, d) to list.
    
    # However, storing this list for all p might be large.
    # n_points = 2401.
    # Each p is on q-1 lines for each of n_dirs directions? No.
    # For a fixed d, p is on exactly 1 line (the one containing p).
    # The points on that line are line(p, d).
    # So for a fixed d, p affects all x in line(p, d) \ {p}.
    # Total affected (x, d) pairs for a given p is n_dirs * (q - 1) = 50 * 48 = 2400.
    # Total storage: 2401 * 2400 * 2 bytes (x, d) ~ 11 MB. This is acceptable.
    
    # Let's precompute `affects[p]` as a list of (x, d) tuples.
    # Or better, two arrays: `aff_x[p]`, `aff_d[p]`.
    
    aff_x = np.zeros((n_points, n_dirs * (q - 1)), dtype=np.int32)
    aff_d = np.zeros((n_points, n_dirs * (q - 1)), dtype=np.int32)
    
    # Fill affects
    # For each p:
    #   For each d:
    #     Get line(p, d). Remove p.
    #     The remaining points are x.
    #     Store (x, d) for each x.
    
    for p in range(n_points):
        idx = 0
        for d in range(n_dirs):
            # Get points on line(p, d)
            pts = lines[d, p]
            # pts does not contain p? 
            # Wait, lines[d, p] was computed as x + t*v for t=1..q-1.
            # So lines[d, p] contains the OTHER points on the line through p.
            # So for a fixed p and d, the x's that are affected by removing p are exactly lines[d, p].
            # Because if p is on line(x, d), then x is on line(p, d).
            # And x != p.
            # So Affected(p) for direction d is lines[d, p].
            
            aff_x[p, idx:idx + (q - 1)] = pts
            aff_d[p, idx:idx + (q - 1)] = d
            idx += (q - 1)

    # Now iterate through points in N and try to remove them.
    # Order: Sort by "safety" or just iterate.
    # Let's iterate multiple passes if time permits, or just one pass in a good order.
    # Good order: Points that are on lines with high not_in_N_count (less critical).
    # Or simply: try to remove all points that are currently removable.
    
    # To be safe and fast, we'll do a single pass first, then maybe more.
    # We need to update not_in_N_count and cover_count dynamically.
    
    # Let's create a list of candidate points to remove.
    # We'll iterate through all points in N.
    
    # To speed up the check "is p removable":
    # We need to check all (x, d) in affects[p].
    # For each (x, d), if not_in_N_count[d, x] == 0, then we must have cover_count[x] > 1.
    
    # We can vectorize this check for a single p.
    
    # Let's do a few passes.
    # Pass 1: Remove points that are obviously safe.
    
    # We'll use a queue or list of points to try.
    # Start with all points in N.
    
    # To avoid O(N^2) complexity, we can just iterate.
    # N is ~2000. 2000 * 2400 checks = 5 million checks.
    # Each check involves array lookups.
    # 5 million lookups in Python is slow (~1-2s).
    # But we can vectorize the check for a single p.
    
    # Let's try to remove points in a single pass, sorted by some heuristic.
    # Heuristic: Points with higher degree of redundancy.
    # For now, just iterate.
    
    # We'll maintain a list of indices to try.
    # Since removing a point changes the state, we must re-evaluate.
    # A simple greedy: 
    # While possible:
    #   Find a point p in N that is removable.
    #   Remove it.
    #   Update counts.
    
    # Finding a removable point is O(N * K) where K is size of affects[p].
    # If we scan all N points, it's O(N^2 * K).
    # 2000^2 * 2400 = 10^10. Too slow.
    
    # Better:
    # Maintain a set of "critical" points x where cover_count[x] == 1.
    # Let CriticalX be the set of x with cover_count[x] == 1.
    # For each x in CriticalX, let d_x be the unique direction covering it.
    # Let Line_x be the set of points on line(x, d_x) \ {x}.
    # A point p is NOT removable if p is in Line_x for any x in CriticalX.
    # So, Forbidden(p) = Union of Line_x for x in CriticalX.
    # If p is not in Forbidden(p), it is removable?
    # Wait.
    # If p is in Line_x, removing p increases not_in_N_count[d_x, x] from 0 to 1.
    # This makes cover_count[x] drop from 1 to 0. Invalid.
    # So yes, p is removable iff p is NOT in Forbidden(p).
    
    # We can maintain Forbidden set efficiently?
    # When we remove a point p, not_in_N_count changes for all x in Affected(p).
    # This may change cover_count[x].
    # If cover_count[x] becomes 1, x becomes Critical.
    # If cover_count[x] becomes 0, we have a problem (but we prevent this).
    # If cover_count[x] was 1 and becomes 2, x leaves Critical.
    
    # So we need to update CriticalX dynamically.
    
    # Let's implement this.
    
    # Data structures:
    # cover_count: array
    # not_in_N_count: array
    # in_N: boolean array
    # critical_x: set of x with cover_count[x] == 1
    # unique_dir[x]: for x in critical_x, the direction d such that not_in_N_count[d, x] == 0
    # forbidden: set of points p that are on the line of any critical x.
    #   forbidden = Union over x in critical_x of { y | y in line(x, unique_dir[x]) \ {x} }
    
    # Initialization:
    critical_x = set()
    unique_dir = {}
    forbidden = set()
    
    for x in range(n_points):
        if cover_count[x] == 1:
            critical_x.add(x)
            # Find the direction
            d = np.where(not_in_N_count[:, x] == 0)[0][0]
            unique_dir[x] = d
            # Add points on line(x, d) \ {x} to forbidden
            # These are lines[d, x]
            for y in lines[d, x]:
                forbidden.add(y)
                
    # Now, iterate through points in N.
    # We want to remove points that are NOT in forbidden.
    # But removing a point changes forbidden.
    
    # Let's pick a point p in N \ forbidden.
    # Remove it.
    # Update not_in_N_count and cover_count for all (x, d) in Affected(p).
    # Update critical_x, unique_dir, forbidden.
    
    # To make this efficient, we should pick p wisely.
    # Maybe pick p with many "alternative" coverings?
    # Or just pick the first one found.
    
    # Let's do a loop.
    # To avoid infinite loops or getting stuck, we'll limit iterations.
    # Max iterations: number of points in N.
    
    # We'll maintain a list of candidate points.
    # Initially, all p in N such that p not in forbidden.
    
    # Since forbidden changes, we need to re-check.
    # Simple approach:
    # While True:
    #   Find a p in N such that p not in forbidden.
    #   If none, break.
    #   Remove p.
    #   Update state.
    
    # Finding p:
    # We can just scan N.
    # To speed up, we can keep a list of "safe" points and update it.
    
    # Let's just scan.
    # N size ~2000.
    # Each scan is O(N).
    # Total removals ~1000.
    # Total work ~ 1000 * 2000 = 2 million.
    # Plus update cost.
    # Update cost for removing p:
    #   Affected(p) has size 2400.
    #   For each (x, d), update not_in_N_count[d, x].
    #   Check if cover_count[x] changes.
    #   If it does, update critical_x and forbidden.
    #   Updating forbidden:
    #     If x becomes critical, add lines[d, x] to forbidden.
    #     If x stops being critical, remove lines[d, x] from forbidden?
    #     This is hard to do with a set if multiple x contribute to the same point in forbidden.
    #     We need a counter: forbidden_count[y] = number of critical x such that y in line(x, d_x).
    
    forbidden_count = np.zeros(n_points, dtype=np.int32)
    
    # Re-initialize forbidden_count
    for x in critical_x:
        d = unique_dir[x]
        for y in lines[d, x]:
            forbidden_count[y] += 1
            
    # Now, p is removable iff forbidden_count[p] == 0.
    
    # Let's run the loop.
    
    removed = 0
    max_iter = n_points * 2
    for _ in range(max_iter):
        if time.time() > deadline - 30:
            break
            
        # Find a removable point
        p_found = -1
        # Scan N
        # We can optimize by keeping a list of candidates, but let's try simple scan first.
        # To make it faster, we can check random points or use a heap.
        # Let's just scan.
        
        # Optimization: Only scan points in N.
        # We can maintain a list of active points in N.
        
        # For now, simple scan.
        # To avoid O(N^2) if N is large, we can break early.
        
        # Let's try to find any p in N with forbidden_count[p] == 0.
        # We can vectorize this check!
        
        # in_N is boolean array.
        # forbidden_count is int array.
        # candidates = np.where(in_N & (forbidden_count == 0))[0]
        
        candidates = np.where(in_N & (forbidden_count == 0))[0]
        
        if len(candidates) == 0:
            break
            
        # Pick one. Maybe the one with highest "redundancy"?
        # For now, pick the first.
        p = candidates[0]
        
        # Remove p
        in_N[p] = False
        removed += 1
        
        # Update state
        # For each (x, d) in Affected(p):
        #   not_in_N_count[d, x] += 1
        #   If not_in_N_count[d, x] was 0, cover_count[x] -= 1
        #   Else, cover_count[x] unchanged
        
        # We need to iterate through Affected(p).
        # aff_x[p] and aff_d[p] contain the pairs.
        
        # To vectorize the update of not_in_N_count:
        # not_in_N_count[aff_d[p], aff_x[p]] += 1
        # This works!
        
        old_counts = not_in_N_count[aff_d[p], aff_x[p]]
        not_in_N_count[aff_d[p], aff_x[p]] += 1
        
        # Identify which (x, d) had old_count == 0
        # These are the ones where cover_count[x] decreased.
        mask = old_counts == 0
        
        if np.any(mask):
            # Get the x indices where cover_count decreased
            # Note: multiple d might affect the same x?
            # No, for a fixed p, each x appears at most once in Affected(p)?
            # Affected(p) is union over d of lines[d, p].
            # Can x be in lines[d1, p] and lines[d2, p]?
            # Yes, if x=p? No, x != p.
            # If x is on line(p, d1) and line(p, d2), then line(p, d1) = line(p, d2) = line(p, x).
            # This implies d1 and d2 are parallel.
            # But our directions are distinct slopes.
            # So each x appears in at most one direction's line for a fixed p.
            # Wait, is this true?
            # Line(p, d) is the unique line through p with direction d.
            # If x is on it, then the line through x and p has direction d.
            # Since directions are unique, x can be on at most one such line.
            # So x is unique in Affected(p).
            
            x_decreased = aff_x[p][mask]
            d_decreased = aff_d[p][mask]
            
            # Update cover_count
            # cover_count[x] -= 1 for each x in x_decreased
            # We can use np.add.at for safety, but since x are unique, simple indexing works.
            # However, if x appears multiple times in x_decreased (which shouldn't happen), it would be wrong.
            # Let's assume uniqueness.
            
            # We need to update critical_x and forbidden_count.
            # For each x in x_decreased:
            #   old_cover = cover_count[x]
            #   new_cover = old_cover - 1
            #   cover_count[x] = new_cover
            #   
            #   If old_cover == 1 and new_cover == 0: INVALID (should not happen if we checked forbidden_count)
            #   If old_cover == 1 and new_cover == 1: x remains critical? No, cover_count became 1 from 2?
            #      Wait, mask was old_count == 0.
            #      So not_in_N_count[d, x] was 0.
            #      This means d was a covering direction.
            #      So cover_count[x] included d.
            #      So cover_count[x] decreases by 1.
            
            #   Cases:
            #   1. old_cover == 1: x was critical. Now cover_count[x] = 0.
            #      This means x is no longer covered. INVALID.
            #      But we only removed p if forbidden_count[p] == 0.
            #      forbidden_count[p] == 0 implies p is NOT on any critical line.
            #      If x was critical, p must be on line(x, d_x).
            #      So p would be in forbidden.
            #      So this case should not happen.
            
            #   2. old_cover > 1: x is not critical (or was critical? No, critical means cover_count==1).
            #      If old_cover > 1, x was not critical.
            #      new_cover = old_cover - 1.
            #      If new_cover == 1, x BECOMES critical.
            #      If new_cover > 1, x remains non-critical.
            
            # So we only need to handle the case where x BECOMES critical (new_cover == 1).
            # And we don't need to handle x LEAVING critical because that would require cover_count to increase,
            # which doesn't happen when removing a point (counts only increase).
            # Wait, removing a point from N increases not_in_N_count, which decreases cover_count.
            # So cover_count only decreases.
            # So x can only BECOME critical or LOSE coverage.
            # It cannot LEAVE critical (cover_count 1 -> 2).
            
            # So:
            # For each x in x_decreased:
            #   new_cover = cover_count[x] - 1
            #   cover_count[x] = new_cover
            #   If new_cover == 1:
            #       x becomes critical.
            #       Find unique_dir[x].
            #       Add lines[unique_dir[x], x] to forbidden_count.
            
            # Let's process this.
            
            # We need to find unique_dir for new critical points.
            # unique_dir[x] is the d such that not_in_N_count[d, x] == 0.
            # Since cover_count[x] == 1, there is exactly one such d.
            
            # We can vectorize the update of cover_count.
            cover_count[x_decreased] -= 1
            
            # Find new critical points
            new_critical_mask = cover_count[x_decreased] == 1
            if np.any(new_critical_mask):
                new_critical_x = x_decreased[new_critical_mask]
                
                # For each new critical x, find unique direction
                for x in new_critical_x:
                    # Find d such that not_in_N_count[d, x] == 0
                    # We know there is exactly one.
                    d = np.where(not_in_N_count[:, x] == 0)[0][0]
                    unique_dir[x] = d
                    # Add points on line(x, d) \ {x} to forbidden_count
                    # These are lines[d, x]
                    forbidden_count[lines[d, x]] += 1
                    
                    # Add to critical_x set (if needed for logic, but we use forbidden_count)
                    critical_x.add(x)
                    
    # Build result
    points = []
    for i in range(n_points):
        if in_N[i]:
            x = i % q
            y = i // q
            xa, xb = x % p, x // p
            ya, yb = y % p, y // p
            points.append([[xa, xb], [ya, yb]])

    if not points:
        # Fallback
        points = []
        for x in range(q):
            for y in range(q):
                xa, xb = x % p, x // p
                ya, yb = y % p, y // p
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
