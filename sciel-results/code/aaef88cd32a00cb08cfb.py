# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 570.0  # seconds; safety margin before the 600s kill

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3 in F_49 = F_7[alpha]

    # ---- Build baseline Nikodym set (Hermitian unital, "skip step 2.5") ----
    # A point is EXCLUDED iff it is in the unital U and its y-coordinate is NOT in Y.
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    in_N = np.ones((p, p, p, p), dtype=bool)  # indexed [xa, xb, ya, yb]
    for xa in range(p):
        for ya in range(p):
            for yb in range(p):
                # in_unital: 2*xa + ya^2 - yb^2*k == 0 mod p
                if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                    if (ya, yb) not in y_special:
                        in_N[xa, :, ya, yb] = False

    # ---- Precompute directions and line masks ----
    # Directions: (1, m) for m in F_49, and (0, 1)
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append((1, ma, 1, mb))
    dirs.append((0, 0, 1, 0))  # (0,1)
    n_dirs = len(dirs)

    # line_mask[dir_idx, xa, xb, ya, yb]: 1 if point is on the line through origin
    # with direction dirs[dir_idx].
    line_mask = np.zeros((n_dirs, p, p, p, p), dtype=bool)
    for di, (vx_a, vx_b, vy_a, vy_b) in enumerate(dirs):
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        # Solve t*vx = x, t*vy = y (consistent if on line)
                        if vx_a == 0 and vx_b == 0:
                            # direction (0,1): line is {(0, y) for y}
                            if xa == 0 and xb == 0:
                                line_mask[di, xa, xb, ya, yb] = True
                        else:
                            # t = x / vx
                            # Check consistency: t * vy == y
                            # (x_a, x_b) = t * (vx_a, vx_b)
                            # t = x / vx
                            # If vx is a unit (nonzero), t is uniquely determined.
                            # t_a + t_b*alpha = (xa + xb*alpha) / (vx_a + vx_b*alpha)
                            # = (xa + xb*alpha) * (vx_a - vx_b*alpha) / (vx_a^2 - vx_b^2*k)
                            # Denominator: vx_a^2 - vx_b^2*k mod p
                            den = (vx_a * vx_a - vx_b * vx_b * k) % p
                            if den != 0:
                                den_inv = pow(den, -1, p)
                                # numerator: (xa + xb*alpha)*(vx_a - vx_b*alpha)
                                # = (xa*vx_a - xb*vx_b*k) + (xb*vx_a - xa*vx_b)*alpha
                                num_a = (xa * vx_a - xb * vx_b * k) % p
                                num_b = (xb * vx_a - xa * vx_b) % p
                                t_a = (num_a * den_inv) % p
                                t_b = (num_b * den_inv) % p
                                # Check t * vy == y
                                # (t_a + t_b*alpha) * (vy_a + vy_b*alpha)
                                # = (t_a*vy_a - t_b*vy_b*k) + (t_a*vy_b + t_b*vy_a)*alpha
                                chk_a = (t_a * vy_a - t_b * vy_b * k) % p
                                chk_b = (t_a * vy_b + t_b * vy_a) % p
                                if chk_a == ya % p and chk_b == yb % p:
                                    line_mask[di, xa, xb, ya, yb] = True
                            else:
                                # vx is non-zero but denominator 0 -> vx is a zero divisor?
                                # In F_q, nonzero elements are units, so den != 0 for nonzero vx.
                                # This case shouldn't happen, but handle gracefully.
                                pass

    # ---- Precompute for each point, for each direction, the set of other points on the line ----
    # For point x and direction v, the line is {x + t*v : t in F_49}.
    # Other points are x + t*v for t != 0.
    # We need: all other points on the line are in N.
    # Equivalently: for all t != 0, x + t*v is in N.
    # We can precompute for each (x, v): the count of t != 0 such that x + t*v is NOT in N.
    # If count == 0, then x is "covered" by direction v.
    # x is critical if for ALL directions v, count > 0.

    # Precompute line members for each (point, direction): list of other points.
    # Actually, we need: for each (x, v), the set of points {x + t*v : t != 0}.
    # We can precompute this as a boolean mask or as a list.

    # For efficiency, precompute for each (x, v): the 48 other points (49-1=48).
    # point_index[xa, xb, ya, yb] -> flat index
    # We'll store for each (x_flat, v_idx): the list of flat indices of other points on the line.

    # Let's precompute the "other points" for each (x, v) as a flat index array.
    # Total: 2401 * 50 = 120050 entries, each of length 48. That's ~5.7M integers.
    # Store as a flat array of shape (2401, 50, 48) with flat indices.

    # Actually, let's think about this differently. For each (x, v), we need to check
    # if all 48 other points are in N. We can precompute the "bad count" for each (x, v)
    # and update it as we remove points.

    # Let's precompute the other points for each (x, v).
    # flat_index[xa, xb, ya, yb] = xa*p^3 + xb*p^2 + ya*p + yb
    def flat_idx(xa, xb, ya, yb):
        return xa * p * p * p + xb * p * p + ya * p + yb

    def unflat(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return xa, xb, ya, yb

    # Precompute other_points[x_flat, v_idx] = array of 48 flat indices
    other_points = np.zeros((p * p * p * p, n_dirs, p - 1), dtype=np.int32)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_flat = flat_idx(xa, xb, ya, yb)
                    for di, (vx_a, vx_b, vy_a, vy_b) in enumerate(dirs):
                        # Generate x + t*v for t != 0
                        # t = (ta, tb)
                        cnt = 0
                        for ta in range(p):
                            for tb in range(p):
                                if ta == 0 and tb == 0:
                                    continue
                                # x + t*v
                                # (xa, xb) + (ta, tb)*(vx_a, vx_b)
                                # = (xa, xb) + (ta*vx_a - tb*vx_b*k, ta*vx_b + tb*vx_a)
                                nx_a = (xa + ta * vx_a - tb * vx_b * k) % p
                                nx_b = (xb + ta * vx_b + tb * vx_a) % p
                                ny_a = (ya + ta * vy_a - tb * vy_b * k) % p
                                ny_b = (yb + ta * vy_b + tb * vy_a) % p
                                other_points[x_flat, di, cnt] = flat_idx(nx_a, nx_b, ny_a, ny_b)
                                cnt += 1
                        assert cnt == p - 1

    # ---- Initialize bad_count[x_flat, v_idx] = number of other points NOT in N ----
    in_N_flat = in_N.reshape(-1)
    bad_count = np.zeros((p * p * p * p, n_dirs), dtype=np.int32)
    for x_flat in range(p * p * p * p):
        if in_N_flat[x_flat]:
            for di in range(n_dirs):
                # Count other points not in N
                oc = other_points[x_flat, di]
                bad_count[x_flat, di] = np.sum(~in_N_flat[oc])

    # A point is critical if for all directions, bad_count > 0.
    # i.e., min over directions of bad_count > 0.
    # Actually, we need: for ALL directions, there exists at least one other point not in N.
    # So critical iff min_d bad_count[x, d] > 0.
    # Wait, that's not right. We need: for all d, bad_count[x, d] > 0.
    # That means the point cannot be removed because removing it would break coverage
    # for some direction... no wait.
    
    # Let me reconsider. A point x is in N. We want to remove x from N.
    # After removing x, x is no longer in N. For x to still be "covered" (i.e., there
    # exists a direction v such that all other points on the line through x with direction v
    # are in N), we need: there exists v such that all 48 other points are in N.
    # That is: exists v such that bad_count[x, v] == 0.
    # So x is critical (cannot be removed) iff for ALL v, bad_count[x, v] > 0.
    # i.e., min_v bad_count[x, v] > 0.
    
    # Wait, that's the same thing. min > 0 iff all > 0.
    # So critical iff min_d bad_count[x, d] > 0.
    
    # Let's compute the minimum bad count per point.
    min_bad = np.min(bad_count, axis=1)
    
    # Points that are NOT critical (can be removed): min_bad == 0
    # We want to remove as many as possible.
    
    # Greedy removal: repeatedly find a non-critical point, remove it, update bad_counts.
    # When we remove point x, for each other point y on a line with x, bad_count[y, v]
    # decreases by 1 (since x is no longer in N, so it's now "not in N" for y's perspective).
    # Wait, no. bad_count[y, v] counts other points on the line through y with direction v
    # that are NOT in N. When we remove x from N, x becomes "not in N". So for any y
    # such that x is on the line through y with direction v, bad_count[y, v] increases by 1.
    
    # Hmm, that means removing a point makes other points MORE likely to be critical,
    # which seems counterproductive. Let me reconsider.
    
    # Actually wait. Let me re-examine. The Nikodym condition is:
    # For every point x in F_q^2 (whether in N or not), there exists a direction v
    # such that all OTHER points on the line through x with direction v are in N.
    
    # So for a point x NOT in N, we still need it to be covered.
    # For a point x IN N, we need it to be covered.
    
    # When we remove x from N:
    # 1. x itself: was covered before (since it was in N). Now x is not in N.
    #    We need x to still be covered: exists v such that all other points on line
    #    through x with direction v are in N. This is the condition min_bad[x] == 0
    #    BEFORE removal.
    # 2. Other points y: for y to be covered, we need exists v such that all other
    #    points on line through y with direction v are in N. Removing x from N means
    #    x is now "not in N". If x was on the line through y with direction v, then
    #    the condition for y with direction v is now harder to satisfy (one more point
    #    is not in N). So removing x can only make coverage HARDER for other points.
    
    # This means: once we remove a point, it can only make it harder to remove others.
    # So greedy removal should work: remove points that are currently non-critical,
    # and each removal only makes the remaining points harder to remove.
    
    # But wait, that means the order matters, and we should remove points that are
    # "easiest" to remove first? Or does it not matter?
    
    # Actually, the key insight is: removing a point x from N means x is no longer
    # available to help cover other points. So we want to remove points that are
    # "least useful" for covering others. A point x is "useful" for covering y
    # if x is on a line through y with direction v where v is the ONLY direction
    # that covers y (or one of few directions).
    
    # This is getting complex. Let me just do a simple greedy: repeatedly find any
    # non-critical point, remove it, update, repeat until no non-critical points remain.
    
    # For efficiency, I'll use a priority queue or just scan.
    
    # Let's track which points are currently in N.
    in_N_now = in_N_flat.copy()
    
    # We'll maintain bad_count and update it as we remove points.
    # When we remove point x (set in_N_now[x] = False):
    # For each direction v, for each other point y on the line through x with direction v
    # (i.e., y = x + t*v for t != 0, or x = y + t'*v for some t' != 0):
    #   bad_count[y, v] += 1  (since x is now not in N)
    
    # But we need the reverse: for each y and v, which x's are on the line through y
    # with direction v? That's the same as other_points[y, v].
    # So when we remove x, for each y such that x in other_points[y, v], bad_count[y, v] += 1.
    
    # This is expensive to update naively. Let me think of a better way.
    
    # Alternative: instead of maintaining bad_count incrementally, just recompute
    # min_bad periodically or use a different strategy.
    
    # Given the small size (2401 points), let's just do a simple greedy with
    # periodic recomputation.
    
    # Actually, let me reconsider the whole approach. The key observation from the
    # knowledge is that greedy pruning works. Let me just implement it carefully.
    
    # Strategy: 
    # 1. Start with baseline N.
    # 2. Compute for each point x in N: is it critical? (min_bad[x] > 0)
    # 3. Remove all non-critical points simultaneously? No, that might not work
    #    because removing one might make another critical.
    # 4. Better: iteratively remove one non-critical point at a time, updating.
    
    # For the update: when we remove x, we need to update bad_count for all y, v
    # such that x is on the line through y with direction v.
    # x is on the line through y with direction v iff y is on the line through x
    # with direction v (since lines are symmetric... wait, not exactly).
    # Actually, the line through y with direction v is {y + t*v : t in F_q}.
    # x is on this line iff x = y + t*v for some t, i.e., y = x + (-t)*v.
    # So x is on the line through y with direction v iff y is on the line through x
    # with direction v (with t' = -t).
    # So the set of (y, v) pairs where x is on the line through y with direction v
    # is the same as the set of (y, v) pairs where y is on the line through x with
    # direction v, i.e., y in other_points[x, v].
    
    # So when we remove x: for each v, for each y in other_points[x, v]:
    #   bad_count[y, v] += 1
    
    # This is 49 * 48 = 2352 updates per removal. With up to 2401 removals,
    # that's ~5.6M updates. Should be fast enough.
    
    # Let's implement this.
    
    # Precompute: for each x, the list of (y, v) pairs where x is on the line
    # through y with direction v. This is: for each v, y in other_points[x, v].
    # We already have other_points[x, v] as flat indices.
    
    # Let's just do the greedy removal.
    
    # For efficiency, I'll use a simple loop and check time periodically.
    
    # Find all non-critical points initially.
    # critical[x] = (min_bad[x] > 0) and in_N_now[x]
    # We want to remove non-critical points.
    
    # Let's use a worklist approach.
    
    # Actually, let me just do a simple iterative approach:
    # Repeat:
    #   Find all x in N such that min_bad[x] == 0 (non-critical).
    #   If none, stop.
    #   Remove all of them simultaneously? Or one at a time?
    #   If we remove them all at once, we need to be careful: removing x might make
    #   y critical, but if y was also non-critical before, it's fine to remove it too.
    #   Actually, removing multiple points at once is safe if we check the condition
    #   based on the state BEFORE the removals. But that might not be optimal.
    
    # Let me just remove one at a time, choosing the one with the "least impact"
    # (e.g., the one that is on the fewest lines for other points).
    
    # Actually, for simplicity and correctness, let me just remove one at a time,
    # always picking the first non-critical point we find.
    
    # Given the small size, this should be fine.
    
    removed = np.zeros(p * p * p * p, dtype=bool)
    
    while time.time() < _DEADLINE - 30:
        # Find a non-critical point in N
        found = -1
        for x in range(p * p * p * p):
            if in_N_now[x] and not removed[x]:
                if np.min(bad_count[x]) == 0:
                    found = x
                    break
        if found == -1:
            break
        
        # Remove found
        x = found
        in_N_now[x] = False
        removed[x] = True
        
        # Update bad_count for all y, v where x is on the line through y with direction v
        for v in range(n_dirs):
            for y in other_points[x, v]:
                bad_count[y, v] += 1
    
    # Build the result
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_flat = flat_idx(xa, xb, ya, yb)
                    if in_N_now[x_flat]:
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
