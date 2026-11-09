import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin

    p = 7
    q = p * p
    alpha_sq = 3
    
    def elem_to_idx(a, b):
        return a * p + b

    def idx_to_elem(idx):
        return (idx // p, idx % p)

    # Field tables
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    elems = [idx_to_elem(i) for i in range(q)]
    
    for i in range(q):
        a_i, b_i = elems[i]
        for j in range(q):
            a_j, b_j = elems[j]
            add_table[i, j] = elem_to_idx((a_i + a_j) % p, (b_i + b_j) % p)
            prod_a = (a_i * a_j + b_i * b_j * alpha_sq) % p
            prod_b = (a_i * b_j + b_i * a_j) % p
            mul_table[i, j] = elem_to_idx(prod_a, prod_b)

    total_points = q * q
    num_dirs = q + 1
    
    # Precompute lines
    # lines[x, d] is array of other points on line through x with direction d
    # Shape: (total_points, num_dirs, q-1)
    one_idx = elem_to_idx(1, 0)
    zero_idx = elem_to_idx(0, 0)
    
    # Direction vectors
    # d in 0..q-1: v = (1, d)
    # d = q: v = (0, 1)
    
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    
    x_indices = np.arange(total_points) // q
    y_indices = np.arange(total_points) % q
    
    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx
            
        # t in 1..q-1
        t_vals = np.arange(1, q)
        txs = np.array([mul_table[t, vx_idx] for t in t_vals])
        tys = np.array([mul_table[t, vy_idx] for t in t_vals])
        
        # For each point x, compute x + t*v
        # x_indices[:, None] + txs[None, :] using add_table
        nx = add_table[x_indices[:, None], txs[None, :]]
        ny = add_table[y_indices[:, None], tys[None, :]]
        line_points[:, d_idx, :] = nx * q + ny

    in_N = np.ones(total_points, dtype=bool)
    
    # Helper to compute witnesses
    def compute_witnesses():
        # W[x] is bitmask or list of directions
        # We just need count and if count==1, which one
        # count[x]
        # unique_dir[x]
        counts = np.zeros(total_points, dtype=np.int32)
        unique_dirs = np.full(total_points, -1, dtype=np.int32)
        
        # For each x, check each d
        # If all points in line_points[x, d] are in N, then d is a witness
        for x in range(total_points):
            if not in_N[x]:
                continue
            c = 0
            ud = -1
            for d in range(num_dirs):
                pts = line_points[x, d, :]
                if np.all(in_N[pts]):
                    c += 1
                    if c > 1:
                        ud = -2 # more than 1
                        break
                    else:
                        ud = d
            counts[x] = c
            unique_dirs[x] = ud
        return counts, unique_dirs

    # Main loop
    while time.time() < deadline:
        counts, unique_dirs = compute_witnesses()
        
        # Find critical points: count == 1
        critical_mask = (counts == 1) & in_N
        critical_points = np.where(critical_mask)[0]
        
        if len(critical_points) == 0:
            # No critical points? Then we can remove any point?
            # If no point has exactly 1 witness, all have 0 or >=2.
            # If 0, they are unsafe. But we start with all safe.
            # If we remove points, some might become 0.
            # If counts[x] == 0, x is unsafe. This shouldn't happen if we only remove safely.
            # If all counts >= 2, we can remove any point?
            # Removing y might reduce count of x by 1.
            # If x had 2, it becomes 1.
            # So we can remove any point that is not required to keep others at >= 1?
            # Actually, if no point has count 1, we can remove a point y.
            # Which points lose a witness? Those x where y is on a witness line.
            # If x had count 2, it becomes 1. That's fine, it's still safe.
            # So we can remove ANY point if no point has count 1?
            # Let's remove one.
            candidates = np.where(in_N)[0]
            if len(candidates) > 0:
                # Pick one
                y = candidates[0]
                in_N[y] = False
                continue
            else:
                break

        # Compute blocked set B
        # B = union of lines through critical points
        blocked = np.zeros(total_points, dtype=bool)
        for x in critical_points:
            d = unique_dirs[x]
            pts = line_points[x, d, :]
            blocked[pts] = True
            # Also block x itself?
            # If x is critical, can we remove x?
            # If we remove x, x is gone.
            # Does removing x break others?
            # x is on witness lines of others?
            # Yes.
            # But the condition for removing y is: y is not on the unique witness line of any critical point.
            # Is x on the unique witness line of any critical point?
            # Maybe.
            # Let's just block the points on the lines.
            # What about x itself?
            # If we remove x, do we break any critical point z?
            # z breaks if x is on L_dz(z).
            # So if x is in B, we cannot remove x.
            # If x is not in B, we can remove x?
            # But x is critical.
            # If we remove x, x is no longer in N.
            # The condition for x is no longer needed.
            # But did x's removal break others?
            # If x was on the unique witness line of some z, then x would be in B (because z is critical).
            # So if x is not in B, removing x is safe.
            # So x itself is removable if not in B.
            # But we are computing B from lines.
            # Let's add x to blocked? No.
            # If x is critical, it relies on L_d(x).
            # Other points might rely on x.
            # If z is critical and x is on L_dz(z), then x is in B.
            # So if x is not in B, no critical point relies on x.
            # So x is safe to remove.
            pass
        
        # Removable points: in N and not in blocked
        removable_mask = in_N & (~blocked)
        
        # Also, can we remove critical points that are not blocked?
        # Yes, as reasoned above.
        # So removable_mask is correct.
        
        if not np.any(removable_mask):
            # No removable points?
            # This means every point in N is on the unique witness line of some critical point.
            # This is a deadlock for batch removal.
            # Try single point removal with careful check?
            # Or break.
            # Let's try to find a point that minimizes damage.
            # Actually, if we are stuck, we might need to break a cycle.
            # But for now, let's just break to avoid infinite loop.
            # Or try random restart?
            # Let's try removing one point from the critical set that is "least" blocking?
            # No, just break.
            break
        
        # Remove all removable points
        in_N[removable_mask] = False

    # Final verification and construction
    # Check validity
    counts, _ = compute_witnesses()
    if np.any(counts[in_N] == 0):
        # Invalid, fallback to all points
        in_N = np.ones(total_points, dtype=bool)
        
    points = []
    for p_idx in range(total_points):
        if in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
