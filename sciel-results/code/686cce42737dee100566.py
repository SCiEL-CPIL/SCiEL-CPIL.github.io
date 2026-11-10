# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    alpha_sq = 3

    # Precompute field operations
    def elem_add(e1, e2):
        a1, b1 = divmod(e1, p)
        a2, b2 = divmod(e2, p)
        return ((a1 + a2) % p) * p + ((b1 + b2) % p)

    def elem_mul(e1, e2):
        a1, b1 = divmod(e1, p)
        a2, b2 = divmod(e2, p)
        ra = (a1 * a2 + b1 * b2 * alpha_sq) % p
        rb = (a1 * b2 + a2 * b1) % p
        return ra * p + rb

    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        for j in range(q):
            add_table[i, j] = elem_add(i, j)
            mul_table[i, j] = elem_mul(i, j)

    # Directions: 0..48 slopes, 49 vertical
    # Precompute line_other[idx, d, t]
    line_other = np.zeros((q * q, 50, 48), dtype=np.int32)
    for x in range(q):
        for y in range(q):
            idx = x * q + y
            for d in range(49):
                for t_idx, t in enumerate(range(1, q)):
                    a_x, b_x = divmod(x, p)
                    a_t, b_t = divmod(t, p)
                    ox = ((a_x + a_t) % p) * p + ((b_x + b_t) % p)
                    td = mul_table[t, d]
                    a_td, b_td = divmod(td, p)
                    a_y, b_y = divmod(y, p)
                    oy = ((a_y + a_td) % p) * p + ((b_y + b_td) % p)
                    line_other[idx, d, t_idx] = ox * q + oy
            for t_idx, t in enumerate(range(1, q)):
                a_t, b_t = divmod(t, p)
                a_y, b_y = divmod(y, p)
                oy = ((a_y + a_t) % p) * p + ((b_y + b_t) % p)
                line_other[idx, 49, t_idx] = x * q + oy

    # Construct S as the FULL Unital
    # S = { (x,y) : 2*x_a + y_a^2 - 3*y_b^2 == 0 }
    S = np.zeros(q * q, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            x = xa * p + xb
            for ya in range(p):
                for yb in range(p):
                    y = ya * p + yb
                    if (2 * xa + ya * ya - yb * yb * alpha_sq) % p == 0:
                        S[x * q + y] = True

    N = ~S

    # Validity check and repair (Vectorized)
    def check_and_repair(N, S, line_other, deadline):
        # N and S are boolean arrays of shape (2401,)
        # line_other is (2401, 50, 48)
        
        while True:
            if time.time() > deadline:
                break
            
            # Vectorized check:
            # N[line_other] has shape (2401, 50, 48)
            # valid_lines[i, d] is True if all 48 other points on line i,d are in N
            valid_lines = np.all(N[line_other], axis=2)
            valid_counts = np.sum(valid_lines, axis=1)
            
            unsatisfied = np.where(valid_counts == 0)[0]
            if len(unsatisfied) == 0:
                break
            
            # Pick an unsatisfied point
            idx = unsatisfied[0]
            
            # Find a point in S that is on a line through idx
            # We can flatten line_other[idx] to get all 2400 neighbors
            neighbors = line_other[idx].flatten()
            
            # Find neighbors that are in S
            # S is boolean, so S[neighbors] gives boolean mask
            in_S = S[neighbors]
            
            if not np.any(in_S):
                # No points in S on any line through idx?
                # This implies all neighbors are in N.
                # But then valid_counts[idx] should be > 0.
                # Contradiction. So this case shouldn't happen if logic is correct.
                # However, if idx itself is in S?
                # The condition is about OTHER points.
                # If idx is in S, it doesn't affect the "others in N" check directly,
                # but idx being in S means it's not in N.
                # The validity check for idx depends on neighbors being in N.
                # If all neighbors are in N, idx is satisfied regardless of idx's status.
                # So if unsatisfied, there MUST be a neighbor in S.
                break
            
            # Pick the first such neighbor
            # Get indices of True in in_S
            candidate_indices = np.where(in_S)[0]
            if len(candidate_indices) > 0:
                # Pick one, maybe the one that appears most frequently in S?
                # For now, just pick the first one.
                p_idx = neighbors[candidate_indices[0]]
                
                # Remove p_idx from S, add to N
                S[p_idx] = False
                N[p_idx] = True
            else:
                break

        return N, S

    start_time = time.time()
    deadline = start_time + 590 # Safety margin

    N, S = check_and_repair(N, S, line_other, deadline)

    points = []
    for x in range(q):
        for y in range(q):
            if N[x * q + y]:
                xa, xb = divmod(x, p)
                ya, yb = divmod(y, p)
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
