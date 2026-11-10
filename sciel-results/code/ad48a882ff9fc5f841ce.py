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

    # Validity check and repair
    # valid[idx, d] is True if all others on line are in N
    # We want all idx to have at least one valid d.
    
    def check_and_repair(N, S, line_other, deadline):
        # Compute valid counts
        valid_count = np.zeros(q * q, dtype=np.int32)
        valid = np.zeros((q * q, 50), dtype=bool)
        
        # Initial check
        for idx in range(q * q):
            for d in range(50):
                others = line_other[idx, d, :]
                if np.all(N[others]):
                    valid[idx, d] = True
                    valid_count[idx] += 1
        
        # Repair loop
        # Find unsatisfied points (valid_count == 0)
        # Remove them from S (set N to True)
        # Repeat
        
        while True:
            if time.time() > deadline:
                break
            unsatisfied = np.where(valid_count == 0)[0]
            if len(unsatisfied) == 0:
                break
            
            # Pick one unsatisfied point to remove from S
            # Heuristic: pick one that is in S
            # If it's not in S, it's already in N, but still unsatisfied?
            # If idx is in N, and unsatisfied, it means no line through idx is fully in N.
            # Removing from S only helps if idx was in S.
            # If idx is in N, we cannot "remove" it. We must remove OTHER points from S?
            # No, removing points from S adds them to N, which can only HELP validity.
            # Wait. If idx is in N, and unsatisfied, it means for all d, there is some point on line(idx,d) in S.
            # To fix idx, we must remove SOME of those S points from S (add to N).
            # So we should pick a point p in S that lies on a line through an unsatisfied point.
            
            # Simpler: Just iterate through S points and remove one if it helps any unsatisfied point.
            # Or, just remove ALL unsatisfied points from S if they are in S.
            # If an unsatisfied point is in N, we must remove a neighbor from S.
            
            # Let's try removing the first unsatisfied point from S if it is in S.
            # If it is in N, find a point on its lines that is in S and remove it.
            
            fixed = False
            for idx in unsatisfied:
                if S[idx]:
                    # Remove idx from S
                    S[idx] = False
                    N[idx] = True
                    # Update valid counts for all points affected by idx
                    # This is expensive. Let's just recompute valid counts from scratch each iteration?
                    # 2401 * 50 * 48 ops ~ 6M. 6M * iterations.
                    # If iterations < 100, it's 600M. Might be too slow in Python.
                    # But we break early if valid.
                    
                    # Recompute valid
                    valid_count = np.zeros(q * q, dtype=np.int32)
                    valid = np.zeros((q * q, 50), dtype=bool)
                    for i in range(q * q):
                        for d in range(50):
                            others = line_other[i, d, :]
                            if np.all(N[others]):
                                valid[i, d] = True
                                valid_count[i] += 1
                    fixed = True
                    break
                else:
                    # idx is in N but unsatisfied.
                    # Find a point p on any line through idx that is in S.
                    # Remove p from S.
                    for d in range(50):
                        others = line_other[idx, d, :]
                        for p_idx in others:
                            if S[p_idx]:
                                S[p_idx] = False
                                N[p_idx] = True
                                valid_count = np.zeros(q * q, dtype=np.int32)
                                valid = np.zeros((q * q, 50), dtype=bool)
                                for i in range(q * q):
                                    for d2 in range(50):
                                        others2 = line_other[i, d2, :]
                                        if np.all(N[others2]):
                                            valid[i, d2] = True
                                            valid_count[i] += 1
                                fixed = True
                                break
                        if fixed:
                            break
                if fixed:
                    break
            
            if not fixed:
                # No S points to remove? Should not happen if S is not empty.
                break

        return N, S

    start_time = time.time()
    deadline = start_time + 600
    
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
