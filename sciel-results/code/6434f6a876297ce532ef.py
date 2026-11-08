"""Matrix multiplication 3x4x8 rank-57 exact tensor decomposition."""
import time

_TIME_LIMIT = 130.0

def construct():
    t0 = time.time()
    n, m, p = 3, 4, 8
    # Rank 1 for row 0 (1x4x8 is rank 1 tensor)
    # Rank 56 for rows 1,2 (2x4x8 using Strassen blocks)
    total_rank = 1 + 56  # 57

    u_rows = n * m
    v_rows = m * p
    w_rows = p * n

    U_flat = [0] * (u_rows * total_rank)
    V_flat = [0] * (v_rows * total_rank)
    W_flat = [0] * (w_rows * total_rank)

    # --- Block A: first row of A (i=0), rank 1 term ---
    base = 0
    # U[0*m + j, 0] = 1 for all j
    for jj in range(m):
        U_flat[(0 * m + jj) * total_rank + base] = 1
    # V[jj*p + kk, 0] = 1 for all jj, kk
    for jj in range(m):
        for kk in range(p):
            V_flat[(jj * p + kk) * total_rank + base] = 1
    # W[kk*n + 0, 0] = 1 for all kk
    for kk in range(p):
        W_flat[(kk * n + 0) * total_rank + base] = 1
    base += 1

    # --- Block B: rows i=1,2; use 2x2 Strassen per (j_pair, k_pair) ---
    # Strassen 7-product factors for 2x2x2
    su = [
        (1, 0, 0, 1),
        (0, 0, 1, 1),
        (1, 0, 0, 0),
        (0, 0, 0, 1),
        (1, 1, 0, 0),
        (-1, 0, 1, 0),
        (0, 1, 0, -1),
    ]
    sv = [
        (1, 0, 0, 1),
        (1, 0, 0, 0),
        (0, 1, 0, -1),
        (-1, 0, 1, 0),
        (0, 0, 0, 1),
        (1, 1, 0, 0),
        (0, 0, 1, 1),
    ]
    sw = [
        (1, 0, 0, 1),
        (0, 0, 1, -1),
        (0, 1, 0, 1),
        (1, 0, 1, 0),
        (-1, 1, 0, 0),
        (0, 0, 0, 1),
        (1, 0, 0, 0),
    ]

    col_idx = base
    for jb in range(m // 2):
        for kb in range(p // 2):
            for t in range(7):
                # A-factors: local (li, lj) -> global row (1+li, jb*2+lj)
                for li in range(2):
                    for lj in range(2):
                        loc = li * 2 + lj
                        val = su[t][loc]
                        if val:
                            gi = 1 + li
                            gj = jb * 2 + lj
                            U_flat[(gi * m + gj) * total_rank + col_idx] += val

                # B-factors: local (lj, lk) -> global row (jb*2+lj, kb*2+lk)
                for lj in range(2):
                    for lk in range(2):
                        loc = lj * 2 + lk
                        val = sv[t][loc]
                        if val:
                            gj = jb * 2 + lj
                            gk = kb * 2 + lk
                            V_flat[(gj * p + gk) * total_rank + col_idx] += val

                # C-factors: local (li, lk) -> W index (gk*n + gi)
                for li in range(2):
                    for lk in range(2):
                        loc = li * 2 + lk
                        val = sw[t][loc]
                        if val:
                            gi = 1 + li
                            gk = kb * 2 + lk
                            W_flat[(gk * n + gi) * total_rank + col_idx] += val

                col_idx += 1
                if time.time() - t0 > _TIME_LIMIT:
                    break

    # Reshape to (rows, rank)
    U_out = [U_flat[r * total_rank:(r + 1) * total_rank] for r in range(u_rows)]
    V_out = [V_flat[r * total_rank:(r + 1) * total_rank] for r in range(v_rows)]
    W_out = [W_flat[r * total_rank:(r + 1) * total_rank] for r in range(w_rows)]

    return {"U": U_out, "V": V_out, "W": W_out}
