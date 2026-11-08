"""Matrix multiplication 3x4x8 rank-88 exact tensor decomposition.

Construction:
- Row i=0: full schoolbook 1x4x8 = 32 terms
- Rows i=1,2: 2x4x8, partitioned into 2x2x2 Strassen blocks:
  2 j-pairs x 4 k-pairs x 7 = 56 terms
- Total: 32 + 56 = 88
"""
import time

_TIME_LIMIT = 130.0

def construct():
    t0 = time.time()
    n, m, p = 3, 4, 8
    total_rank = 88

    u_rows = n * m  # 12
    v_rows = m * p  # 32
    w_rows = p * n  # 24

    U_flat = [0] * (u_rows * total_rank)
    V_flat = [0] * (v_rows * total_rank)
    W_flat = [0] * (w_rows * total_rank)

    # --- Block A: first row of A (i=0), full schoolbook 4x8 = 32 terms ---
    base = 0
    for jj in range(m):
        for kk in range(p):
            U_flat[(0 * m + jj) * total_rank + base] = 1
            V_flat[(jj * p + kk) * total_rank + base] = 1
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
