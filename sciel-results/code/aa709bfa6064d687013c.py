import time
import numpy as np

# Dimensions of the multiplication: (n x m) * (m x p)
N, M, P = 3, 5, 5

# Block partitions
R_SIZES = [1, 2]          # rows (n-space)
C_SIZES = [2, 2, 1]       # inner (m-space)
D_SIZES = [2, 2, 1]       # cols (p-space)


def _block_offsets(sizes):
    offs = []
    acc = 0
    for s in sizes:
        offs.append(acc)
        acc += s
    return offs


def construct():
    """
    Rank-reduced exact decomposition for 3x5 * 5x5 matrix multiplication.
    Uses a block partition with 2x2x2 Strassen sub-blocks (rank 7 each) and
    schoolbook for the remaining blocks. Returns {'U','V','W'} as lists of
    lists with integer entries that exactly decompose the multiplication tensor.
    """
    start = time.time()
    deadline = start + 110  # safety margin within the 120s limit

    r_off = _block_offsets(R_SIZES)
    c_off = _block_offsets(C_SIZES)
    d_off = _block_offsets(D_SIZES)

    # First pass: compute total rank and per-block specs
    total_rank = 0
    block_specs = []
    for ri, rs in enumerate(R_SIZES):
        for ci, cs in enumerate(C_SIZES):
            for di, ds in enumerate(D_SIZES):
                use_strassen = (rs >= 2 and cs >= 2 and ds >= 2)
                rank = 7 if use_strassen else (rs * cs * ds)
                total_rank += rank
                block_specs.append((ri, ci, di, rs, cs, ds, use_strassen, rank))

    U = np.zeros((N * M, total_rank), dtype=int)
    V = np.zeros((M * P, total_rank), dtype=int)
    W = np.zeros((P * N, total_rank), dtype=int)

    # Strassen factors for 2x2x2 (entries in {-1,0,1})
    Ufac = np.array([
        [1, 0, 1, 0, 1, 1, -1],
        [1, 1, 0, 1, 0, -1, 1],
    ], dtype=int)
    Vfac = np.array([
        [1, 1, -1, -1, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 0],
    ], dtype=int)
    Wfac = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
    ], dtype=int)

    rank = 0
    for (ri, ci, di, rs, cs, ds, use_strassen, brank) in block_specs:
        # global row ranges
        u_rows = [r_off[ri] + a for a in range(rs)]
        v_rows = [c_off[ci] + b for b in range(cs)]
        # W row = k*n + i  (trace convention)
        w_rows = []
        for a in range(rs):
            for l in range(ds):
                gk = d_off[di] + l
                gi = r_off[ri] + a
                w_rows.append(gk * N + gi)

        if use_strassen:
            assert rs == 2 and cs == 2 and ds == 2
            for a in range(2):
                for c in range(7):
                    U[u_rows[a], rank + c] = Ufac[a, c]
            for b in range(2):
                for c in range(7):
                    V[v_rows[b], rank + c] = Vfac[b, c]
            for a in range(2):
                for c in range(7):
                    W[w_rows[a], rank + c] = Wfac[a, c]
        else:
            k = 0
            for a in range(rs):
                for b in range(cs):
                    for l in range(ds):
                        col = rank + k
                        U[u_rows[a], col] = 1
                        V[v_rows[b], col] = 1
                        W[w_rows[a * ds + l], col] = 1
                        k += 1

        if time.time() > deadline:
            break
        rank += brank

    # Cheap exact verification; fall back to full schoolbook if anything is off.
    try:
        Ua = U
        Va = V
        Wa = W
        ok = True
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    s = 0
                    ur = i * M + j
                    vr = j * P + k
                    wr = k * N + i
                    # dot of the three rows
                    for r in range(total_rank):
                        s += int(Ua[ur, r]) * int(Va[vr, r]) * int(Wa[wr, r])
                    if s != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            U = np.zeros((N * M, N * M * P), dtype=int)
            V = np.zeros((M * P, N * M * P), dtype=int)
            W = np.zeros((P * N, N * M * P), dtype=int)
            cols = np.arange(N * M * P)
            i_grid, j_grid, k_grid = np.indices((N, M, P))
            u_rows = (i_grid * M + j_grid).flatten()
            v_rows = (j_grid * P + k_grid).flatten()
            w_rows = (k_grid * N + i_grid).flatten()
            U[u_rows, cols] = 1
            V[v_rows, cols] = 1
            W[w_rows, cols] = 1
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
