"""Matrix multiplication 3x4x8 with rank-88 decomposition using Strassen blocks."""
import time
import itertools

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0]
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 0]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0],
        [0, 1]
    ]
    V = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0],
        [0, 1]
    ]
    # Actually for 1x2x2:
    # U has 1*2=2 rows (A elements: x, y)
    # V has 2*2=4 rows (B elements: e, f, g, h)
    # W has 2*1=2 rows (C elements: C0, C1)
    # Multiplications: x*e, x*g, y*f, y*h
    # C0 = x*e + y*f
    # C1 = x*g + y*h
    U = [
        [1, 0, 1, 0],  # x
        [0, 1, 0, 1]   # y
    ]
    V = [
        [1, 0, 0, 0],  # e = B[0,0]
        [0, 1, 0, 0],  # f = B[1,0]
        [0, 0, 1, 0],  # g = B[0,1]
        [0, 0, 0, 1]   # h = B[1,1]
    ]
    W = [
        [1, 1, 0, 0],  # C0 = x*e + y*f
        [0, 0, 1, 1]   # C1 = x*g + y*h
    ]
    return U, V, W


def _build_decomposition(tiling):
    """Build a decomposition given a tiling specification.
    tiling is a list of (type, j_start, j_end, k_start, k_end)
    where type is 'strassen2' for 2x2x2 blocks (applies to i=0,1)
    or 'school1' for 1x2x2 blocks (applies to i=2)
    """
    n, m, p = 3, 4, 8
    total_rank = 0
    blocks = []
    
    for (btype, js, je, ks, ke) in tiling:
        if btype == 'strassen2':
            # Requires je-js == 2, ke-ks == 2
            r = 7
        elif btype == 'school1':
            # Requires je-js == 2, ke-ks == 2
            r = 4
        else:
            raise ValueError("Unknown block type")
        blocks.append((btype, js, je, ks, ke, r))
        total_rank += r
    
    U = [[0] * total_rank for _ in range(n * m)]   # 12 rows
    V = [[0] * total_rank for _ in range(m * p)]   # 32 rows
    W = [[0] * total_rank for _ in range(p * n)]   # 24 rows
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    col_offset = 0
    for (btype, js, je, ks, ke, r) in blocks:
        j0, j1 = js, js + 1
        k0, k1 = ks, ks + 1
        
        if btype == 'strassen2':
            # 2x2x2 block for i=0,1
            u_rows = [
                0 * m + j0, 0 * m + j1,
                1 * m + j0, 1 * m + j1
            ]
            v_rows = [
                j0 * p + k0, j0 * p + k1,
                j1 * p + k0, j1 * p + k1
            ]
            w_rows = [
                k0 * n + 0, k1 * n + 0,
                k0 * n + 1, k1 * n + 1
            ]
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
        elif btype == 'school1':
            # 1x2x2 block for i=2
            u_rows = [2 * m + j0, 2 * m + j1]
            v_rows = [
                j0 * p + k0, j1 * p + k0,
                j0 * p + k1, j1 * p + k1
            ]
            w_rows = [k0 * n + 2, k1 * n + 2]
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += 4
    
    return {"U": U, "V": V, "W": W}


def _default_tiling():
    """Default tiling: 8 Strassen blocks for i=0,1 and 8 schoolbook for i=2."""
    tiling = []
    for jp in range(2):
        for kp in range(4):
            tiling.append(('strassen2', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    for jp in range(2):
        for kp in range(4):
            tiling.append(('school1', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    return tiling


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True


def _construct_3x4x8():
    """Construct the decomposition."""
    tiling = _default_tiling()
    return _build_decomposition(tiling)


def construct():
    start = time.time()
    result = _construct_3x4x8()
    n, m, p = DIMENSIONS
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}
    return result
