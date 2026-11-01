"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time

_t0 = time.time()
_DEADLINE = _t0 + 140.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15

    # Block-diagonal decomposition:
    #   - 7 disjoint 2x2x2 blocks (column pairs (0,1)..(12,13)), each rank 7 (Strassen)
    #   - 1 residual 2x2x1 block (column 14), rank 4 (trivial)
    # Total rank = 7*7 + 4 = 53
    rank = 53

    # Allocate flat factor matrices as 2-D lists of ints
    U = [[0] * rank for _ in range(n * m)]   # 4 x 53
    V = [[0] * rank for _ in range(m * p)]   # 30 x 53
    W = [[0] * rank for _ in range(p * n)]   # 30 x 53

    # U row indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V row indices: j*p + k  (j = 0 or 1, k = 0..14)
    # W row indices: k*n + i  (k = 0..14, i = 0 or 1)

    # Strassen 2x2x2 factor pattern (local U/V/W coefficient triples per term)
    # P1 = (A00+A11)(B0k0+B1k1)
    # P2 = (A10+A11)B0k0
    # P3 = A00(B0k1-B1k1)
    # P4 = A11(B1k0-B0k0)
    # P5 = (A00+A01)B1k1
    # P6 = (A10-A00)(B0k0+B0k1)
    # P7 = (A01-A11)(B1k0+B1k1)
    #
    # Output reconstruction:
    # C0k0 = P1 + P4 - P5 + P7
    # C1k0 = P2 + P4
    # C0k1 = P3 + P5
    # C1k1 = P1 - P2 + P3 + P6

    # Precompute the coefficient tables for one Strassen block as tuples of (row, coeff)
    u_entries = [
        # term index -> list of (U_row, coeff)
        [(0, 1), (3, 1)],   # P1
        [(2, 1), (3, 1)],   # P2
        [(0, 1)],           # P3
        [(3, 1)],           # P4
        [(0, 1), (1, 1)],   # P5
        [(2, 1), (0, -1)],  # P6
        [(1, 1), (3, -1)],  # P7
    ]
    v_entries = [
        # term index -> list of (V_row_offset_from_k0/k1 pattern, coeff) -- we encode k0/k1 explicitly below
        [(('B0k0', 1)), (('B1k1', 1))],  # P1
        [(('B0k0', 1))],                  # P2
        [(('B0k1', 1)), (('B1k1', -1))],  # P3
        [(('B1k0', 1)), (('B0k0', -1))],  # P4
        [(('B1k1', 1))],                  # P5
        [(('B0k0', 1)), (('B0k1', 1))],   # P6
        [(('B1k0', 1)), (('B1k1', 1))],   # P7
    ]
    w_entries = [
        # term index -> list of (W_row_pattern, coeff)
        [(('C0k0', 1)), (('C1k1', 1))],  # P1
        [(('C1k0', 1)), (('C1k1', -1))], # P2
        [(('C0k1', 1)), (('C1k1', 1))],  # P3
        [(('C0k0', 1)), (('C1k0', 1))],  # P4
        [(('C0k0', -1)), (('C0k1', 1))], # P5
        [(('C1k1', 1))],                 # P6
        [(('C0k0', 1))],                 # P7
    ]

    def v_row(code, k0, k1):
        # code like 'B0k0', 'B1k1', etc.
        j = int(code[1])
        k = k0 if code.endswith('k0') else k1
        return j * p + k

    def w_row(code, k0, k1):
        # code like 'C0k0', 'C1k1', etc.
        i = int(code[1])
        k = k0 if code.endswith('k0') else k1
        return k * n + i

    # Place 7 Strassen blocks
    for blk in range(7):
        k0 = 2 * blk
        k1 = 2 * blk + 1
        base = 7 * blk
        for t in range(7):
            r = base + t
            for (u_row_i, c) in u_entries[t]:
                U[u_row_i][r] += c
            for (code, c) in v_entries[t]:
                V[v_row(code, k0, k1)][r] += c
            for (code, c) in w_entries[t]:
                W[w_row(code, k0, k1)][r] += c

    # Place the residual 2x2x1 block for column 14 (rank 4, trivial)
    b14 = 49
    # Q1 = A00 * B014 -> C014
    U[0][b14 + 0] = 1
    V[0 * p + 14][b14 + 0] = 1
    W[14 * 2 + 0][b14 + 0] = 1
    # Q2 = A01 * B114 -> C014
    U[1][b14 + 1] = 1
    V[1 * p + 14][b14 + 1] = 1
    W[14 * 2 + 0][b14 + 1] = 1
    # Q3 = A10 * B014 -> C114
    U[2][b14 + 2] = 1
    V[0 * p + 14][b14 + 2] = 1
    W[14 * 2 + 1][b14 + 2] = 1
    # Q4 = A11 * B114 -> C114
    U[3][b14 + 3] = 1
    V[1 * p + 14][b14 + 3] = 1
    W[14 * 2 + 1][b14 + 3] = 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
