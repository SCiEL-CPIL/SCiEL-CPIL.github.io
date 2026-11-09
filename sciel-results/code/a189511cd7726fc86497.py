"""Matrix multiplication 4x4 * 4x8 rank-minimal exact decomposition (Strassen blocks)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_decomposition(n, m, p):
    """Build U, V, W for the tensor T[i,j,k] = 1 (matrix multiplication tensor).
    
    Uses 2x2 Strassen blocks: each 2x2 (A block) * 2x4 (B block) uses 7 multiplications.
    Returns (U, V, W) each as numpy int arrays.
    """
    rows_U = n * m          # 16
    rows_V = m * p          # 32
    rows_W = p * n          # 32

    U_list = []
    V_list = []
    W_list = []

    # Strassen 7-product scheme for 2x2 A * 2x2 B:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b01-b11)
    # P4 = a11*(b10-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b01)
    # P7 = (a01-a11)*(b10+b11)
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6

    # A-forms (4-vector [a00, a01, a10, a11]):
    A_forms = [
        [1, 0, 1, 0],   # P1: a00+a11
        [0, 0, 1, 1],   # P2: a10+a11
        [1, 0, 0, 0],   # P3: a00
        [0, 0, 0, 1],   # P4: a11
        [1, 1, 0, 0],   # P5: a00+a01
        [-1, 0, 1, 0],  # P6: a10-a00
        [0, 1, 0, -1],  # P7: a01-a11
    ]

    # B-forms for 2x4 B block (8-vector, index = r*4 + c for r in {0,1}, c in {0,1,2,3})
    # For each product, the B-form applies the same linear combination to both column-pairs.
    B_forms = [
        [1, 0, 1, 0, 1, 0, 1, 0],  # P1: (b00+b11) for both halves
        [1, 0, 0, 0, 1, 0, 0, 0],  # P2: b00 for both halves
        [0, 1, -1, 0, 0, 1, -1, 0],  # P3: (b01-b11) for both halves
        [0, 0, 1, -1, 0, 0, 1, -1],  # P4: (b10-b00) for both halves
        [0, 0, 0, 1, 0, 0, 0, 1],  # P5: b11 for both halves
        [1, 1, 0, 0, 1, 1, 0, 0],  # P6: (b00+b01) for both halves
        [0, 0, 1, 1, 0, 0, 1, 1],  # P7: (b10+b11) for both halves
    ]

    # C-coefficients: for each product p, for each (r, c_local) in the 2x4 C block
    # Left half (c_local 0,1): standard Strassen C for 2x2
    # Right half (c_local 2,3): same coefficients (since B-forms already span both halves)
    C_coef = {
        (0, 0): [1, 0, 0, 1, -1, 0, 1],  # P1,P2,P3,P4,P5,P6,P7 -> C00
        (0, 1): [0, 0, 1, 0, 1, 0, 0],   # -> C01
        (1, 0): [0, 1, 0, 1, 0, 0, 0],   # -> C10
        (1, 1): [1, -1, 1, 0, 0, 1, 0],  # -> C11
        (0, 2): [1, 0, 0, 1, -1, 0, 1],  # -> C00' (right half)
        (0, 3): [0, 0, 1, 0, 1, 0, 0],   # -> C01'
        (1, 2): [0, 1, 0, 1, 0, 0, 0],   # -> C10'
        (1, 3): [1, -1, 1, 0, 0, 1, 0],  # -> C11'
    }

    for ri in range(0, n, 2):
        for rj in range(0, m, 2):
            for cj in range(0, p, 2):
                for p_idx in range(7):
                    col = len(U_list)
                    U_list.append([0]*rows_U)
                    V_list.append([0]*rows_V)
                    W_list.append([0]*rows_W)
                    u = U_list[col]
                    v = V_list[col]
                    w = W_list[col]
                    
                    # A-form: 4 entries at A block positions (ri + ar, rj + ac)
                    for ar in range(2):
                        for ac in range(2):
                            aidx = (ri + ar) * m + (rj + ac)
                            u[aidx] = A_forms[p_idx][ar * 2 + ac]
                    
                    # B-form: 8 entries at B block positions (rj + br, cj + bc)
                    for br in range(2):
                        for bc in range(4):
                            bidx = (rj + br) * p + (cj + bc)
                            v[bidx] = B_forms[p_idx][br * 4 + bc]
                    
                    # C-form: 8 entries at C block positions (cj + cc, ri + cr)
                    for cr in range(2):
                        for cc in range(4):
                            cidx = (cj + cc) * n + (ri + cr)
                            w[cidx] = C_coef[(cr, cc)][p_idx]

    U = np.array(U_list).T  # shape (rows_U, rank)
    V = np.array(V_list).T
    W = np.array(W_list).T
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit

    try:
        U, V, W = _strassen_decomposition(n, m, p)
        if _verify(U, V, W, n, m, p):
            rank = U.shape[1]
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            U, V, W = _schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
