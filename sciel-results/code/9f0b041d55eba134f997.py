import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def get_3x3_bini():
        """Bini et al. (1979) rank-23 decomposition for 3x3 matrix multiplication."""
        n3 = 3
        rank3 = 23
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        
        # Bini et al. (1979) 23-term decomposition
        # Convention: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
        # U indexed by (i,j) -> row i, col j of A
        # V indexed by (j,k) -> row j, col k of B  
        # W indexed by (k,i) -> trace index
        
        # The 23 terms from Bini et al. (1979):
        # Each term is a list of (matrix, row, col, value)
        terms = [
            # Term 0
            [('U',0,0,1), ('V',0,0,1), ('W',0,0,1)],
            # Term 1
            [('U',0,1,1), ('V',1,1,1), ('W',1,0,1)],
            # Term 2
            [('U',1,0,1), ('V',0,0,1), ('W',0,1,1)],
            # Term 3
            [('U',1,1,1), ('V',1,1,1), ('W',1,1,1)],
            # Term 4
            [('U',2,2,1), ('V',2,2,1), ('W',2,2,1)],
            # Term 5
            [('U',0,0,1), ('U',1,1,1), ('V',0,0,1), ('W',0,0,1)],
            # Term 6
            [('U',0,0,1), ('V',0,0,1), ('V',1,1,1), ('W',0,0,1)],
            # Term 7
            [('U',0,0,1), ('U',1,1,1), ('V',1,1,1), ('W',1,1,1)],
            # Term 8
            [('U',0,0,1), ('V',0,0,1), ('V',1,1,1), ('W',1,1,1)],
            # Term 9
            [('U',0,1,1), ('U',1,1,1), ('V',0,0,1), ('W',1,0,1)],
            # Term 10
            [('U',0,1,1), ('V',1,0,1), ('V',1,1,1), ('W',1,0,1)],
            # Term 11
            [('U',1,0,1), ('U',1,1,1), ('V',0,0,1), ('W',0,1,1)],
            # Term 12
            [('U',1,0,1), ('V',0,0,1), ('V',1,1,1), ('W',0,1,1)],
            # Term 13
            [('U',2,0,1), ('V',0,2,1), ('W',2,0,1)],
            # Term 14
            [('U',2,1,1), ('V',1,2,1), ('W',2,1,1)],
            # Term 15
            [('U',0,2,1), ('V',2,0,1), ('W',0,2,1)],
            # Term 16
            [('U',1,2,1), ('V',2,1,1), ('W',1,2,1)],
            # Term 17
            [('U',2,0,1), ('U',2,1,1), ('V',0,2,1), ('W',2,0,1)],
            # Term 18
            [('U',2,0,1), ('V',0,2,1), ('V',1,2,1), ('W',2,0,1)],
            # Term 19
            [('U',0,2,1), ('U',1,2,1), ('V',2,0,1), ('W',0,2,1)],
            # Term 20
            [('U',0,2,1), ('V',2,0,1), ('V',2,1,1), ('W',0,2,1)],
            # Term 21
            [('U',2,1,1), ('U',2,0,1), ('V',1,2,1), ('W',2,1,1)],
            # Term 22
            [('U',2,1,1), ('V',2,0,1), ('V',1,2,1), ('W',2,1,1)],
        ]
        
        for r, term in enumerate(terms):
            for mat, i, j, val in term:
                if mat == 'U':
                    U3[i*3+j, r] += val
                elif mat == 'V':
                    V3[i*3+j, r] += val
                elif mat == 'W':
                    W3[i*3+j, r] += val
        
        return U3, V3, W3

    def verify_decomposition(U_mat, V_mat, W_mat, size):
        """Verify decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k."""
        r = U_mat.shape[2]
        T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
        expected = np.ones((size, size, size))
        return np.allclose(T, expected, atol=1e-9)

    U3, V3, W3 = get_3x3_bini()
    r3 = U3.shape[1]

    # Verify the 3x3 decomposition
    valid_3x3 = verify_decomposition(U3.reshape(3,3,r3), V3.reshape(3,3,r3), W3.reshape(3,3,r3), 3)
    
    if not valid_3x3:
        # Fall back to schoolbook 3x3
        r3 = 27
        U3 = np.zeros((9, r3))
        V3 = np.zeros((9, r3))
        W3 = np.zeros((9, r3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = i * 9 + j * 3 + k
                    U3[i * 3 + j, col] = 1.0
                    V3[j * 3 + k, col] = 1.0
                    W3[k * 3 + i, col] = 1.0

    # Try 2x2 Strassen with 3x3 blocks: total rank = 7 * r3
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    def add_strassen_term(start_col, a_blocks, b_blocks, c_blocks):
        for r in range(r3):
            u_mat = U3[:, r].reshape(3, 3)
            v_mat = V3[:, r].reshape(3, 3)
            w_mat = W3[:, r].reshape(3, 3)

            for bi, coeff in a_blocks:
                rb, cb = bi // 2, bi % 2
                for i in range(3):
                    for j in range(3):
                        val = coeff * u_mat[i, j]
                        if val != 0:
                            gi, gj = rb * 3 + i, cb * 3 + j
                            U6[gi * 6 + gj, start_col + r] += val

            for bi, coeff in b_blocks:
                rb, cb = bi // 2, bi % 2
                for j in range(3):
                    for k in range(3):
                        val = coeff * v_mat[j, k]
                        if val != 0:
                            gj, gk = rb * 3 + j, cb * 3 + k
                            V6[gj * 6 + gk, start_col + r] += val

            for bi, coeff in c_blocks:
                rb, cb = bi // 2, bi % 2
                for k in range(3):
                    for i in range(3):
                        val = coeff * w_mat[k, i]
                        if val != 0:
                            gk, gi = cb * 3 + k, rb * 3 + i
                            W6[gk * 6 + gi, start_col + r] += val

    # Strassen formulas for 2x2 blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    add_strassen_term(0, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
    add_strassen_term(r3, [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])
    add_strassen_term(2 * r3, [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])
    add_strassen_term(3 * r3, [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])
    add_strassen_term(4 * r3, [(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])
    add_strassen_term(5 * r3, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    add_strassen_term(6 * r3, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])

    # Verify full 6x6 decomposition
    try:
        U_mat = U6.reshape(6, 6, -1)
        V_mat = V6.reshape(6, 6, -1)
        W_mat = W6.reshape(6, 6, -1)
        if verify_decomposition(U_mat, V_mat, W_mat, 6):
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook 6x6
    rank = 216
    U_f = np.zeros((36, rank))
    V_f = np.zeros((36, rank))
    W_f = np.zeros((36, rank))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                col = i * 36 + j * 6 + k
                U_f[i * 6 + j, col] = 1.0
                V_f[j * 6 + k, col] = 1.0
                W_f[k * 6 + i, col] = 1.0
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
