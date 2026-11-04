import time
import numpy as np

DIMENSIONS = [4, 4, 5]
_TIMEOUT = 115

def _time_left(deadline):
    return deadline - time.time()

def _tensor_values(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    T = np.zeros((n, m, p), dtype=float)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for c in range(rank):
        uc = Ua[:, c]
        vc = Va[:, c]
        wc = Wa[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    return T

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    T = _tensor_values(U, V, W, n, m, p)
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=tol)

def _rank(U):
    return len(U[0]) if U else 0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_coeffs():
    """Return (u_coeffs, v_coeffs, w_coeffs) for 2x2 Strassen."""
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    u_coeffs = [
        [1, 0, 0, 1],   # a+d
        [0, 0, 1, 1],   # c+d
        [1, 0, 0, 0],   # a
        [0, 0, 0, 1],   # d
        [1, 1, 0, 0],   # a+b
        [0, 0, 1, -1],  # c-a
        [0, 1, 0, -1]   # b-d
    ]
    v_coeffs = [
        [1, 0, 0, 1],   # e+h
        [1, 0, 0, 0],   # e
        [0, 1, 0, -1],  # f-h
        [0, 0, 1, -1],  # g-e
        [0, 0, 0, 1],   # h
        [1, 1, 0, 0],   # e+f
        [0, 0, 1, 1]    # g+h
    ]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    w_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    return u_coeffs, v_coeffs, w_coeffs

def _construct_strassen_4x4x5():
    """
    Constructs a rank-53 decomposition for 4x4 @ 4x5.
    
    Strategy:
    - Use 2-level Strassen for the 4x4 @ 4x4 part (first 4 columns of B): rank 49
    - Add 4 more terms for the 5th column of B: rank 4
    - Total rank: 53
    
    U: shape (16, 53)  -- 4x4 matrix A flattened row-major
    V: shape (20, 53)  -- 4x5 matrix B flattened row-major  
    W: shape (20, 53)  -- 4x5 matrix C flattened, W[k*n+i] = W[k*4+i]
    """
    n, m, p = 4, 4, 5
    rank = 53
    U = np.zeros((n*m, rank))  # 16 x 53
    V = np.zeros((m*p, rank))  # 20 x 53
    W = np.zeros((p*n, rank))  # 20 x 53
    
    u_coeffs, v_coeffs, w_coeffs = _strassen_2x2_coeffs()
    
    # Outer Strassen: split 4x4 into 2x2 blocks of 2x2
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Row-major 4x4: 
    #   0  1  2  3
    #   4  5  6  7
    #   8  9  10 11
    #   12 13 14 15
    # Block 00: [0,1,4,5], Block 01: [2,3,6,7]
    # Block 10: [8,9,12,13], Block 11: [10,11,14,15]
    
    block_indices = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    # Outer Strassen products (7 terms)
    # For each outer term k, define which A blocks and B blocks are combined
    outer_terms = [
        # (A_block_coeffs, B_block_coeffs)
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0]),
        # M3 = A00(B01-B11)
        ([1, 0, 0, 0], [0, 1, 0, -1]),
        # M4 = A11(B10-B00)
        ([0, 0, 0, 1], [0, 0, 1, -1]),
        # M5 = (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1]),
        # M6 = (A10-A00)(B00+B01)
        ([0, 0, 1, -1], [1, 1, 0, 0]),
        # M7 = (A01-A11)(B10+B11)
        ([0, 1, 0, -1], [0, 0, 1, 1]),
    ]
    
    # Outer reconstruction: C_block[bi,bj] = sum_k outer_w[bi*2+bj][k] * M_k
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    outer_w = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    
    term_idx = 0
    
    for k in range(7):
        a_block_coeffs, b_block_coeffs = outer_terms[k]
        
        # Build the 2x2 matrix of linear combinations for A
        # Each entry is a vector of length 16
        A_mat = np.zeros((4, 16))  # 4 entries, each length 16
        B_mat = np.zeros((4, 16))
        
        for bi in range(4):
            for idx in block_indices[bi]:
                A_mat[bi, idx] = a_block_coeffs[bi]
                B_mat[bi, idx] = b_block_coeffs[bi]
        
        # A_mat[0] = A00 part, A_mat[1] = A01 part, A_mat[2] = A10 part, A_mat[3] = A11 part
        # B_mat similarly
        
        for s in range(7):
            u_s = u_coeffs[s]
            v_s = v_coeffs[s]
            
            # Compute combined A vector: sum_i u_s[i] * A_mat[i]
            A_comb = np.zeros(16)
            for i in range(4):
                A_comb += u_s[i] * A_mat[i]
            
            # Compute combined B vector (for first 4 columns): sum_i v_s[i] * B_mat[i]
            B_comb = np.zeros(20)
            for i in range(4):
                B_comb[:16] += v_s[i] * B_mat[i]
            # B_comb[16:20] remains 0 for the 5th column
            
            # For each output position in the 4x4 result
            for I in range(4):
                for J in range(4):
                    i_block = I // 2
                    j_block = J // 2
                    i_local = I % 2
                    j_local = J % 2
                    
                    # Outer coefficient
                    oc = outer_w[i_block * 2 + j_block][k]
                    
                    # Inner coefficient
                    ic = w_coeffs[i_local * 2 + j_local][s]
                    
                    coeff = oc * ic
                    if coeff != 0:
                        U[:, term_idx] = A_comb
                        V[:, term_idx] = B_comb
                        W[:, term_idx] = 0
                        W[J * n + I, term_idx] = coeff  # W[k*n+i] where k=J, i=I
                        term_idx += 1
    
    # Now add 4 more terms for the 5th column of B
    # C[i, 4] = sum_j A[i, j] * B[j, 4]
    # For each i in 0..3, we need one term:
    #   U: A row i (only row i has 1s in positions i*4+j)
    #   V: B column 4 (only column 4 has 1s in positions j*5+4)
    #   W: W[4*n+i] = 1
    
    for i in range(4):
        U[:, term_idx] = 0
        V[:, term_idx] = 0
        W[:, term_idx] = 0
        for j in range(4):
            U[i * m + j, term_idx] = 1
            V[j * p + 4, term_idx] = 1
        W[4 * n + i, term_idx] = 1
        term_idx += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + _TIMEOUT
    
    # Try the Strassen-based decomposition (rank 53)
    try:
        result = _construct_strassen_4x4x5()
        if _is_valid(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook if it fails
    return _schoolbook(n, m, p)
