"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _get_strassen_222():
    """Returns U, V, W for 2x2x2 matrix multiplication using Strassen (Rank 7)."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    
    # M3 = A00(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    
    # M4 = A11(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _get_schoolbook(n, m, p):
    """Returns U, V, W for nxmxp matrix multiplication using Schoolbook."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Total rank: 42
    total_rank = 42
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block, u_rows, v_rows, w_rows):
        nonlocal col_offset
        r = U_block.shape[1]
        # u_rows, v_rows, w_rows are lists of global indices corresponding to rows of U_block, V_block, W_block
        for i_local, i_global in enumerate(u_rows):
            U[i_global, col_offset:col_offset+r] = U_block[i_local]
        for j_local, j_global in enumerate(v_rows):
            V[j_global, col_offset:col_offset+r] = V_block[j_local]
        for k_local, k_global in enumerate(w_rows):
            W[k_global, col_offset:col_offset+r] = W_block[k_local]
        col_offset += r

    # Block 1: A1 B11 (2x2x2 Strassen)
    # A1: i in {0,1}, j in {0,1} -> Global A idx: i*5+j
    # B11: j in {0,1}, k in {0,1} -> Global B idx: j*5+k
    # C: i in {0,1}, k in {0,1} -> Global W idx: k*2+i
    U1, V1, W1 = _get_strassen_222()
    u_rows_1 = [i*5+j for i in range(2) for j in range(2)]
    v_rows_1 = [j*5+k for j in range(2) for k in range(2)]
    w_rows_1 = [k*2+i for k in range(2) for i in range(2)]
    add_block(U1, V1, W1, u_rows_1, v_rows_1, w_rows_1)
    
    # Block 2: A1 B12 (2x2x2 Strassen)
    # A1: same
    # B12: j in {0,1}, k in {2,3} -> Global B idx: j*5+k
    # C: i in {0,1}, k in {2,3} -> Global W idx: k*2+i
    U2, V2, W2 = _get_strassen_222()
    u_rows_2 = [i*5+j for i in range(2) for j in range(2)]
    v_rows_2 = [j*5+k for j in range(2) for k in range(2,4)]
    w_rows_2 = [k*2+i for k in range(2,4) for i in range(2)]
    add_block(U2, V2, W2, u_rows_2, v_rows_2, w_rows_2)
    
    # Block 3: A1 B13 (2x2x1 Schoolbook)
    # A1: same
    # B13: j in {0,1}, k=4 -> Global B idx: j*5+4
    # C: i in {0,1}, k=4 -> Global W idx: 4*2+i
    U3, V3, W3 = _get_schoolbook(2, 2, 1)
    u_rows_3 = [i*5+j for i in range(2) for j in range(2)]
    v_rows_3 = [j*5+4 for j in range(2)]
    w_rows_3 = [4*2+i for i in range(2)]
    add_block(U3, V3, W3, u_rows_3, v_rows_3, w_rows_3)
    
    # Block 4: A21 B21 (2x2x5)
    # A21: i in {0,1}, j in {2,3} -> Global A idx: i*5+j
    # B21: j in {2,3}, k in {0..4} -> Global B idx: j*5+k
    
    # Block 4.1: A21 B21_1 (2x2x2 Strassen)
    # B21_1: j in {2,3}, k in {0,1}
    U41, V41, W41 = _get_strassen_222()
    u_rows_41 = [i*5+j for i in range(2) for j in range(2,4)]
    v_rows_41 = [j*5+k for j in range(2,4) for k in range(2)]
    w_rows_41 = [k*2+i for k in range(2) for i in range(2)]
    add_block(U41, V41, W41, u_rows_41, v_rows_41, w_rows_41)
    
    # Block 4.2: A21 B21_2 (2x2x2 Strassen)
    # B21_2: j in {2,3}, k in {2,3}
    U42, V42, W42 = _get_strassen_222()
    u_rows_42 = [i*5+j for i in range(2) for j in range(2,4)]
    v_rows_42 = [j*5+k for j in range(2,4) for k in range(2,4)]
    w_rows_42 = [k*2+i for k in range(2,4) for i in range(2)]
    add_block(U42, V42, W42, u_rows_42, v_rows_42, w_rows_42)
    
    # Block 4.3: A21 B21_3 (2x2x1 Schoolbook)
    # B21_3: j in {2,3}, k=4
    U43, V43, W43 = _get_schoolbook(2, 2, 1)
    u_rows_43 = [i*5+j for i in range(2) for j in range(2,4)]
    v_rows_43 = [j*5+4 for j in range(2,4)]
    w_rows_43 = [4*2+i for i in range(2)]
    add_block(U43, V43, W43, u_rows_43, v_rows_43, w_rows_43)
    
    # Block 5: A22 B22 (2x1x5 Schoolbook)
    # A22: i in {0,1}, j=4 -> Global A idx: i*5+4
    # B22: j=4, k in {0..4} -> Global B idx: 4*5+k
    # C: i in {0,1}, k in {0..4} -> Global W idx: k*2+i
    U5, V5, W5 = _get_schoolbook(2, 1, 5)
    u_rows_5 = [i*5+4 for i in range(2)]
    v_rows_5 = [4*5+k for k in range(5)]
    w_rows_5 = [k*2+i for k in range(5) for i in range(2)]
    add_block(U5, V5, W5, u_rows_5, v_rows_5, w_rows_5)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
