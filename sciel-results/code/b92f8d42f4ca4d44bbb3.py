import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def _construct_strassen():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11, A22
    U[0, 0] = 1; U[3, 0] = 1
    # V: B11, B22
    V[0, 0] = 1; V[3, 0] = 1
    # W: C11, C22
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C21 += M2, C22 -= M2? 
    # Wait, let's use the standard Strassen formulas for C.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's redefine W contributions carefully.
    # W[k*2+i, r] contributes to C[i, k] (since output index is k*n+i? No, W[k*n+i, r])
    # Output C is 2x2. Index c = k*2 + i.
    # C[0,0] (i=0, k=0) -> W[0, r]
    # C[0,1] (i=0, k=1) -> W[2, r]
    # C[1,0] (i=1, k=0) -> W[1, r]
    # C[1,1] (i=1, k=1) -> W[3, r]
    
    # M1: C11 += 1, C22 += 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2: C21 += 1, C22 -= 1
    W[1, 1] = 1
    W[3, 1] = -1
    
    # M3: C12 += 1, C22 += 1
    W[2, 2] = 1
    W[3, 2] = 1
    
    # M4: C11 += 1, C21 += 1
    W[0, 3] = 1
    W[1, 3] = 1
    
    # M5: C11 -= 1, C12 += 1
    W[0, 4] = -1
    W[2, 4] = 1
    
    # M6: C22 += 1
    W[3, 5] = 1
    
    # M7: C11 += 1
    W[0, 6] = 1
    
    # Now U and V for each M
    # M1: (A11+A22)(B11+B22)
    # A11 is U[0], A22 is U[3]
    U[0, 0] = 1; U[3, 0] = 1
    # B11 is V[0], B22 is V[3]
    V[0, 0] = 1; V[3, 0] = 1

    # M2: (A21+A22)B11
    # A21 is U[2], A22 is U[3]
    U[2, 1] = 1; U[3, 1] = 1
    # B11 is V[0]
    V[0, 1] = 1

    # M3: A11(B12-B22)
    # A11 is U[0]
    U[0, 2] = 1
    # B12 is V[1], B22 is V[3]
    V[1, 2] = 1; V[3, 2] = -1

    # M4: A22(B21-B11)
    # A22 is U[3]
    U[3, 3] = 1
    # B21 is V[2], B11 is V[0]
    V[2, 3] = 1; V[0, 3] = -1

    # M5: (A11+A12)B22
    # A11 is U[0], A12 is U[1]
    U[0, 4] = 1; U[1, 4] = 1
    # B22 is V[3]
    V[3, 4] = 1

    # M6: (A21-A11)(B11+B12)
    # A21 is U[2], A11 is U[0]
    U[2, 5] = 1; U[0, 5] = -1
    # B11 is V[0], B12 is V[1]
    V[0, 5] = 1; V[1, 5] = 1

    # M7: (A12-A22)(B21+B22)
    # A12 is U[1], A22 is U[3]
    U[1, 6] = 1; U[3, 6] = -1
    # B21 is V[2], B22 is V[3]
    V[2, 6] = 1; V[3, 6] = 1

    return U, V, W

def _construct_schoolbook_221():
    """
    Returns U, V, W for 2x2x1 matrix multiplication.
    U shape (4, 4), V shape (2, 4), W shape (2, 4)
    Note: V and W have different shapes in the local block context?
    Local A is 2x2, B is 2x1, C is 2x1.
    U index: i*2+j (4 rows)
    V index: j*1+k? No, B is 2x1. j in 0..1, k in 0..0.
    So V index is j. (2 rows)
    W index: k*2+i? k in 0..0, i in 0..1.
    So W index is i. (2 rows)
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    
    # Term 0: A00 B0 -> C0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # Term 1: A01 B1 -> C0
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # Term 2: A10 B0 -> C1
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # Term 3: A11 B1 -> C1
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _construct_kronecker():
    n, m, p = DIMENSIONS
    # Total rank 36
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_str, V_str, W_str = _construct_strassen()
    U_sch, V_sch, W_sch = _construct_schoolbook_221()
    
    r = 0
    
    # Helper to add a block
    def add_block(a_offset, a_size, v_offset, v_size, w_offset, w_size, U_b, V_b, W_b):
        nonlocal r
        n_b, m_b, p_b = a_size, v_size, w_size
        # U_b shape (n_b * m_b, rank_b)
        # V_b shape (m_b * p_b, rank_b)
        # W_b shape (p_b * n_b, rank_b)
        rank_b = U_b.shape[1]
        
        for i in range(n_b):
            for j in range(m_b):
                local_u_idx = i * m_b + j
                global_u_idx = a_offset + i * m + j
                U[global_u_idx, r:r+rank_b] = U_b[local_u_idx, :]
                
        for j in range(m_b):
            for k in range(p_b):
                local_v_idx = j * p_b + k
                global_v_idx = v_offset + j * p + k
                V[global_v_idx, r:r+rank_b] = V_b[local_v_idx, :]
                
        for k in range(p_b):
            for i in range(n_b):
                local_w_idx = k * n_b + i
                global_w_idx = w_offset + k * n + i
                W[global_w_idx, r:r+rank_b] = W_b[local_w_idx, :]
                
        r += rank_b

    # Block 1: A[0:2, 0:2], B[0:2, 0:2]
    # a_offset=0, a_size=2 (n=2, m_block=2)
    # v_offset=0, v_size=2 (m_block=2, p_block=2)
    # w_offset=0, w_size=2 (p_block=2, n=2)
    add_block(0, 2, 0, 2, 0, 2, U_str, V_str, W_str)
    
    # Block 2: A[0:2, 0:2], B[0:2, 2:4]
    add_block(0, 2, 2, 2, 2, 2, U_str, V_str, W_str)
    
    # Block 3: A[0:2, 0:2], B[0:2, 4:5]
    # Schoolbook 2x2x1
    add_block(0, 2, 4, 1, 4, 1, U_sch, V_sch, W_sch)
    
    # Block 4: A[0:2, 2:4], B[2:4, 0:2]
    # a_offset=2 (j starts at 2)
    # v_offset=10 (j starts at 2, k starts at 0 -> 2*5+0=10)
    # w_offset=0
    add_block(2, 2, 10, 2, 0, 2, U_str, V_str, W_str)
    
    # Block 5: A[0:2, 2:4], B[2:4, 2:4]
    # v_offset=12 (j=2, k=2 -> 2*5+2=12)
    # w_offset=2
    add_block(2, 2, 12, 2, 2, 2, U_str, V_str, W_str)
    
    # Block 6: A[0:2, 2:4], B[2:4, 4:5]
    # v_offset=14 (j=2, k=4 -> 2*5+4=14)
    # w_offset=4
    add_block(2, 2, 14, 1, 4, 1, U_sch, V_sch, W_sch)
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U, V, W = _construct_kronecker()
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
