"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _get_strassen_2x2_decomposition():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2 = (c+d)e
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1
    
    # M3 = a(f-h)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1
    
    # M4 = d(g-f)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1
    
    # M5 = (a+b)h
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    """
    rank = U.shape[1]
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _construct_4x4x5_strassen():
    """
    Construct 4x4x5 decomposition using Strassen for 4x4x4 part (rank 49)
    and schoolbook for 4x4x1 part (rank 4). Total rank 53.
    """
    n, m, p = 4, 4, 5
    U_2, V_2, W_2 = _get_strassen_2x2_decomposition()
    
    # For 4x4x4 Strassen via Kronecker of 2x2x2 Strassen
    # The 4x4 matrices are partitioned into 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # Similarly for B and C
    
    # The Kronecker construction:
    # For each of the 7 Strassen terms at the block level,
    # and each of the 7 Strassen terms at the element level,
    # we get a combined term.
    
    # Block-level Strassen steps:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # M3 = A00(B01-B11) -> contributes to C01, C11
    # M4 = A11(B10-B00) -> contributes to C00, C10
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    
    # Element-level Strassen steps (for 2x2x2):
    # Same 7 terms as in _get_strassen_2x2_decomposition
    
    # The combined decomposition has 7*7 = 49 terms.
    
    # For each block-level step s (0-6) and element-level step t (0-6):
    # The term involves:
    # - A elements from the blocks specified by step s, with coefficients from element step t
    # - B elements from the blocks specified by step s, with coefficients from element step t
    # - C elements from the blocks specified by step s, with coefficients from element step t
    
    # Block-level step definitions:
    # Each step: (A_blocks, B_blocks, C_blocks) where each is list of (block_idx, sign)
    block_steps = [
        { 'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)] },  # M1
        { 'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)] },  # M2
        { 'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)] },  # M3
        { 'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)] },  # M4
        { 'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)] },  # M5
        { 'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)] },  # M6
        { 'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)] },  # M7
    ]
    
    # Map block index and inner index to global 4x4 index
    def map_block_inner(b_idx, i_idx):
        # b_idx: 0:(0:2,0:2), 1:(0:2,2:4), 2:(2:4,0:2), 3:(2:4,2:4)
        # i_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) within the 2x2 block
        r_base = 0 if b_idx < 2 else 2
        c_base = 0 if b_idx % 2 == 0 else 2
        
        r_off = 0 if i_idx < 2 else 1
        c_off = 0 if i_idx % 2 == 0 else 1
        
        return r_base + r_off, c_base + c_off
    
    rank_4x4 = 49
    U_4x4 = np.zeros((16, rank_4x4))
    V_4x4 = np.zeros((16, rank_4x4))
    W_4x4 = np.zeros((16, rank_4x4))
    
    col = 0
    for s_idx, step in enumerate(block_steps):
        for t_idx in range(7):
            u_inner = U_2[:, t_idx]
            v_inner = V_2[:, t_idx]
            w_inner = W_2[:, t_idx]
            
            # Accumulate U
            for b_idx, sign in step['A']:
                for i_idx in range(4):
                    val = u_inner[i_idx]
                    if val != 0:
                        r, c = map_block_inner(b_idx, i_idx)
                        U_4x4[r*4+c, col] += val * sign
                        
            # Accumulate V
            for b_idx, sign in step['B']:
                for i_idx in range(4):
                    val = v_inner[i_idx]
                    if val != 0:
                        r, c = map_block_inner(b_idx, i_idx)
                        V_4x4[r*4+c, col] += val * sign
                        
            # Accumulate W
            for b_idx, sign in step['C']:
                for i_idx in range(4):
                    val = w_inner[i_idx]
                    if val != 0:
                        r, c = map_block_inner(b_idx, i_idx)
                        W_4x4[r*4+c, col] += val * sign
                        
            col += 1
    
    # Now extend to 4x4x5 by adding 4 terms for the 5th column
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # Term j: U[i*4+j, j] = 1, V[j*5+4, j] = 1, W[4*4+i, j] = 1 for all i
    
    U_extra = np.zeros((16, 4))
    V_extra = np.zeros((20, 4))
    W_extra = np.zeros((20, 4))
    
    for j in range(4):
        for i in range(4):
            U_extra[i*4+j, j] = 1
        V_extra[j*5+4, j] = 1
        for i in range(4):
            W_extra[4*4+i, j] = 1
    
    U = np.hstack([U_4x4, U_extra])
    V = np.hstack([V_4x4, V_extra])
    W = np.hstack([W_4x4, W_extra])
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try Strassen + Schoolbook (Rank 53)
    try:
        U, V, W = _construct_4x4x5_strassen()
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        else:
            print("Strassen+Schoolbook verification failed.")
    except Exception as e:
        print(f"Error in Strassen+Schoolbook: {e}")
    
    # Fallback to Schoolbook (Rank 80)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
