"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Strategy:
    - Decompose the 2x14 matrix B into 7 blocks of 2x2 matrices.
    - Apply Strassen's algorithm (rank 7) to each block independently.
    - Since the A matrix is shared, we sum the contributions.
    - Total rank = 7 blocks * 7 Strassen terms = 49.
    
    This is the theoretical minimum for this structure given the known rank of T_2,2,2 is 7.
    """
    _t0 = time.time()
    _DEADLINE = 120.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28
    
    num_blocks = p // 2  # 7
    
    # Strassen terms for 2x2 multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define coefficients for each term
    # U coeffs: A00, A01, A10, A11
    # V coeffs: B00, B01, B10, B11 (local to block)
    # W coeffs: C00, C01, C10, C11 (local to block)
    
    strassen_terms = [
        # M1
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        # M2
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        # M3
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
        # M4
        ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)),
        # M5
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        # M6
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),
        # M7
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]
    
    rank = num_blocks * 7
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for block in range(num_blocks):
        for t_idx in range(7):
            col = block * 7 + t_idx
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t_idx]
            
            # U: A indices
            # A[i,j] -> index i*m + j
            # u_coeffs order: A00, A01, A10, A11
            U[0, col] = u_coeffs[0] # A00
            U[1, col] = u_coeffs[1] # A01
            U[2, col] = u_coeffs[2] # A10
            U[3, col] = u_coeffs[3] # A11
            
            # V: B indices
            # B[j,k] -> index j*p + k
            # Block 'block' covers columns 2*block and 2*block+1
            # v_coeffs order: B00, B01, B10, B11 (local)
            # Local 00 -> global (0, 2*block)
            # Local 01 -> global (0, 2*block+1)
            # Local 10 -> global (1, 2*block)
            # Local 11 -> global (1, 2*block+1)
            
            col_0 = 2 * block
            col_1 = 2 * block + 1
            
            V[0 * p + col_0, col] = v_coeffs[0] # B0, col_0
            V[0 * p + col_1, col] = v_coeffs[1] # B0, col_1
            V[1 * p + col_0, col] = v_coeffs[2] # B1, col_0
            V[1 * p + col_1, col] = v_coeffs[3] # B1, col_1
            
            # W: C indices
            # C[i,k] -> index k*n + i
            # w_coeffs order: C00, C01, C10, C11 (local)
            # Local 00 -> global (0, 2*block) -> index (2*block)*n + 0
            # Local 01 -> global (0, 2*block+1) -> index (2*block+1)*n + 0
            # Local 10 -> global (1, 2*block) -> index (2*block)*n + 1
            # Local 11 -> global (1, 2*block+1) -> index (2*block+1)*n + 1
            
            W[col_0 * n + 0, col] = w_coeffs[0] # C0, col_0
            W[col_1 * n + 0, col] = w_coeffs[1] # C0, col_1
            W[col_0 * n + 1, col] = w_coeffs[2] # C1, col_0
            W[col_1 * n + 1, col] = w_coeffs[3] # C1, col_1
    
    # Verification
    def verify(Uf, Vf, Wf):
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over rank
                    # T[i,j,k] should be 1
                    val = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
                    if col >= rank: break
                if col >= rank: break
            if col >= rank: break

    if time.time() - _t0 > _DEADLINE - 5:
        pass

    # Convert to lists with integer types if possible
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_list]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_list]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
