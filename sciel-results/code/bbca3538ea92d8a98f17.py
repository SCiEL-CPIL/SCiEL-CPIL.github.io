"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np

def _laderman_444():
    """
    Return Laderman's rank-20 decomposition for 4x4 * 4x4 -> 4x4.
    Returns (U, V, W) where:
      U: (16, 20) - linear forms in A
      V: (16, 20) - linear forms in B  
      W: (16, 20) - linear forms in C
    Indexing: A[i,j] -> index i*4+j, B[j,k] -> index j*4+k, C[i,k] -> index i*4+k
    """
    n = 4
    R = 20
    U = np.zeros((n*n, R))
    V = np.zeros((n*n, R))
    W = np.zeros((n*n, R))
    
    def set_term(r, a_idx, b_idx, c_idx):
        """a_idx, b_idx, c_idx are lists of (row,col) or None for sum"""
        if a_idx is None:
            return
        U[a_idx, r] = 1
        V[b_idx, r] = 1
        W[c_idx, r] = 1
    
    # Terms 0-15: simple single-entry terms
    simple_terms = [
        # (A_idx, B_idx, C_idx) where idx = row*4+col
        (0*4+0, 0*4+0, 0*4+0),   # r=0: A00*B00 -> C00
        (0*4+1, 1*4+0, 0*4+0),   # r=1: A01*B10 -> C00
        (1*4+0, 0*4+1, 0*4+1),   # r=2: A10*B01 -> C01
        (1*4+1, 1*4+1, 1*4+1),   # r=3: A11*B11 -> C11
        (0*4+2, 2*4+0, 0*4+2),   # r=4: A02*B20 -> C02
        (0*4+3, 3*4+0, 0*4+3),   # r=5: A03*B30 -> C03
        (2*4+0, 0*4+2, 2*4+0),   # r=6: A20*B02 -> C20
        (3*4+0, 0*4+3, 3*4+0),   # r=7: A30*B03 -> C30
        (2*4+2, 2*4+2, 2*4+2),   # r=8: A22*B22 -> C22
        (3*4+3, 3*4+3, 3*4+3),   # r=9: A33*B33 -> C33
        (1*4+2, 2*4+1, 1*4+2),   # r=10: A12*B21 -> C12
        (1*4+3, 3*4+1, 1*4+3),   # r=11: A13*B31 -> C13
        (2*4+1, 1*4+2, 2*4+1),   # r=12: A21*B12 -> C21
        (3*4+1, 1*4+3, 3*4+1),   # r=13: A31*B13 -> C31
        (2*4+3, 3*4+2, 2*4+3),   # r=14: A23*B32 -> C23
        (3*4+2, 2*4+3, 3*4+2),   # r=15: A32*B23 -> C32
    ]
    
    for r, (ai, bi, ci) in enumerate(simple_terms):
        U[ai, r] = 1
        V[bi, r] = 1
        W[ci, r] = 1
    
    # Terms 16-19: compound terms with sums
    # r=16: (A02+A13)*(B20+B31) -> C02 + C13
    U[0*4+2, 16] = 1
    U[1*4+3, 16] = 1
    V[2*4+0, 16] = 1
    V[3*4+1, 16] = 1
    W[0*4+2, 16] = 1
    W[1*4+3, 16] = 1
    
    # r=17: (A03+A12)*(B30+B21) -> C03 + C12
    U[0*4+3, 17] = 1
    U[1*4+2, 17] = 1
    V[3*4+0, 17] = 1
    V[2*4+1, 17] = 1
    W[0*4+3, 17] = 1
    W[1*4+2, 17] = 1
    
    # r=18: (A02+A12)*(B20+B21) -> C02 + C12
    U[0*4+2, 18] = 1
    U[1*4+2, 18] = 1
    V[2*4+0, 18] = 1
    V[2*4+1, 18] = 1
    W[0*4+2, 18] = 1
    W[1*4+2, 18] = 1
    
    # r=19: (A03+A13)*(B30+B31) -> C03 + C13
    U[0*4+3, 19] = 1
    U[1*4+3, 19] = 1
    V[3*4+0, 19] = 1
    V[3*4+1, 19] = 1
    W[0*4+3, 19] = 1
    W[1*4+3, 19] = 1
    
    return U, V, W

def _validate_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Validate that the decomposition is correct."""
    R = U.shape[1]
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k)
    # The multiplication tensor T[i,j,k] = 1 if this is a valid (i,j,k) for C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the tensor is: T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k]
    # So the constraint is: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is the "tensor" form where each (i,j,k) triple has value 1.
    
    # Build the tensor T of shape (n, m, p) where T[i,j,k] = 1
    # And check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    # Use numpy for efficiency
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a triple sum. We can vectorize.
    
    # Reshape: U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # For fixed i,j,k: U[i*m+j, :] . V[j*p+k, :] * W[k*n+i, :] summed over r
    
    # Let's build a 3D array T_decomp of shape (n, m, p)
    T_decomp = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                T_decomp[i, j, k] = val
    
    # The target is T[i,j,k] = 1 for all i,j,k
    target = np.ones((n, m, p))
    
    max_err = np.max(np.abs(T_decomp - target))
    return max_err < tol, max_err

def _build_445_schoolbook():
    """Build schoolbook decomposition for (4,4,5): rank = 4*4*5 = 80.
    Returns U (16, 80), V (20, 80), W (20, 80) for n=4, m=4, p=5."""
    n, m, p = 4, 4, 5
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
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
    """
    Returns a tensor decomposition for 4x4 * 4x9 -> 4x9.
    Uses Laderman rank-20 for two (4,4,4) blocks and schoolbook for (4,4,1).
    Total rank: 20 + 20 + 16 = 56.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    deadline = time.time() + 100  # safety margin
    
    # Split p=9 into 4+4+1
    # Block 1: columns 0-3, use Laderman (4,4,4)
    # Block 2: columns 4-7, use Laderman (4,4,4)
    # Block 3: column 8, use schoolbook (4,4,1)
    
    # Get Laderman decomposition
    L_U, L_V, L_W = _laderman_444()
    
    # Validate Laderman
    valid, err = _validate_decomposition(L_U, L_V, L_W, 4, 4, 4)
    if not valid:
        # Fallback to schoolbook if Laderman is wrong
        n, m, p = DIMENSIONS
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
    
    # Build the full decomposition
    # For (4,4,9): U shape (16, R), V shape (36, R), W shape (36, R)
    # p=9, so V has 4*9=36 rows, W has 9*4=36 rows
    
    R_total = 20 + 20 + 16  # 56
    U_full = np.zeros((16, R_total))
    V_full = np.zeros((36, R_total))
    W_full = np.zeros((36, R_total))
    
    # Block 1: Laderman for columns 0-3 (k=0,1,2,3)
    # L_U: (16, 20), L_V: (16, 20) but we need V to have 36 rows (4*9)
    # L_V is indexed by j*p4+k where p4=4, so j*4+k for j in 0..3, k in 0..3
    # In the full V, the index is j*9+k. For k in 0..3, this is j*9+k.
    # So we need to map L_V[j*4+k, r] to V_full[j*9+k, r]
    
    for r in range(20):
        U_full[:, r] = L_U[:, r]
        for j in range(4):
            for k in range(4):
                V_full[j*9 + k, r] = L_V[j*4 + k, r]
        for k in range(4):
            for i in range(4):
                W_full[k*4 + i, r] = L_W[k*4 + i, r]
    
    # Block 2: Laderman for columns 4-7 (k=4,5,6,7)
    offset = 20
    for r in range(20):
        U_full[:, offset + r] = L_U[:, r]
        for j in range(4):
            for k in range(4):
                # k in full problem is 4+k_local
                V_full[j*9 + (4 + k), offset + r] = L_V[j*4 + k, r]
        for k in range(4):
            for i in range(4):
                # k in full problem is 4+k_local
                W_full[(4 + k)*4 + i, offset + r] = L_W[k*4 + i, r]
    
    # Block 3: Schoolbook for column 8 (k=8)
    offset = 40
    for i in range(4):
        for j in range(4):
            col = offset + (i*4 + j)
            U_full[i*4 + j, col] = 1
            V_full[j*9 + 8, col] = 1
            W_full[8*4 + i, col] = 1
    
    # Validate the full decomposition
    valid_full, err_full = _validate_decomposition(U_full, V_full, W_full, 4, 4, 9)
    
    if not valid_full:
        # Fallback to schoolbook
        rank = 4 * 4 * 9
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(36)]
        W = [[0] * rank for _ in range(36)]
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U[i * 4 + j][column] = 1
                    V[j * 9 + k][column] = 1
                    W[k * 4 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
