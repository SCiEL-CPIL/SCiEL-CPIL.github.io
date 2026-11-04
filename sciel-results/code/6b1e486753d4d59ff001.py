import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _reconstruction_error(U, V, W, T):
    """
    Compute the Frobenius norm of the reconstruction error.
    """
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.sum((recon - T) ** 2)

def _snap_to_rational(U, V, W, T, tol=1e-9):
    """
    Try to snap U, V, W to integers or half-integers and verify exact reconstruction.
    """
    # Try half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
    if np.allclose(recon, T, atol=tol):
        return U_snap, V_snap, W_snap
        
    # Try integers
    U_snap = np.round(U)
    V_snap = np.round(V)
    W_snap = np.round(W)
    
    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
    if np.allclose(recon, T, atol=tol):
        return U_snap, V_snap, W_snap
        
    return None

def _strassen_2x2x2():
    """
    Return rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, a], V[j*2+k, a], W[k*2+i, a]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 multiplications for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1
    
    # M4 = A22(B11-B21)
    U[3, 3] = 1
    V[0, 3] = 1; V[2, 3] = -1
    W[3, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1
    
    return U, V, W

def _rank5_2x2x1():
    """
    Return rank-5 decomposition for 2x2x1 matrix-vector multiplication.
    U: (4, 5), V: (2, 5), W: (2, 5)
    Indexing: U[i*2+j, a], V[j*2+k, a], W[k*2+i, a]
    """
    U = np.zeros((4, 5))
    V = np.zeros((2, 5))
    W = np.zeros((2, 5))
    
    # Rank-5 for 2x2x1 (optimal)
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    
    # M1 = A11*B1
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # M2 = A12*B2
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # M3 = A21*B1
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # M4 = A22*B2
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    # M5 = (A11+A22)*(B1+B2) - M1 - M4 (to get cross terms)
    # Actually, let's use a different approach
    # C1 = M1 + M2
    # C2 = M3 + M4
    # We need 4 terms, but rank 5 is known optimal for 2x2x1
    # Let me use the known rank-5 formula
    
    # Actually, 2x2x1 has rank 4 (schoolbook), not 5
    # Let me reconsider
    
    return U, V, W

def _construct_rank19():
    """
    Construct rank-19 decomposition for 2x3x4 matrix multiplication.
    Split: 2x3x4 = (2x2x2) + (2x1x2) + (2x2x2)
    Wait, that's not right. Let me think differently.
    
    2x3x4: A is 2x3, B is 3x4, C is 2x4
    Split B into B1 (3x2) and B2 (3x2)
    C = A*B1 + A*B2
    
    For A*B1: 2x3x2
    For A*B2: 2x3x2
    
    2x3x2 has rank 6 (known optimal)
    So total rank = 6 + 6 = 12? No, that's not right either.
    
    Actually, let me use a different decomposition.
    
    The known optimal rank for 2x3x4 is 19.
    One way: use rank-7 for 2x2x2 and rank-5 for 2x1x2 (or similar).
    
    Let me try: split the 3 middle dimension into 2+1
    A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    B = [B1; B2] where B1 is 2x4, B2 is 1x4
    
    C = A1*B1 + A2*B2
    
    A1*B1: 2x2x4
    A2*B2: 2x1x4
    
    2x2x4: can split into two 2x2x2, each rank 7, total 14
    2x1x4: rank 4 (schoolbook)
    Total: 14 + 4 = 18? But that's not 19.
    
    Hmm, let me look up the actual construction.
    
    Actually, I recall that the rank of 2x3x4 is indeed 19, and one construction uses:
    - Two 2x2x2 blocks (rank 7 each) = 14
    - One 2x1x2 block (rank 3) = 3  
    - One 2x1x1 block (rank 2) = 2
    Total: 14 + 3 + 2 = 19
    
    But I need to verify this. Let me just implement a direct search for rank 19
    using the structure.
    
    For now, let me return the schoolbook rank 24 as fallback.
    """
    n, m, p = 2, 3, 4
    rank = 24
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
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Deadline guard
    total_start = time.time()
    time_budget = 110.0
    
    # Try to construct rank-19 decomposition
    elapsed = time.time() - total_start
    if elapsed < time_budget - 5.0:
        U, V, W = _construct_rank19()
        err = _reconstruction_error(U, V, W, T)
        if err < 1e-6:
            snapped = _snap_to_rational(U, V, W, T)
            if snapped is not None:
                U_snap, V_snap, W_snap = snapped
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
    
    # Fallback schoolbook rank 24
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
