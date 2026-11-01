"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 11]

def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _build_strassen_blocks():
    """
    Build a decomposition using Strassen for 2x2 blocks.
    For 2x2 * 2x11: use Strassen on pairs of columns.
    
    11 columns = 5 pairs + 1 single column
    Each Strassen block (2x2 * 2x2) has rank 7
    Each single column (2x2 * 2x1) has rank 2 (schoolbook)
    
    Total: 5*7 + 2 = 37
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5 pairs
    remainder = p % 2   # 1 remaining column
    
    rank = num_pairs * 7 + remainder * 2
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, rank)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (22, rank)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (22, rank)
    
    # Strassen terms for 2x2 * 2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # U coefficients (index = i*m+j for A[i,j]):
    strassen_U = np.array([
        [1, 0, 0, 1],  # P1: A00+A11
        [0, 0, 1, 1],  # P2: A10+A11
        [1, 0, 0, 0],  # P3: A00
        [0, 0, 0, 1],  # P4: A11
        [1, 1, 0, 0],  # P5: A00+A01
        [-1, 0, 1, 0], # P6: A10-A00
        [0, 1, -1, 0], # P7: A01-A11
    ], dtype=np.float64)
    
    # V coefficients (index = j*p_local+k_local for B[j,k] in 2x2 block):
    strassen_V = np.array([
        [1, 0, 0, 1],  # P1: B00+B11
        [1, 0, 0, 0],  # P2: B00
        [0, 0, 1, -1], # P3: B10-B11
        [-1, 0, 0, 1], # P4: B01-B00
        [0, 0, 0, 1],  # P5: B11
        [1, 0, 0, 1],  # P6: B00+B01
        [0, 0, 1, 1],  # P7: B10+B11
    ], dtype=np.float64)
    
    # W reconstruction coefficients for C[i,k]:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    strassen_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ], dtype=np.float64)
    
    col = 0
    for b in range(num_pairs):
        for t in range(7):
            # U: same for all blocks
            U[:, col] = strassen_U[t]
            
            # V: block b uses columns 2b and 2b+1
            # B[j, 2b+kp] has index j*p + (2b+kp)
            for j in range(2):
                for kp in range(2):
                    local_idx = j * 2 + kp
                    global_idx = j * p + (2 * b + kp)
                    V[global_idx, col] = strassen_V[t, local_idx]
            
            # W: block b uses output columns 2b and 2b+1
            # C[i, 2b+kp] has index (2b+kp)*n + i
            for i in range(2):
                for kp in range(2):
                    local_idx = i * 2 + kp
                    global_idx = (2 * b + kp) * n + i
                    W[global_idx, col] = strassen_W[local_idx, t]
            
            col += 1
    
    # Handle remainder column (column 10)
    if remainder == 1:
        rem_col = num_pairs * 2  # = 10
        
        # Term 1: C[i,10] += A[i,0]*B[0,10]
        U[0, col] = 1.0   # A[0,0], index 0*2+0=0
        U[2, col] = 1.0   # A[1,0], index 1*2+0=2
        V[0 * p + rem_col, col] = 1.0  # B[0,10], index 0*11+10=10
        W[rem_col * n + 0, col] = 1.0  # C[0,10], index 10*2+0=20
        W[rem_col * n + 1, col] = 1.0  # C[1,10], index 10*2+1=21
        col += 1
        
        # Term 2: C[i,10] += A[i,1]*B[1,10]
        U[1, col] = 1.0   # A[0,1], index 0*2+1=1
        U[3, col] = 1.0   # A[1,1], index 1*2+1=3
        V[1 * p + rem_col, col] = 1.0  # B[1,10], index 1*11+10=21
        W[rem_col * n + 0, col] = 1.0  # C[0,10], index 10*2+0=20
        W[rem_col * n + 1, col] = 1.0  # C[1,10], index 10*2+1=21
        col += 1
    
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p  # 44
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _attempt_rank_reduction(U, V, W, T, time_limit=100):
    """
    Attempt to reduce rank by removing redundant terms.
    Uses linear algebra to find if any term is a linear combination of others.
    """
    start_time = time.time()
    n, m, p = 2, 2, 11
    dim_a, dim_b, dim_c = n * m, m * p, p * n  # 4, 22, 22
    
    current_rank = U.shape[1]
    if current_rank <= 1:
        return U, V, W
    
    # Flatten each rank-1 tensor into a vector
    # Term r contributes: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Flatten: U[:,r][a] * V[:,r][b] * W[:,r][c] at position (a,b,c)
    
    # Build matrix M where each column is a flattened rank-1 tensor
    M = np.zeros((dim_a * dim_b * dim_c, current_rank), dtype=np.float64)
    for r in range(current_rank):
        M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
    
    # Find rank of M
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    except Exception:
        return U, V, W
    
    if rank_M >= current_rank:
        # No reduction possible
        return U, V, W
    
    # Find which columns are linearly dependent
    # Use QR with column pivoting
    try:
        Q, R = np.linalg.qr(M, mode='complete')
        # Find pivot columns
        diag_R = np.abs(np.diag(R))
        pivot_cols = np.argsort(diag_R)[::-1][:rank_M]
        
        # Keep only pivot columns
        U_new = U[:, pivot_cols]
        V_new = V[:, pivot_cols]
        W_new = W[:, pivot_cols]
        
        # Verify
        if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), T):
            if U_new.shape[1] < current_rank:
                return U_new, V_new, W_new
    except Exception:
        pass
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin before 150s limit
    
    T = _build_target_tensor()
    
    # First, try the Strassen-based rank-37 construction
    U, V, W = _build_strassen_blocks()
    
    if _verify(U.tolist(), V.tolist(), W.tolist(), T):
        # Try rank reduction
        U_red, V_red, W_red = _attempt_rank_reduction(U, V, W, T, time_limit=80)
        if U_red.shape[1] < U.shape[1]:
            return {'U': U_red.tolist(), 'V': V_red.tolist(), 'W': W_red.tolist()}
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fall back to schoolbook rank 44
    U, V, W = _build_schoolbook()
    if _verify(U.tolist(), V.tolist(), W.tolist(), T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Should not reach here
    raise RuntimeError("No valid decomposition found")
