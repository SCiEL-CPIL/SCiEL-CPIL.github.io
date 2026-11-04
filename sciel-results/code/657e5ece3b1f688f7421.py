"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""

import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor T[i,j,k] = delta."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reconstruct tensor: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m + j, j*p + k, k*n + i] = 1.0
    
    return np.allclose(recon, target, atol=tol)

def _try_rank_decomp(n, m, p, target, max_rank, time_budget, seed=42):
    """Try to find a rank-r decomposition using random initialization + refinement."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    for rank in range(1, max_rank + 1):
        # Try multiple random initializations
        for attempt in range(5):
            U = np_rng.normal(size=(n*m, rank))
            V = np_rng.normal(size=(m*p, rank))
            W = np_rng.normal(size=(p*n, rank))
            
            # Simple gradient-free refinement: just check if we got lucky
            if _check_decomposition(U, V, W, n, m, p, tol=1e-6):
                return rank, U.tolist(), V.tolist(), W.tolist()
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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
    
    return rank, U, V, W

def _kron_decomposition(n, m, p):
    """
    Try Kronecker product approach: decompose into sub-blocks.
    For 3x3 x 3x7, split the 7 columns into groups and use block structure.
    """
    n, m, p = 3, 3, 7
    # Split p=7 into 3+3+1
    p_groups = [3, 3, 1]
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for pg in p_groups:
        # For each group of pg columns, we need rank n*m*pg for schoolbook
        # But we can try to share structure
        pass
    
    # This approach is complex; fall back to schoolbook
    return None

def construct():
    """
    Returns a valid rank decomposition for 3x3 by 3x7 matrix multiplication.
    Attempts to find a lower-rank decomposition, falls back to schoolbook (rank 63).
    
    Shapes:
      U: (9, rank)  -- n*m = 9 rows
      V: (21, rank) -- m*p = 21 rows
      W: (21, rank) -- p*n = 21 rows, indexed as W[k*n+i, r]
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin for 150s limit
    
    # First, get the schoolbook as fallback
    sb_rank, U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = sb_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Build target tensor for verification
    target = _build_target_tensor(n, m, p)
    
    # Try Kronecker-style structured decomposition
    # View 3x3 x 3x7 as: for each output column k, compute A * b_k
    # This is 7 independent 3x3 x 3x1 multiplications
    # Each 3x3 x 3x1 has schoolbook rank 9, total 7*9=63
    # Try to find a shared rank-r decomposition for 3x3 x 3x1
    
    # Approach: Use the fact that 3x3 x 3x1 tensor can be seen as
    # T[i,j,k] = delta_{i,k} for the "identity" part, but actually
    # for matrix-vector: C[i] = sum_j A[i,j] * B[j]
    # Tensor: T[i,j,k] where i is output row, j is input col, k is... 
    # Actually for 3x3 x 3x1: U is (9, r), V is (3, r), W is (3, r)
    # T[i,j,k] = delta_{i,k} * 1 (for the multiplication tensor)
    
    # Let's try to find rank-r for 3x3 x 3x1 (matrix-vector mult)
    # Target tensor for 3x3 x 3x1: T[i,j,k] = 1 if i==k else 0, for all j
    # Wait, that's not right. The tensor for A(3x3)*b(3x1)=c(3x1) is:
    # c[i] = sum_j A[i,j] b[j]
    # So T[i,j,k] should be such that sum_{i,j} A[i,j] b[j] T[i,j,k] = c[k]
    # This means T[i,j,k] = delta_{i,k} (for each j)
    # So the tensor is: T[i,j,k] = 1 if i==k, else 0
    
    mv_target = np.zeros((9, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == k:
                    mv_target[i*3 + j, j, k] = 1.0
    
    # Try to find low-rank decomposition for 3x3 x 3x1
    # Schoolbook rank is 9
    # Try ranks from 4 to 8 (Strassen-like for smaller sizes)
    
    found_mv = False
    for r in range(4, 9):
        if time.time() > deadline - 20:
            break
        for attempt in range(10):
            np_rng = np.random.default_rng(1000 + attempt)
            U_mv = np_rng.normal(size=(9, r))
            V_mv = np_rng.normal(size=(3, r))
            W_mv = np_rng.normal(size=(3, r))
            
            # Check: sum_r U[i*m+j, r] * V[j, r] * W[k, r] should equal mv_target[i*m+j, j, k]
            recon = np.einsum('ir,jr,kr->ijk', U_mv, V_mv, W_mv)
            if np.allclose(recon, mv_target, atol=1e-6):
                found_mv = True
                break
        if found_mv:
            break
    
    if found_mv:
        # If we found a rank-r decomposition for 3x3 x 3x1, 
        # we can potentially use it for the full problem
        # But this doesn't directly give us the 3x3 x 3x7 decomposition
        # unless we can share the U factors across all 7 columns
        pass
    
    # Given the complexity, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
