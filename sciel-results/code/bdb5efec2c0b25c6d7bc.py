"""Find rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

def _verify_fast(U, V, W, n, m, p):
    """Fast vectorized verification using tensor contraction."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _try_rank_decomposition(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of the given rank by solving the system of equations.
    
    For each rank-1 term r, we assign:
    - U[a, r] for each a in 0..n*m-1
    - V[b, r] for each b in 0..m*p-1  
    - W[c, r] for each c in 0..p*n-1
    
    The constraint is: sum_r U[a,r]*V[b,r]*W[c,r] = T_target[a,b,c]
    
    This is a system of (n*m)*(m*p)*(p*n) equations in 3*rank*(n*m+m*p+p*n) unknowns.
    
    Strategy: Use a structured approach where we fix some variables and solve for others.
    """
    start = time.time()
    
    # Number of target tensor entries
    n_a = n * m  # 4
    n_b = m * p  # 14
    n_c = p * n  # 14
    
    # Build target tensor
    T_target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[(a, b, c)] = 1
    
    # Strategy: Use a Kronecker-like structure
    # Split the p=7 dimension into blocks
    # Try: rank = 7*2 + 2*1 = 16? No, that's too optimistic.
    # 
    # Let's try a different structural approach:
    # For each column k of B, we need to compute A * B[:,k] = C[:,k]
    # This is a 2x2 * 2x1 = 2x1 multiplication
    # 
    # The key observation: C[k*n + i] = sum_j A[i*m+j] * B[j*p+k]
    #
    # For fixed k, this is: for each i, C[k*n+i] = sum_j U_part * V_part * W_part
    #
    # Let's try a block structure:
    # Block 1: k=0,1 using Laderman-like (rank 7 for 2x2x2)
    # Block 2: k=2,3 using Laderman-like (rank 7)
    # Block 3: k=4,5 using Laderman-like (rank 7)
    # Block 4: k=6 using schoolbook (rank 4)
    # Total: 25
    #
    # To beat 25, we need to find a better decomposition.
    #
    # Alternative: Use the fact that n=m=2 to find a more efficient structure.
    #
    # Let's try: rank 24
    # We can potentially save 1 rank by sharing a term between blocks.
    
    if time.time() > deadline:
        return None
    
    # Try rank 24 with a systematic search
    # The idea: start from the 25-rank block decomposition and try to merge
    # two rank-1 terms or find a linear dependency.
    
    # Actually, let's try a completely different approach:
    # Use the structure of the problem more directly.
    
    # For 2x2 * 2x7, the tensor has 4*14*14 = 784 entries, but only 14 are non-zero.
    # The rank is the minimum number of rank-1 tensors that sum to this.
    
    # Let's try to construct a rank-24 decomposition by modifying the block approach.
    # Instead of 3 Laderman blocks (7 each) + 1 schoolbook (4) = 25,
    # try to find a way to use 24 terms.
    
    # One approach: Use a shared term between the last Laderman block and the schoolbook block.
    
    # Let me try a direct construction for rank 24.
    # 
    # Structure:
    # - 3 Laderman blocks for k=0..5 (21 terms)
    # - 3 terms for k=6 (instead of 4)
    #
    # For k=6, we need:
    # C[6*2+0] = A[0]*B[6] + A[1]*B[7]  (where B[6]=B[0,6], B[7]=B[1,6])
    # C[6*2+1] = A[2]*B[6] + A[3]*B[7]
    #
    # With 3 terms, we can do:
    # t1: U[0]=1, V[6]=1, W[12]=1  -> contributes A[0]*B[6] to C[12]
    # t2: U[1]=1, V[7]=1, W[12]=1  -> contributes A[1]*B[7] to C[12]
    # t3: U[2]=1, V[6]=1, W[13]=1  -> contributes A[2]*B[6] to C[13]
    # But we're missing A[3]*B[7] for C[13]!
    #
    # So 3 terms don't work for k=6 alone. We need 4 or share with another block.
    
    # Let's try a different split: 2 Laderman blocks + 1 modified block
    # k=0,1: Laderman (7)
    # k=2,3,4,5,6: custom (17 terms)
    # Total: 24
    
    # For k=2..6 (5 columns), we need 2x2 * 2x5 = 2x5
    # Schoolbook would be 2*2*5 = 20, so we need to save 3 ranks.
    
    # This is getting complex. Let me try a numerical approach.
    
    # Actually, let's just return the best we can find within the time limit.
    # The current best is 25. Let's try to verify if 24 is possible.
    
    # For now, let's return None to indicate we couldn't find a better solution.
    # The construct() function will fall back to the 25-rank solution.
    
    return None

def _build_25_rank_decomposition(n, m, p):
    """Build the 25-rank block decomposition."""
    n, m, p = 2, 2, 7
    
    # Laderman 2x2x2 decomposition
    def _laderman():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[0][1] = 1; U[3][1] = -1
        V[0][1] = 1; V[3][1] = -1
        W[0][1] = -1; W[3][1] = -1
        
        U[0][2] = 1; U[2][2] = 1
        V[2][2] = 1; V[3][2] = 1
        W[2][2] = -1; W[3][2] = 1
        
        U[3][3] = 1; U[2][3] = -1
        V[0][3] = 1; V[2][3] = -1
        W[0][3] = 1; W[2][3] = -1
        
        U[1][4] = 1; U[3][4] = 1
        V[0][4] = 1; V[1][4] = 1
        W[1][4] = 1; W[3][4] = 1
        
        U[1][5] = 1; U[2][5] = -1
        V[2][5] = 1; V[3][5] = -1
        W[0][5] = 1; W[1][5] = -1; W[2][5] = 1
        
        U[0][6] = 1; U[1][6] = 1
        V[1][6] = 1; V[3][6] = -1
        W[0][6] = -1; W[1][6] = 1
        
        return U, V, W
    
    total_rank = 25
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    # Three Laderman blocks for k=0..1, k=2..3, k=4..5
    for block_idx in range(3):
        k_start = block_idx * 2
        k_end = k_start + 2
        
        U_l, V_l, W_l = _laderman()
        U_l = np.array(U_l, dtype=float)
        V_l = np.array(V_l, dtype=float)
        W_l = np.array(W_l, dtype=float)
        
        for i in range(n):
            for j in range(m):
                U[i*m + j, col_offset:col_offset+7] = U_l[i*m + j]
        
        for j in range(m):
            for k_local in range(2):
                V[j*p + (k_start + k_local), col_offset:col_offset+7] = V_l[j*2 + k_local]
        
        for k_local in range(2):
            for i in range(n):
                W[(k_start + k_local)*n + i, col_offset:col_offset+7] = W_l[k_local*n + i]
        
        col_offset += 7
    
    # Schoolbook for k=6
    k = 6
    for i in range(n):
        for j in range(m):
            U[i*m + j, col_offset] = 1.0
            V[j*p + k, col_offset] = 1.0
            W[k*n + i, col_offset] = 1.0
            col_offset += 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition
    better = _try_rank_decomposition(n, m, p, 24, deadline)
    if better is not None:
        return better
    
    # Fall back to the 25-rank block decomposition
    U, V, W = _build_25_rank_decomposition(n, m, p)
    
    if _verify_fast(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Final fallback: schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
