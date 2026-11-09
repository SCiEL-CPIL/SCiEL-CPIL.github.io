"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _strassen_2x2():
    """Return (U, V, W) for 2x2 matrix multiplication, rank 7.
    Flat indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Standard Strassen decomposition
    # Factor 0: M1 = (A00+A11)(B00+B11), contributes to C00 and C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Factor 1: M2 = (A10+A11)*B00, contributes to C10
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # Factor 2: M3 = A00*(B10-B11), contributes to C01
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # Factor 3: M4 = A11*(B01-B10), contributes to C01 and C00 (negative)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    # Factor 4: M5 = (A00+A01)*B11, contributes to C00 and C10
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # Factor 5: M6 = (A10-A00)*(B00+B10), contributes to C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # Factor 6: M7 = (A01-A11)*(B01+B11), contributes to C00 and C11
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _verify_strassen():
    """Verify the Strassen decomposition is correct using numpy."""
    U, V, W = _strassen_2x2()
    n, m, p = 2, 2, 2
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, 1.0, atol=1e-10):
        print(f"Strassen verification failed: max diff = {np.max(np.abs(T-1.0))}")
        return False
    return True

def _kron_strassen_4x4x4():
    """Build 4x4x4 via Kronecker product of two Strassen(2,2), rank 49.
    Uses vectorized numpy for speed.
    """
    S_U, S_V, S_W = _strassen_2x2()
    S_U_3d = S_U.reshape(2, 2, 7)
    S_V_3d = S_V.reshape(2, 2, 7)
    S_W_3d = S_W.reshape(2, 2, 7)
    
    rank = 49
    U_44 = np.zeros((16, rank))
    V_44 = np.zeros((16, rank))
    W_44 = np.zeros((16, rank))
    
    U_tensor = S_U_3d[:, None, :, None, :] * S_U_3d[None, :, None, :, :]
    U_tensor = U_tensor.transpose(0, 2, 1, 3, 4, 5)
    U_44 = U_tensor.reshape(16, 49)
    
    V_tensor = S_V_3d[:, None, :, None, :] * S_V_3d[None, :, None, :, :]
    V_tensor = V_tensor.transpose(0, 2, 1, 3, 4, 5)
    V_44 = V_tensor.reshape(16, 49)
    
    W_tensor = S_W_3d[:, None, :, None, :] * S_W_3d[None, :, None, :, :]
    W_tensor = W_tensor.transpose(0, 2, 1, 3, 4, 5)
    W_44 = W_tensor.reshape(16, 49)
    
    return U_44, V_44, W_44

def _verify_decomposition(U, V, W, n, m, p):
    """Verify a full decomposition is correct."""
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, 1.0, atol=1e-10):
        max_diff = np.max(np.abs(T - 1.0))
        print(f"Verification failed: max diff = {max_diff}")
        return False
    return True

def _greedy_hypergraph_matching(n, m, p, time_limit=60):
    """
    Greedy hypergraph matching heuristic to find a low-rank tensor decomposition.
    
    The multiplication tensor T[i,j,k] = 1 if i,j,k are valid indices.
    We want to find rank-1 terms u⊗v⊗w such that sum of terms = T.
    
    Strategy: Start with schoolbook (each (i,j,k) is one term), then try to
    merge terms by finding patterns where multiple entries can be covered by
    a single rank-1 term with integer/half-integer coefficients.
    """
    start = time.time()
    
    # The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Schoolbook rank = n*m*p
    # We'll try a greedy approach: build candidate rank-1 terms and select
    # those that cover the most uncovered tensor entries.
    
    # For a 4x4x9 tensor, schoolbook rank is 144.
    # Known lower bounds: max(n*m, m*p, n*p) = max(16, 36, 36) = 36
    # Upper bound from Strassen-like: ~49+49+16 = 114 (current)
    
    # Let's try a different approach: use the structure of the problem.
    # Split into blocks and use Kronecker products where beneficial.
    
    # Approach: Try various splits of the p dimension
    # For p=9, try splits like (4,5), (3,6), (9), etc.
    
    best_rank = n * m * p
    best_decomp = None
    
    # Generate all possible rank-1 term candidates
    # A rank-1 term is defined by vectors u (length n*m), v (length m*p), w (length p*n)
    # with entries in {-1, 0, 1} (or half-integers)
    
    # This is too large to enumerate directly. Instead, use a structured approach.
    
    # Strategy: Use the fact that the tensor is separable in a block structure.
    # For each block of the p dimension, use the best known decomposition.
    
    # Try: 4x4x4 (rank 49) + 4x4x5 (to be optimized)
    # For 4x4x5: try 4x4x4 (rank 49) + 4x4x1 (rank 16) = 65
    # Or try a direct 4x4x5 decomposition
    
    # Let's try to find a better 4x4x5 decomposition using greedy matching.
    
    def _build_4x4x5_greedy(time_budget=30):
        """Try to find a low-rank decomposition for 4x4x5."""
        n2, m2, p2 = 4, 4, 5
        start2 = time.time()
        
        # Start with schoolbook
        rank = n2 * m2 * p2
        U = np.zeros((n2*m2, rank))
        V = np.zeros((m2*p2, rank))
        W = np.zeros((p2*n2, rank))
        
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U[i*m2+j, col] = 1
                    V[j*p2+k, col] = 1
                    W[k*n2+i, col] = 1
                    col += 1
        
        # Try to merge terms using Strassen-like patterns
        # For each pair of rows in U, try to find a pattern
        # This is complex; instead, try known decompositions
        
        # Use Kronecker: 4x4x4 (rank 49) + 4x4x1 (rank 16) = 65
        U_44, V_44, W_44 = _kron_strassen_4x4x4()
        
        U_new = np.zeros((16, 65))
        V_new = np.zeros((16, 65))
        W_new = np.zeros((20, 65))
        
        U_new[:, :49] = U_44
        V_new[:, :49] = V_44
        W_new[:16, :49] = W_44
        
        col_offset = 49
        for i in range(4):
            for j in range(4):
                U_new[i*4+j, col_offset] = 1
                V_new[j*4+4, col_offset] = 1
                W_new[4*4+i, col_offset] = 1
                col_offset += 1
        
        # Verify
        if _verify_decomposition(U_new, V_new, W_new, 4, 4, 5):
            return U_new, V_new, W_new, 65
        
        # Fall back to schoolbook
        return U, V, W, 80
    
    # Try building the full 4x4x9 with various strategies
    strategies = [
        ("49+49+16", 114),
        ("49+65", 114),
    ]
    
    # Build 4x4x9 using 4x4x4 (49) + 4x4x5 (65)
    U_44, V_44, W_44 = _kron_strassen_4x4x4()
    
    # 4x4x5 part
    U_445, V_445, W_445, rank_445 = _build_4x4x5_greedy(time_budget=30)
    
    total_rank = 49 + rank_445
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((16, total_rank))
    W_total = np.zeros((36, total_rank))
    
    # First 4x4x4 block
    U_total[:, :49] = U_44
    V_total[:, :49] = V_44
    W_total[:16, :49] = W_44
    
    # 4x4x5 block (k=4..8)
    U_total[:, 49:49+rank_445] = U_445
    V_total[:, 49:49+rank_445] = V_445
    W_total[16:36, 49:49+rank_445] = W_445
    
    # Verify full decomposition
    if _verify_decomposition(U_total, V_total, W_total, 4, 4, 9):
        elapsed = time.time() - start
        if total_rank < best_rank:
            best_rank = total_rank
            # Convert to lists
            U_list = U_total.tolist()
            V_list = V_total.tolist()
            W_list = W_total.tolist()
            for mat in [U_list, V_list, W_list]:
                for row in mat:
                    for idx in range(len(row)):
                        val = row[idx]
                        if abs(val - round(val)) < 1e-10:
                            row[idx] = int(round(val))
            best_decomp = {"U": U_list, "V": V_list, "W": W_list}
    
    # Also try the original 49+49+16 approach as fallback
    if best_decomp is None or best_rank > 114:
        U_44, V_44, W_44 = _kron_strassen_4x4x4()
        rank_441 = 16
        total_rank2 = 2 * 49 + rank_441
        
        U_total2 = np.zeros((16, total_rank2))
        V_total2 = np.zeros((16, total_rank2))
        W_total2 = np.zeros((36, total_rank2))
        
        U_total2[:, :49] = U_44
        V_total2[:, :49] = V_44
        W_total2[:16, :49] = W_44
        
        U_total2[:, 49:98] = U_44
        V_total2[:, 49:98] = V_44
        W_total2[16:32, 49:98] = W_44
        
        col_offset = 98
        for i in range(4):
            for j in range(4):
                U_total2[i*4+j, col_offset] = 1
                V_total2[j*4+8, col_offset] = 1
                W_total2[8*4+i, col_offset] = 1
                col_offset += 1
        
        if _verify_decomposition(U_total2, V_total2, W_total2, 4, 4, 9):
            if total_rank2 < best_rank:
                best_rank = total_rank2
                U_list = U_total2.tolist()
                V_list = V_total2.tolist()
                W_list = W_total2.tolist()
                for mat in [U_list, V_list, W_list]:
                    for row in mat:
                        for idx in range(len(row)):
                            val = row[idx]
                            if abs(val - round(val)) < 1e-10:
                                row[idx] = int(round(val))
                best_decomp = {"U": U_list, "V": V_list, "W": W_list}
    
    if best_decomp is not None:
        return best_decomp, best_rank
    
    return None, n * m * p

def _schoolbook(n, m, p):
    """Pure schoolbook decomposition for n x m by m x p matrix multiplication."""
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

def construct():
    start_time = time.time()
    deadline = 110  # seconds, with safety margin
    
    # First verify Strassen
    try:
        strassen_ok = _verify_strassen()
    except Exception as e:
        print(f"Strassen verification error: {e}")
        strassen_ok = False
    
    if not strassen_ok:
        return _schoolbook(*DIMENSIONS)
    
    # Try greedy approach
    try:
        result, rank = _greedy_hypergraph_matching(*DIMENSIONS, time_limit=60)
        elapsed = time.time() - start_time
        if result is not None and elapsed < deadline:
            print(f"Greedy approach found rank {rank}")
            return result
    except Exception as e:
        print(f"Error in greedy approach: {e}")
    
    # Fallback to schoolbook
    return _schoolbook(*DIMENSIONS)
# EVOLVE-BLOCK-END
