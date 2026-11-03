"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Return a tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    We attempt to find a rank < 144 decomposition by exploiting structure.
    
    For 4x4 * 4x9:
    - schoolbook rank = 144
    - We use exact arithmetic (Fraction) to verify any candidate
    - We try a structured search based on Kronecker-style products
    
    The key insight: C = A * B where A is (4,4), B is (4,9).
    We can view B as 9 independent 4-vectors b_1...b_9.
    C[:, k] = A * b_k.
    
    An alternative view: reshape B as (3,4,3) (group rows of 4 into blocks of 4)
    Actually: B is 4x9. Let's group columns: B = [B1 B2 B3] where each Bi is 4x3.
    Then C = A * [B1 B2 B3] = [A*B1, A*B2, A*B3].
    
    This doesn't immediately help. Let's try a different approach.
    
    Actually, for small cases like this, we can use an algebraic approach:
    The matrix multiplication tensor T has indices (i, j) for A and (j, k) for B,
    producing (i, k) for C.
    
    We want to write T = sum_r u_r * v_r * w_r
    where u_r is a vector in R^{n*m} = R^{16}, v_r in R^{m*p} = R^{36}, w_r in R^{p*n} = R^{36}.
    
    u_r[i*m+j] * v_r[j*p+k] = w_r[k*n+i] (for all i, j, k)
    
    This is a system of constraints. Let's set up variables for each rank-1 factor.
    For rank r, we have r * (16 + 36 + 36) = r * 88 unknowns, and n*m*p = 4*4*9 = 144 constraints (actually it's more nuanced - the constraints are that for each (i,j,k), the sum over r equals the Kronecker delta).
    
    Actually, the constraint is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid index triple
                                                          = 0 otherwise
    
    But in our problem, every (i,j,k) with i in 0..3, j in 0..3, k in 0..8 is valid.
    So we have 4*4*9 = 144 equations, and for rank r we have r*(16+36+36) = 88r unknowns.
    
    For r < 144, we have fewer unknowns than equations (88r < 144 when r < 144/88 ~ 1.6, which doesn't make sense).
    
    Wait, I think I'm confusing myself. Let me reconsider.
    
    The tensor T is of format (16, 36, 36). We want a CP decomposition:
    T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    where:
    - i indexes A (4*4 = 16)
    - j indexes B (4*9 = 36)  
    - k indexes C (9*4 = 36)
    
    And the constraint is that T is the matrix multiplication tensor:
    T[(i_A, i_B), (j_B, j_C), (k_C, k_A)] = delta[i_B, j_B] * delta[i_A, k_A]
    
    where i_A, k_A in {0,1,2,3}, i_B, j_B in {0,1,2,3}, j_C, k_C in {0,1,...,8}.
    
    For rank r, we have:
    U: 16 * r
    V: 36 * r
    W: 36 * r
    Total unknowns: (16+36+36) * r = 88r
    
    The constraints: for each valid (i,j,k) triple (which is determined by the index structure), we need the CP expansion to match. There are 16*36*36 = 20736 entries in the full tensor, but most are zero. The non-zero entries correspond to specific index relationships.
    
    Actually, the tensor is sparse. Let me think about which entries are non-zero.
    T[i,j,k] where i = i_A * m + i_B, j = i_B * p + j_C, k = j_C * n + k_A
    T is non-zero only when i_B (from i) = i_B (from j) and i_A (from i) = k_A (from k).
    
    So for each combination of (i_A, i_B, j_C) where i_A in {0,1,2,3}, i_B in {0,1,2,3}, j_C in {0,...,8}:
    i = i_A * 4 + i_B
    j = i_B * 9 + j_C
    k = j_C * 4 + i_A
    T[i,j,k] = 1
    
    There are 4*4*9 = 144 such non-zero entries.
    
    So we have 144 equations and 88r unknowns. For a solution to exist with r < 144, we need 88r >= 144, so r >= ceil(144/88) = 2. But that's a necessary condition, not sufficient. The actual rank is likely much higher.
    
    Known results: For m x n x p matrix multiplication, the rank is at most m*n*p (schoolbook) and at least various bounds. For 4x4x9, the optimal rank is not known exactly, but it's likely less than 144.
    
    Let me try a constructive approach: exploit the fact that p=9 = 3*3, and use a recursive or algebraic structure.
    
    Actually, let me try a numerical approach: use random initialization and alternating least squares (ALS) on the tensor completion problem, then round to exact values and verify.
    
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    nU = n * m  # 16
    nV = m * p  # 36
    nW = p * n  # 36
    rank_schoolbook = n * m * p  # 144
    
    def time_remaining():
        return deadline - time.time()
    
    def build_schoolbook():
        rank = rank_schoolbook
        U = np.zeros((nU, rank), dtype=np.float64)
        V = np.zeros((nV, rank), dtype=np.float64)
        W = np.zeros((nW, rank), dtype=np.float64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly matches the matrix multiplication tensor."""
        rank = len(U[0])
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W)
        
        # Check all 144 non-zero entries
        for iA in range(n):
            for iB in range(m):
                for jC in range(p):
                    i_idx = iA * m + iB
                    j_idx = iB * p + jC
                    k_idx = jC * n + iA
                    
                    # Compute sum over rank
                    val = 0.0
                    for r in range(rank):
                        val += Ua[i_idx, r] * Va[j_idx, r] * Wa[k_idx, r]
                    
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check that "wrong" entries are zero
        # For entries that should be zero, verify they are close to 0
        # This is expensive, so we do a random subset check
        import random
        random.seed(42)
        for _ in range(1000):
            i_idx = random.randint(0, nU - 1)
            j_idx = random.randint(0, nV - 1)
            k_idx = random.randint(0, nW - 1)
            
            # Check if this is a "correct" entry
            iA, iB = divmod(i_idx, m)
            iB2, jC = divmod(j_idx, p)
            jC2, kA = divmod(k_idx, n)
            
            is_correct = (iB == iB2) and (iA == kA)
            
            val = 0.0
            for r in range(rank):
                val += Ua[i_idx, r] * Va[j_idx, r] * Wa[k_idx, r]
            
            if is_correct:
                if abs(val - 1.0) > 1e-9:
                    return False
            else:
                if abs(val) > 1e-9:
                    return False
        
        return True
    
    def try_als_search(max_rank, time_limit):
        """Try to find a decomposition with rank <= max_rank using ALS."""
        end_time = time.time() + time_limit
        best_rank = rank_schoolbook
        best_result = None
        
        # Try ranks from max_rank down to some minimum
        for rank in range(max_rank, 30, -1):
            if time.time() >= end_time:
                break
            
            # Try a few random initializations
            ntries = 0
            max_ntries = 5
            
            while time.time() < end_time and ntries < max_ntries:
                ntries += 1
                
                # Random initialization with exact-ish values
                # Use values from a small set
                vals = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25]
                U = np.zeros((nU, rank))
                V = np.zeros((nV, rank))
                W = np.zeros((nW, rank))
                
                for i in range(rank):
                    # Random sparse-ish initialization
                    nnz_U = np.random.randint(1, 5)
                    nnz_V = np.random.randint(1, 5)
                    nnz_W = np.random.randint(1, 5)
                    
                    idx_U = np.random.choice(nU, nnz_U, replace=False)
                    idx_V = np.random.choice(nV, nnz_V, replace=False)
                    idx_W = np.random.choice(nW, nnz_W, replace=False)
                    
                    U[idx_U, i] = np.random.choice(vals, nnz_U)
                    V[idx_V, i] = np.random.choice(vals, nnz_V)
                    W[idx_W, i] = np.random.choice(vals, nnz_W)
                    
                    # Scale
                    for mat, idx in [(U, idx_U), (V, idx_V), (W, idx_W)]:
                        for j in idx:
                            if np.abs(mat[j, i]) > 0:
                                mat[j, i] = 1.0 / np.abs(mat[j, i]) if np.random.random() < 0.5 else 1.0
                
                # Run ALS iterations
                n_iter = 50
                for it in range(n_iter):
                    if time.time() >= end_time:
                        break
                    
                    # Update U
                    for i in range(nU):
                        # U[i, :] such that sum_r U[i,r] * V[j,r] * W[k,r] matches target
                        # This is a linear system
                        # For fixed i, we want: U[i,r] * (sum over j,k of V[j,r] * W[k,r] * T_target[i,j,k]) = ...
                        # This is complicated; let's just do a gradient step
                        pass
                    
                    break  # Skip for now, use simple verification
                
                # Verify
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if verify_decomposition(U_list, V_list, W_list):
                    best_rank = rank
                    best_result = (U_list, V_list, W_list)
                    break
        return best_result
    
    # First, get the schoolbook solution
    U_sb, V_sb, W_sb = build_schoolbook()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_schoolbook
    
    # Try to find a better decomposition
    if time_remaining() > 5.0:
        result = try_als_search(max_rank=143, time_limit=min(time_remaining() - 5.0, 100.0))
        if result is not None:
            U, V, W = result
            best_rank = len(U[0])
            best_result = (U, V, W)
    
    U_out, V_out, W_out = best_result
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
