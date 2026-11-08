import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_terms():
    """Return 7 rank-1 Strassen terms for 2x2x2 matmul."""
    return [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]},
        {'U': [1, 0, 0, 0], 'V': [0, 1, -1, 0], 'W': [0, 0, 1, 1]},
        {'U': [0, 0, 0, 1], 'V': [0, -1, 1, 0], 'W': [1, 1, 0, 0]},
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]},
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]},
        {'U': [0, 1, -1, 0], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]},
    ]

def _build_kronecker_decomposition():
    """Build the 4x4x8 decomposition with rank 98 using Kronecker structure."""
    s222 = _strassen_2x2_terms()
    n, m, p = 4, 4, 8
    rank_total = 98
    
    U = np.zeros((n * m, rank_total), dtype=np.int64)
    V = np.zeros((m * p, rank_total), dtype=np.int64)
    W = np.zeros((p * n, rank_total), dtype=np.int64)
    
    U_big = np.array([t['U'] for t in s222], dtype=np.int64)
    V_big = np.array([t['V'] for t in s222], dtype=np.int64)
    W_big = np.array([t['W'] for t in s222], dtype=np.int64)
    
    U_small = U_big
    V_small = V_big
    W_small = W_big
    
    for g in range(2):
        for s1 in range(7):
            for s2 in range(7):
                r_idx = g * 49 + s1 * 7 + s2
                
                Ub_s1 = U_big[s1].reshape(2, 2)
                Us_s2 = U_small[s2].reshape(2, 2)
                U_prod = Ub_s1[:, None, :, None] * Us_s2[None, :, None, :]
                U_prod_reordered = U_prod.transpose(0, 2, 1, 3)
                U_mat = U_prod_reordered.reshape(4, 4)
                U[:, r_idx] = U_mat.flatten()
                
                Vb_s1 = V_big[s1].reshape(2, 2)
                Vs_s2 = V_small[s2].reshape(2, 2)
                V_prod = Vb_s1[:, None, :, None] * Vs_s2[None, :, None, :]
                V_prod_reordered = V_prod.transpose(0, 2, 1, 3)
                V_mat = V_prod_reordered.reshape(4, 4)
                
                for jj in range(4):
                    for kk_local in range(4):
                        kk_global = 4 * g + kk_local
                        V[jj * 8 + kk_global, r_idx] = V_mat[jj, kk_local]
                        
                Wb_s1 = W_big[s1].reshape(2, 2)
                Ws_s2 = W_small[s2].reshape(2, 2)
                W_prod = Wb_s1[:, None, :, None] * Ws_s2[None, :, None, :]
                W_prod_reordered = W_prod.transpose(0, 2, 1, 3)
                W_mat = W_prod_reordered.reshape(4, 4)
                
                for kk_local in range(4):
                    kk_global = 4 * g + kk_local
                    for ii in range(4):
                        W[kk_global * 4 + ii, r_idx] = W_mat[kk_local, ii]
                        
    return U, V, W, rank_total

def _verify_decomposition(U, V, W, rank, n=4, m=4, p=8, atol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    np.random.seed(123)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m).astype(float)
        Vr = V[:, r].reshape(m, p).astype(float)
        Wr = W[:, r].reshape(p, n).astype(float)
        a_part = np.sum(Ur * A_test, axis=1)
        b_part = np.sum(Vr * B_test, axis=0)
        C_computed += np.outer(a_part, b_part) * Wr.T
        
    if not np.allclose(C_computed, C_expected, atol=atol):
        return False
    return True

def _build_hybrid_decomposition():
    """
    Structural change: Use a hybrid approach combining Strassen on the row-block 
    structure with a more efficient column handling.
    
    Key insight: Instead of full Kronecker (7x7=49 per g-block, total 98),
    we use a 2x2 Strassen on the row dimension (4=2x2) and a direct 
    2x4 decomposition on the column dimension, but with shared structure.
    
    This gives rank 7 * 8 = 56 if we can find an 8-term decomposition for 2x2 x 2x4.
    Actually, 2x2 x 2x4 matmul has tensor size 2*2*2*4=32, and schoolbook is 2*2*4=16.
    We can potentially do better than 16.
    
    Let's try: rank 7 for row Strassen, and for each Strassen term, we need to 
    multiply a 2x2 matrix by a 2x4 matrix. The minimal rank for 2x2 x 2x4 is 
    at most 2*2*4=16 (schoolbook), but might be lower.
    
    Actually, let's use a different structural approach: treat the problem as 
    two 4x4 x 4x4 problems (for the two 4x4 output blocks) and use the 7x7 
    Kronecker for each, but then try to merge/cancel terms.
    
    For now, let's implement a rank-92 decomposition by using a slightly 
    different block structure.
    """
    n, m, p = 4, 4, 8
    # Start with the Kronecker decomposition (rank 98)
    U, V, W, rank = _build_kronecker_decomposition()
    
    # Try to reduce rank by identifying linearly dependent terms
    # or by using a different factorization
    
    # Alternative: Use a 4x4 x 4x4 Strassen-like decomposition (rank 49)
    # for the first 4x4 output block, and a separate one for the second,
    # but with some shared terms.
    
    # For now, return the Kronecker decomposition as baseline
    return U, V, W, rank

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 x 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, try the Kronecker decomposition (rank 98)
    try:
        U, V, W, rank = _build_kronecker_decomposition()
        if time.time() < deadline:
            if _verify_decomposition(U, V, W, rank):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 128)
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.int64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.int64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
