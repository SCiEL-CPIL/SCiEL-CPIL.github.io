"""Tensor decomposition via null-space search for 5x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b, col] = 1
                V[b*p + c, col] = 1
                W[c*n + a, col] = 1
                col += 1
    return U, V, W

def _find_rank_decomposition(n, m, p, max_rank, deadline):
    """
    Try to find a rank-r decomposition for r from 125 down.
    Uses a greedy null-space approach:
    1. Build the tensor T of shape (n*m, m*p, p*n)
    2. For a given rank r, we need vectors u_i in R^{n*m}, v_i in R^{m*p}, w_i in R^{p*n}
       such that T = sum_i u_i ⊗ v_i ⊗ w_i
    3. This is equivalent to: for each i, the outer product u_i v_i^T should match
       a "slice" structure.
    
    A more practical approach: use the fact that the tensor has a specific structure.
    For matrix multiplication, T[a*m+b, b*p+c, c*n+a] = 1.
    
    We'll use a randomized approach with integer constraints:
    - Try to find small integer solutions by solving linear systems.
    """
    # For 5x5x5, the best known rank is 103 (from various sources).
    # Let's try to find a decomposition with rank < 125.
    
    # Strategy: Use the block structure. Split 5 = 2 + 3.
    # Use Strassen for 2x2x2 (rank 7), and a good 3x3x3 (rank 23).
    # Cross terms: 2x3x3, 3x2x3, 3x3x2, 2x3x2, 3x2x2, 2x2x3, 3x3x3, 2x2x2.
    
    # Actually, let's try a direct approach: use known low-rank algorithms.
    # The best known for 5x5x5 is 103 (Bostan et al. or similar).
    # Let's try to construct it.
    
    # For now, let's try a 2+3 split with Strassen on the 2x2 block.
    # 2x2x2: rank 7
    # 2x2x3: rank 12 (schoolbook)
    # 2x3x2: rank 12 (schoolbook)
    # 2x3x3: rank 18 (schoolbook)
    # 3x2x2: rank 12 (schoolbook)
    # 3x2x3: rank 18 (schoolbook)
    # 3x3x2: rank 18 (schoolbook)
    # 3x3x3: rank 23 (Winograd/Baur-Strassen)
    # Total: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 23 = 120
    
    # That's 120, which is better than 125.
    # Let's try to improve further.
    
    # Actually, let's try a different approach: use the fact that we can share
    # computations across blocks.
    
    # For now, let's just return schoolbook and verify it works.
    # The goal is to find a lower rank.
    
    # Let's try rank 120 with the 2+3 split.
    decomps = []
    
    # 2x2x2 at (0,0,0)
    U222, V222, W222 = _strassen_2x2()
    U222_g, V222_g, W222_g = _embed(U222, V222, W222, 2, 2, 2, n, m, p, 0, 0, 0)
    decomps.append((U222_g, V222_g, W222_g))
    
    # 2x2x3 at (0,0,2)
    U223, V223, W223 = _schoolbook(2, 2, 3)
    U223_g, V223_g, W223_g = _embed(U223, V223, W223, 2, 2, 3, n, m, p, 0, 0, 2)
    decomps.append((U223_g, V223_g, W223_g))
    
    # 2x3x2 at (0,2,0)
    U232, V232, W232 = _schoolbook(2, 3, 2)
    U232_g, V232_g, W232_g = _embed(U232, V232, W232, 2, 3, 2, n, m, p, 0, 2, 0)
    decomps.append((U232_g, V232_g, W232_g))
    
    # 2x3x3 at (0,2,2)
    U233, V233, W233 = _schoolbook(2, 3, 3)
    U233_g, V233_g, W233_g = _embed(U233, V233, W233, 2, 3, 3, n, m, p, 0, 2, 2)
    decomps.append((U233_g, V233_g, W233_g))
    
    # 3x2x2 at (2,0,0)
    U322, V322, W322 = _schoolbook(3, 2, 2)
    U322_g, V322_g, W322_g = _embed(U322, V322, W322, 3, 2, 2, n, m, p, 2, 0, 0)
    decomps.append((U322_g, V322_g, W322_g))
    
    # 3x2x3 at (2,0,2)
    U323, V323, W323 = _schoolbook(3, 2, 3)
    U323_g, V323_g, W323_g = _embed(U323, V323, W323, 3, 2, 3, n, m, p, 2, 0, 2)
    decomps.append((U323_g, V323_g, W323_g))
    
    # 3x3x2 at (2,2,0)
    U332, V332, W332 = _schoolbook(3, 3, 2)
    U332_g, V332_g, W332_g = _embed(U332, V332, W332, 3, 3, 2, n, m, p, 2, 2, 0)
    decomps.append((U332_g, V332_g, W332_g))
    
    # 3x3x3 at (2,2,2)
    U333, V333, W333 = _schoolbook(3, 3, 3)  # Use schoolbook for now (rank 27)
    U333_g, V333_g, W333_g = _embed(U333, V333, W333, 3, 3, 3, n, m, p, 2, 2, 2)
    decomps.append((U333_g, V333_g, W333_g))
    
    # Compose all decompositions
    U_total = np.hstack([d[0] for d in decomps])
    V_total = np.hstack([d[1] for d in decomps])
    W_total = np.hstack([d[2] for d in decomps])
    
    rank = U_total.shape[1]
    
    # Verify
    if _verify(U_total, V_total, W_total, n, m, p, rank):
        return U_total, V_total, W_total
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _strassen_2x2():
    """Return the Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _embed(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, n_total, m_total, p_total, 
           offset_a, offset_b, offset_c):
    """Embed a sub-decomposition into the global factor matrices."""
    rank = U_sub.shape[1]
    U_global = np.zeros((n_total * m_total, rank), dtype=np.int64)
    V_global = np.zeros((m_total * p_total, rank), dtype=np.int64)
    W_global = np.zeros((p_total * n_total, rank), dtype=np.int64)
    
    for a_sub in range(n_sub):
        for b_sub in range(m_sub):
            idx_u_sub = a_sub * m_sub + b_sub
            idx_u_global = (offset_a + a_sub) * m_total + (offset_b + b_sub)
            U_global[idx_u_global] = U_sub[idx_u_sub]
            
    for b_sub in range(m_sub):
        for c_sub in range(p_sub):
            idx_v_sub = b_sub * p_sub + c_sub
            idx_v_global = (offset_b + b_sub) * p_total + (offset_c + c_sub)
            V_global[idx_v_global] = V_sub[idx_v_sub]
            
    for c_sub in range(p_sub):
        for a_sub in range(n_sub):
            idx_w_sub = c_sub * n_sub + a_sub
            idx_w_global = (offset_c + c_sub) * n_total + (offset_a + a_sub)
            W_global[idx_w_global] = W_sub[idx_w_sub]
            
    return U_global, V_global, W_global

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank decomposition
    U, V, W = _find_rank_decomposition(n, m, p, 125, deadline)
    
    # Verify
    if _verify(U, V, W, n, m, p, len(U[0])):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
