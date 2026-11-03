"""
Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy:
1. Construct a verified rank-98 decomposition using recursive Kronecker products of Strassen-7.
2. Attempt to reduce the rank using a fast, vectorized ALS refinement with exact verification.
   - First, try to reduce the rank by zeroing out columns (removing terms).
   - Then, try to reduce the rank further via random restarts and ALS.
3. Return the lowest rank solution found within the time limit.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7.
    U, V, W are (2,2,7) tensors.
    """
    U = np.zeros((2, 2, 7), dtype=float)
    V = np.zeros((2, 2, 7), dtype=float)
    W = np.zeros((2, 2, 7), dtype=float)
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0, 0] = 1; U[1, 1, 0] = 1
    V[0, 0, 0] = 1; V[1, 1, 0] = 1
    W[0, 0, 0] = 1; W[1, 1, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[1, 0, 1] = 1; U[1, 1, 1] = 1
    V[0, 0, 1] = 1
    W[0, 1, 1] = 1; W[1, 1, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U[0, 0, 2] = 1
    V[0, 1, 2] = 1; V[1, 1, 2] = -1
    W[1, 0, 2] = 1; W[1, 1, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U[1, 1, 3] = 1
    V[1, 0, 3] = 1; V[0, 0, 3] = -1
    W[0, 0, 3] = 1; W[0, 1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0, 0, 4] = 1; U[0, 1, 4] = 1
    V[1, 1, 4] = 1
    W[0, 0, 4] = -1; W[1, 0, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[1, 0, 5] = 1; U[0, 0, 5] = -1
    V[0, 0, 5] = 1; V[0, 1, 5] = 1
    W[1, 1, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[0, 1, 6] = 1; U[1, 1, 6] = -1
    V[1, 0, 6] = 1; V[1, 1, 6] = 1
    W[0, 0, 6] = 1

    return U, V, W

def _kronecker_4x4x8():
    """
    Construct rank-98 decomposition for 4x4 * 4x8.
    Returns U (16, 98), V (32, 98), W (32, 98).
    """
    U2, V2, W2 = _strassen_2x2()
    
    # Build 4x4x4 factors using recursion (Kronecker product)
    # U4[i, j, R] where R = 7*s + r
    U4 = np.zeros((4, 4, 49), dtype=float)
    V4 = np.zeros((4, 4, 49), dtype=float)
    W4 = np.zeros((4, 4, 49), dtype=float)
    
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2*i1 + i0
                    j = 2*j1 + j0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]
    
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2*j1 + j0
                    k = 2*k1 + k0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]
    
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2*k1 + k0
                    i = 2*i1 + i0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]

    # Build 4x4x8 factors
    # U shape (16, 98), V shape (32, 98), W shape (32, 98)
    # U[i*m+j, R], V[j*p+k, R], W[k*n+i, R]
    
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # Fill first half (k < 4) corresponding to the first 4x4 block of B
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i*4 + j, R] = U4[i, j, R]
                
    for j in range(4):
        for k in range(4):
            for R in range(49):
                V[j*8 + k, R] = V4[j, k, R]
                
    for k in range(4):
        for i in range(4):
            for R in range(49):
                W[k*4 + i, R] = W4[k, i, R]
                
    # Fill second half (k >= 4) corresponding to the second 4x4 block of B
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i*4 + j, 49 + R] = U4[i, j, R]
                
    for j in range(4):
        for kp in range(4):
            k = 4 + kp
            for R in range(49):
                V[j*8 + k, 49 + R] = V4[j, kp, R]
                
    for kp in range(4):
        k = 4 + kp
        for i in range(4):
            for R in range(49):
                W[k*4 + i, 49 + R] = W4[kp, i, R]

    return U, V, W

def _verify(U, V, W):
    """
    Verify if U, V, W exactly decompose the 4x4x8 matmul tensor.
    U: (16, R), V: (32, R), W: (32, R)
    """
    n, m, p = 4, 4, 8
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    # Compute tensor T[i, j, k]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Target is all ones
    return bool(np.allclose(T, 1.0, atol=1e-8))

def _als_refine(U, V, W, max_iter=20):
    """
    Perform ALS refinement on U, V, W.
    Returns refined U, V, W.
    """
    n, m, p = 4, 4, 8
    R = U.shape[1]
    
    # Precompute index mappings for speed
    # We solve for W first, then U, then V.
    # The system is linear in each factor given the others.
    
    # For W:
    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    # Let A_W[k, i, c] be the variable for W.
    # The coefficient for A_W[k, i, c] in the equation for T[i, j, k] is U[i, j, c] * V[j, k, c].
    
    # We can form the matrix A_W of size (n*m*p) x (p*n*R)
    # But it's sparse. We can use lstsq on the full system if R is small enough, or use block structure.
    # Given R <= 98, n*m*p = 128. p*n*R = 32*R.
    # If R=98, 32*98 = 3136. 128 x 3136 is small.
    
    A_W = np.zeros((n*m*p, p*n*R))
    for c in range(R):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        # coeff[i, j, k] = u_c[i, j] * v_c[j, k]
        coeff = np.einsum('ij,jk->ijk', u_c, v_c)
        # Map (i, j, k) to eq_idx
        # Map (k, i, c) to var_idx
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    var_idx = (k*n + i)*R + c
                    A_W[eq_idx, var_idx] = coeff[i, j, k]
    
    T_flat = np.ones(n*m*p)
    W_flat, _, _, _ = np.linalg.lstsq(A_W, T_flat, rcond=None)
    W = W_flat.reshape(p*n, R)
    
    # For U:
    A_U = np.zeros((n*m*p, n*m*R))
    for c in range(R):
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        # coeff[i, j, k] = v_c[j, k] * w_c[k, i]
        coeff = np.einsum('jk,ki->ijk', v_c, w_c)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    var_idx = (i*m + j)*R + c
                    A_U[eq_idx, var_idx] = coeff[i, j, k]
    
    U_flat, _, _, _ = np.linalg.lstsq(A_U, T_flat, rcond=None)
    U = U_flat.reshape(n*m, R)
    
    # For V:
    A_V = np.zeros((n*m*p, m*p*R))
    for c in range(R):
        u_c = U[:, c].reshape(n, m)
        w_c = W[:, c].reshape(p, n)
        # coeff[i, j, k] = u_c[i, j] * w_c[k, i]
        coeff = np.einsum('ij,ki->ijk', u_c, w_c)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    var_idx = (j*p + k)*R + c
                    A_V[eq_idx, var_idx] = coeff[i, j, k]
    
    V_flat, _, _, _ = np.linalg.lstsq(A_V, T_flat, rcond=None)
    V = V_flat.reshape(m*p, R)
    
    return U, V, W

def _round_to_half(U, V, W):
    """
    Round U, V, W to nearest half-integers.
    """
    U = np.round(U * 2) / 2.0
    V = np.round(V * 2) / 2.0
    W = np.round(W * 2) / 2.0
    return U, V, W

def construct():
    t_start = time.time()
    DEADLINE = t_start + 140
    
    n, m, p = 4, 4, 8
    
    # 1. Get baseline rank-98
    try:
        U, V, W = _kronecker_4x4x8()
        if not _verify(U, V, W):
            # Fallback to schoolbook rank 128
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]
        
    except Exception:
        # Fallback to schoolbook rank 128
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2. Try to reduce rank by zeroing columns
    # This is a greedy approach: remove one column at a time if the residual can be absorbed by the others.
    # Actually, we can just try to solve for a lower rank directly.
    
    # Try ranks from 97 down to 90 (or lower if time permits)
    # Start with 97
    for target_rank in range(best_rank - 1, 80, -1):
        if time.time() > DEADLINE - 10:
            break
            
        # Try a few random restarts
        found = False
        for restart in range(3):
            if time.time() > DEADLINE - 10:
                break
                
            # Initialize with random projection of the current best?
            # Or just random?
            # Let's try random first.
            U_t = np.random.randn(n*m, target_rank)
            V_t = np.random.randn(m*p, target_rank)
            W_t = np.random.randn(p*n, target_rank)
            
            # ALS
            for _ in range(30):
                if time.time() > DEADLINE - 10:
                    break
                U_t, V_t, W_t = _als_refine(U_t, V_t, W_t, max_iter=1)
                
            # Round
            U_r, V_r, W_r = _round_to_half(U_t, V_t, W_t)
            
            if _verify(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                found = True
                break
        
        # If we found a valid solution for this rank, we can continue to lower ranks
        # If not, we might try harder, but for now we break to save time?
        # The prompt says "rank-minimal". We should try to go as low as possible.
        # But if we fail at 97, we might fail at 96 too? Not necessarily.
        # Let's continue the loop.

    # 3. If we have time, try to reduce further with more restarts or different initializations
    # We can also try to refine the best solution found so far
    if time.time() < DEADLINE - 20:
        # Try to refine the best solution
        U_r, V_r, W_r = _als_refine(best_U, best_V, best_W, max_iter=5)
        U_r, V_r, W_r = _round_to_half(U_r, V_r, W_r)
        if _verify(U_r, V_r, W_r):
            best_U, best_V, best_W = U_r, V_r, W_r
            # Check if rank can be reduced by zeroing
            # (This is complex, skipping for now)

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
