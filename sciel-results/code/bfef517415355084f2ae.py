"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that U @ diag @ V @ diag @ W reconstructs the MM tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    # Build tensor T of shape (n, m, p)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    
    for c in range(rank):
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        
        W_col_T = W_col.T  # shape (n, p), W_col_T[i,k] = W_col[k,i]
        
        tmp1 = np.einsum('ij,jk->ijk', U_col, V_col)  # (n, m, p)
        tmp2 = tmp1 * W_col_T[:, np.newaxis, :]  # broadcast W_col_T over j
        T_computed += tmp2
    
    # The MM tensor T[i,j,k] = 1 for all i,j,k
    T_expected = np.ones((n, m, p), dtype=np.float64)
    
    diff = np.abs(T_computed - T_expected)
    return np.max(diff) < tol


def _construct_schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W, "rank": rank}


def _strassen_2x2_2x2():
    """
    Strassen's algorithm for 2x2 * 2x2 matrix multiplication.
    Returns rank-7 decomposition.
    
    For matrices A (2x2), B (2x2), C = A*B:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B01 - B11)
    M4 = A11(B10 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    
    # M3 = A00(B01 - B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1  # -B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = A11(B10 - B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1  # -B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1  # -C00
    W[1, 4] = 1  # C01
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # -A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # -A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    
    return {"U": U, "V": V, "W": W, "rank": rank}


def _try_newton(rank, n, m, p, max_iter=200, tol=1e-10, deadline=None, n_trials=10):
    """Try to find a rank-r decomposition via Newton's method on the polynomial system."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    best_result = None
    best_err = float('inf')
    
    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        rng = np.random.RandomState(trial * 1000 + 42)
        
        U = rng.randn(n_m, rank) * 0.1
        V = rng.randn(m_p, rank) * 0.1
        W = rng.randn(p_n, rank) * 0.1
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            S = np.zeros((n, m, p), dtype=np.float64)
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                tmp = np.einsum('ij,jk->ijk', Uc, Vc)
                S += tmp * Wc.T[:, np.newaxis, :]
            
            R = S - 1.0
            
            err = np.max(np.abs(R))
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < tol:
                break
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                Wc_T = Wc.T
                
                step1 = R * Wc_T[:, np.newaxis, :]
                dU_c = np.einsum('ijk,jk->ij', step1, Vc)
                dU[:, c] = dU_c.reshape(n_m)
                
                step2 = R * Wc_T[:, np.newaxis, :]
                dV_c = np.einsum('ijk,ij->jk', step2, Uc)
                dV[:, c] = dV_c.reshape(m_p)
                
                step3 = R * Vc[np.newaxis, :, :]
                dW_ki = np.einsum('ijk,ij->ik', step3, Uc)
                dW[:, c] = dW_ki.T.reshape(p_n)
            
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_err < tol:
            break
    
    if best_result is not None and best_err < tol:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list, "rank": rank, "err": best_err}
    
    return None


def _combine_decompositions(decomp1, decomp2, n, m, p):
    """
    Combine two decompositions by block concatenation.
    decomp1 and decomp2 must have the same dimensions (n, m, p).
    """
    rank1 = decomp1["rank"]
    rank2 = decomp2["rank"]
    total_rank = rank1 + rank2
    
    U1 = np.array(decomp1["U"], dtype=np.float64)
    V1 = np.array(decomp1["V"], dtype=np.float64)
    W1 = np.array(decomp1["W"], dtype=np.float64)
    
    U2 = np.array(decomp2["U"], dtype=np.float64)
    V2 = np.array(decomp2["V"], dtype=np.float64)
    W2 = np.array(decomp2["W"], dtype=np.float64)
    
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
        "rank": total_rank
    }


def _try_block_decomposition(n, m, p, deadline=None):
    """
    Try to construct a decomposition using block structure.
    For 4x4 * 4x5, we can try:
    - 2x2 blocks: (2x2 * 2x2) x 2 blocks in each dimension
    - Use Strassen for each 2x2*2x2 block
    """
    # For 4x4 * 4x5:
    # Split into 2x2 blocks
    # A is 4x4: 4 blocks of 2x2
    # B is 4x5: 4x2 blocks of 2x2 plus one 4x1 column
    
    # This is complex. Let's try a simpler approach:
    # Use schoolbook for the 4x4 * 4x4 part (rank 64)
    # and add 4 more terms for the 5th column (rank 4)
    # Total rank: 68
    
    # Actually, let's try to be smarter.
    # The 4x4 * 4x5 multiplication can be seen as:
    # C[:, 0:4] = A * B[:, 0:4]  (4x4 * 4x4)
    # C[:, 4] = A * B[:, 4]      (4x4 * 4x1)
    
    # For the 4x4 * 4x4 part, we can use Strassen-based decomposition
    # For the 4x4 * 4x1 part, we need 4 multiplications (schoolbook)
    
    # Let's try to use a known decomposition for 4x4 * 4x4
    # The best known rank for 4x4 * 4x4 is 49 (using Strassen recursively)
    # But implementing this is complex.
    
    # For now, return None and rely on Newton's method
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Leave 20s margin
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = schoolbook["rank"]
    
    # Try Newton's method for various ranks
    # Theoretical lower bound: n*m*p = 80 for schoolbook
    # Strassen for 4x4*4x4 would give rank 49, but we have 4x4*4x5
    # Let's try ranks from 70 down to 50
    
    for rank in range(70, 45, -1):
        if time.time() > deadline:
            break
        
        result = _try_newton(rank, n, m, p, max_iter=100, tol=1e-10, deadline=deadline, n_trials=5)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
                best_result = result
                best_rank = rank
                break
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END
