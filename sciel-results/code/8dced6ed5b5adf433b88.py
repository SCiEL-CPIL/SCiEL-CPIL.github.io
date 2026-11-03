import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    
    # Try to construct a rank-132 decomposition using Strassen.
    # If it fails verification, fall back to schoolbook (rank 144).
    
    try:
        result = _strassen_132(n, m, p)
        # Verify the decomposition
        if _verify(result, n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _strassen_132(n, m, p):
    """Construct a rank-132 decomposition using Strassen for first 6 cols + schoolbook for last 3."""
    from collections import defaultdict
    
    W_accum = defaultdict(float)
    
    def add_contrib(A_i_b, A_i_l, A_k_b, A_k_l, B_k_b, B_k_l, B_j_b, B_j_l, C_i_b, C_i_l, C_j_b, C_j_l, coeff):
        u_idx = (A_i_b * 2 + A_i_l) * 4 + (A_k_b * 2 + A_k_l)
        v_idx = (B_k_b * 2 + B_k_l) * 9 + (B_j_b * 3 + B_j_l)
        w_idx = (C_j_b * 3 + C_j_l) * 4 + (C_i_b * 2 + C_i_l)
        W_accum[(u_idx, v_idx, w_idx)] += coeff
    
    # M1 = (A11+A22)(B11+B21) -> C11 += M1
    for i_l in range(2):
        for k_l in range(2):
            for j_l in range(3):
                add_contrib(0, i_l, 0, k_l, 0, k_l, 0, j_l, 0, i_l, 0, j_l, 1)
                add_contrib(0, i_l, 0, k_l, 1, k_l, 0, j_l, 0, i_l, 0, j_l, 1)
                add_contrib(1, i_l, 1, k_l, 0, k_l, 0, j_l, 0, i_l, 0, j_l, 1)
                add_contrib(1, i_l, 1, k_l, 1, k_l, 0, j_l, 0, i_l, 0, j_l, 1)
    
    # M2 = (A21+A22)B11 -> C21 += M2
    for i_l in range(2):
        for k_l in range(2):
            for j_l in range(3):
                add_contrib(1, i_l, 0, k_l, 0, k_l, 0, j_l, 1, i_l, 0, j_l, 1)
                add_contrib(1, i_l, 1, k_l, 0, k_l, 0, j_l, 1, i_l, 0, j_l, 1)
    
    # M3 = A11(B12-B22) -> C12 += M3
    for i_l in range(2):
        for k_l in range(2):
            for j_l in range(3):
                add_contrib(0, i_l, 0, k_l, 0, k_l, 1, j_l, 0, i_l, 1, j_l, 1)
                add_contrib(0, i_l, 0, k_l, 1, k_l, 1, j_l, 0, i_l, 1, j_l, -1)
    
    # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
    for i_l in range(2):
        for k_l in range(2):
            for j_l in range(3):
                add_contrib(1, i_l, 1, k_l, 1, k_l, 0, j_l, 0, i_l, 0, j_l, 1)
                add_contrib(1, i_l, 1, k_l, 0, k_l, 0, j_l, 0, i_l, 0, j_l, -1)
                add_contrib(1, i_l, 1, k_l, 1, k_l, 0, j_l, 1, i_l, 0, j_l, 1)
                add_contrib(1, i_l, 1, k_l, 0, k_l, 0, j_l, 1, i_l, 0, j_l, -1)
    
    # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    for i_l in range(2):
        for k_l in range(2):
            for j_l in range(3):
                add_contrib(0, i_l, 0, k_l, 1, k_l, 1, j_l, 0, i_l, 0, j_l, -1)
                add_contrib(0, i_l, 1, k_l, 1, k_l, 1, j_l, 0, i_l, 0, j_l, -1)
                add_contrib(0, i_l, 0, k_l, 1, k_l, 1, j_l, 0, i_l, 1, j_l, 1)
                add_contrib(0, i_l, 1, k_l, 1, k_l, 1, j_l, 0, i_l, 1, j_l, 1)
    
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    for i_l in range(2):
        for k_l in range(2):
            for j_l in range(3):
                add_contrib(1, i_l, 0, k_l, 0, k_l, 0, j_l, 1, i_l, 1, j_l, 1)
                add_contrib(1, i_l, 0, k_l, 0, k_l, 1, j_l, 1, i_l, 1, j_l, 1)
                add_contrib(0, i_l, 0, k_l, 0, k_l, 0, j_l, 1, i_l, 1, j_l, -1)
                add_contrib(0, i_l, 0, k_l, 0, k_l, 1, j_l, 1, i_l, 1, j_l, -1)
    
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    for i_l in range(2):
        for k_l in range(2):
            for j_l in range(3):
                add_contrib(0, i_l, 1, k_l, 1, k_l, 0, j_l, 0, i_l, 0, j_l, 1)
                add_contrib(0, i_l, 1, k_l, 1, k_l, 1, j_l, 0, i_l, 0, j_l, 1)
                add_contrib(1, i_l, 1, k_l, 1, k_l, 0, j_l, 0, i_l, 0, j_l, -1)
                add_contrib(1, i_l, 1, k_l, 1, k_l, 1, j_l, 0, i_l, 0, j_l, -1)
    
    # S1 = A11*B13 -> C13 += S1
    for i_l in range(2):
        for k_l in range(2):
            for j_l in range(3):
                add_contrib(0, i_l, 0, k_l, 0, k_l, 2, j_l, 0, i_l, 2, j_l, 1)
    
    # S2 = A12*B23 -> C13 += S2
    for i_l in range(2):
        for k_l in range(2):
            for j_l in range(3):
                add_contrib(0, i_l, 1, k_l, 1, k_l, 2, j_l, 0, i_l, 2, j_l, 1)
    
    # S3 = A21*B13 -> C23 += S3
    for i_l in range(2):
        for k_l in range(2):
            for j_l in range(3):
                add_contrib(1, i_l, 0, k_l, 0, k_l, 2, j_l, 1, i_l, 2, j_l, 1)
    
    # S4 = A22*B23 -> C23 += S4
    for i_l in range(2):
        for k_l in range(2):
            for j_l in range(3):
                add_contrib(1, i_l, 1, k_l, 1, k_l, 2, j_l, 1, i_l, 2, j_l, 1)
    
    # Build U, V, W
    unique_terms = [(u, v, w, c) for (u, v, w), c in W_accum.items() if abs(c) > 1e-10]
    rank = len(unique_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_idx, v_idx, w_idx, coeff) in enumerate(unique_terms):
        U[u_idx][r] = 1
        V[v_idx][r] = 1
        W[w_idx][r] = coeff
    
    return {"U": U, "V": V, "W": W}

def _verify(result, n, m, p):
    """Verify the decomposition by checking a few random matrices."""
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    rank = U.shape[1]
    
    # Check dimensions
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Test with random matrices
    np.random.seed(42)
    for _ in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            
            # u_vec has shape (n*m,), reshape to (n, m)
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            # Contribution: w_mat.T * (u_mat @ A) * v_mat? No...
            # The decomposition is: C[i,j] = sum_r W[j*n+i, r] * (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j])
            # This doesn't seem right.
            
            # Let me try: C[i,j] = sum_r W[j*n+i, r] * (U[:, r].dot(A_flat)) * (V[:, r].dot(B_flat))
            # Where A_flat = A.flatten() and B_flat = B.flatten()
            
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            u_dot = np.dot(u_vec, A_flat)
            v_dot = np.dot(v_vec, B_flat)
            
            for i in range(n):
                for j in range(p):
                    C_computed[i, j] += W[j * n + i, r] * u_dot * v_dot
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

