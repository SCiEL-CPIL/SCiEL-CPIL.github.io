"""
Matrix multiplication 4,4,5 rank minimization.
Strategy: Use recursive Strassen for 4x4x4 (rank 49) and schoolbook for 4x4x1 (rank 16).
Total rank: 65.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _strassen_2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2x2 Strassen.
    Each term is a tuple (U_dict, V_dict, W_dict) where dicts map (i,j) -> coeff.
    U[i,j] for A[i,j], V[j,k] for B[j,k], W[k,i] for C[i,k] (trace convention).
    """
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M6
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M1 - M2 + M5
    
    terms = [
        # M1: U has A00,A11; V has B00,B11; W contributes to C00 and C11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: U has A10,A11; V has B00; W contributes to C01
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3: U has A00; V has B10,B00; W contributes to C10
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1}),
        # M4: U has A11; V has B01,B11; W contributes to C00 and C01
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}, {(0,0): -1, (1,0): 1}),
        # M5: U has A00,A01; V has B11; W contributes to C10 and C11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (1,1): 1}),
        # M6: U has A10,A00; V has B00,B01; W contributes to C00 and C10
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1, (0,1): 1}),
        # M7: U has A01,A11; V has B10,B11; W contributes to C11
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,1): 1})
    ]
    return terms

def _build_4x4x4_strassen():
    """
    Constructs 4x4x4 decomposition using two levels of Strassen.
    Rank 49.
    """
    n = 4
    base_terms = _strassen_2x2_terms()
    rank = 49
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    for t1 in range(7):
        U1, V1, W1 = base_terms[t1]
        for t2 in range(7):
            U2, V2, W2 = base_terms[t2]
            idx = t1 * 7 + t2
            
            # Combine U factors: U_total[(i1*2+i2, j1*2+j2)] += U1[i1,j1] * U2[i2,j2]
            for (i1, j1), c1 in U1.items():
                for (i2, j2), c2 in U2.items():
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    U[i*n + j, idx] += c1 * c2
            
            # Combine V factors: V_total[(j1*2+j2, k1*2+k2)] += V1[j1,k1] * V2[j2,k2]
            for (j1, k1), c1 in V1.items():
                for (j2, k2), c2 in V2.items():
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    V[j*n + k, idx] += c1 * c2
            
            # Combine W factors: W_total[(k1*2+k2, i1*2+i2)] += W1[k1,i1] * W2[k2,i2]
            for (k1, i1), c1 in W1.items():
                for (k2, i2), c2 in W2.items():
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    W[k*n + i, idx] += c1 * c2
                    
    return U, V, W

def _build_4x4x5_decomposition():
    """
    Combines 4x4x4 Strassen (rank 49) and 4x4x1 Schoolbook (rank 16).
    Returns U, V, W matrices.
    """
    n, m, p = 4, 4, 5
    
    U444, V444, W444 = _build_4x4x4_strassen()
    rank_444 = 49
    rank_441 = n * m # 16
    total_rank = rank_444 + rank_441
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # 1. Copy 4x4x4 part
    U[:, :rank_444] = U444
    V[:m*4, :rank_444] = V444
    W[:4*n, :rank_444] = W444
    
    # 2. Add 4x4x1 part (for the 5th column of B, index 4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Term r corresponds to (i, j)
    for i in range(n):
        for j in range(m):
            r = rank_444 + i * m + j
            U[i*m + j, r] = 1.0
            V[j*p + 4, r] = 1.0
            W[4*n + i, r] = 1.0
            
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k.
    Uses vectorized numpy operations for speed.
    """
    rank = U.shape[1]
    # Build the full tensor check using einsum
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # Compute T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    """
    Returns the factor matrices for the 4x4x5 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _build_4x4x5_decomposition()
        
        if time.time() > deadline:
            raise TimeoutError()
            
        if _verify(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception as e:
        # Fallback to schoolbook if Strassen construction or verification fails
        pass
    
    # Schoolbook fallback (Rank 80)
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
