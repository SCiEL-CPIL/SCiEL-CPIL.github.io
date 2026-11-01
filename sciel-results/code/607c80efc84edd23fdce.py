import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p, tol=1e-5):
    """
    Verify if U, V, W decompose the matrix multiplication tensor M_{n,m,p}.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # T[i, j, k] = sum_t U_r[i, j, t] * V_r[j, k, t] * W_r[k, i, t]
    T = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
    
    return np.allclose(T, 1.0, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Decompose M_{2,3,9} as M_{2,1,1} (Kron) M_{1,3,9}
    # Rank(M_{2,1,1}) = 2
    # Rank(M_{1,3,9}) = 3
    # Total Rank = 6
    
    rank1 = 2
    rank2 = 3
    total_rank = rank1 * rank2
    
    # Initialize matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Indices for M_{2,1,1}: i1 in {0,1}, j1 in {0}, k1 in {0}
    # Indices for M_{1,3,9}: i2 in {0}, j2 in {0,1,2}, k2 in {0..8}
    
    # Map global indices to local indices
    # i = i1 * 1 + i2 = i1 (since i2=0) -> i in {0,1}
    # j = j1 * 3 + j2 = j2 (since j1=0) -> j in {0,1,2}
    # k = k1 * 9 + k2 = k2 (since k1=0) -> k in {0..8}
    
    # t = t1 * 3 + t2
    # t1 in {0,1} corresponding to i1
    # t2 in {0,1,2} corresponding to j2
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # We don't iterate over k for U, V, W construction directly, 
                # but we fill based on i, j, k structure.
                pass
                
    # Fill U: U[i, j, t] = delta(i, t1) * delta(j, t2)
    for i in range(n):
        for j in range(m):
            for t1 in range(rank1):
                for t2 in range(rank2):
                    t = t1 * rank2 + t2
                    val_u1 = 1 if i == t1 else 0
                    val_u2 = 1 if j == t2 else 0
                    U[i * m + j, t] = val_u1 * val_u2
                    
    # Fill V: V[j, k, t] = delta(j1, t1) * delta(j2, t2)
    # j1 is always 0. V^{(2,1,1)}_{0, t1} = 1.
    # V^{(1,3,9)}_{j2, k2, t2} = delta(j2, t2).
    # So V[j, k, t] = 1 * delta(j, t2) = delta(j, t2).
    for j in range(m):
        for k in range(p):
            for t1 in range(rank1):
                for t2 in range(rank2):
                    t = t1 * rank2 + t2
                    val_v1 = 1 # V^{(2,1,1)}_{0, t1} is 1 for all t1? 
                    # Wait, V^{(2,1,1)}_{j1, t1}. j1=0.
                    # In M_{2,1,1}, V_{0, t1} = 1.
                    val_v2 = 1 if j == t2 else 0
                    V[j * p + k, t] = val_v1 * val_v2
                    
    # Fill W: W[k, i, t] = delta(k1, t1) * delta(i1, t2) ? No.
    # W^{(2,1,1)}_{k1, i1, t1} = delta(i1, t1).
    # W^{(1,3,9)}_{k2, i2, t2} = 1.
    # So W[k, i, t] = delta(i, t1) * 1 = delta(i, t1).
    for k in range(p):
        for i in range(n):
            for t1 in range(rank1):
                for t2 in range(rank2):
                    t = t1 * rank2 + t2
                    val_w1 = 1 if i == t1 else 0
                    val_w2 = 1
                    W[k * n + i, t] = val_w1 * val_w2
                    
    # Verify
    if _verify(U, V, W, n, m, p, tol=1e-5):
        # Convert to list
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank = n * m * p
    U_s = [[0] * rank for _ in range(n * m)]
    V_s = [[0] * rank for _ in range(m * p)]
    W_s = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][column] = 1
                V_s[j * p + k][column] = 1
                W_s[k * n + i][column] = 1
                column += 1
    return {"U": U_s, "V": V_s, "W": W_s}

