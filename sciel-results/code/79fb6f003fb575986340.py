import time
import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, use 120 to be safe (task says 120s)
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=2, p=15
    # U: 4 rows, V: 30 rows, W: 30 rows
    
    # Strassen-based decomposition
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) + 1 single column (14)
    # Each pair: 7 multiplications, single: 4 multiplications
    # Total rank: 7*7 + 4 = 53
    
    rank = 53
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0.0] * rank for _ in range(p * n)]  # 30 x 53
    
    col = 0
    # A entries: a11=U[0], a12=U[1], a21=U[2], a22=U[3]
    # B entries: b1(k)=V[k], b2(k)=V[15+k] for k=0..14
    # C entries: c(i,k)=W[k*2+i] for i=0,1; k=0..14
    
    for pair in range(7):
        k = pair * 2  # first column of pair
        k2 = k + 1    # second column of pair
        
        # m1 = (a11+a22) * (b1(k) + b2(k2))
        U[0][col] = 1.0
        U[3][col] = 1.0
        V[k][col] = 1.0
        V[15 + k2][col] = 1.0
        W[k * 2 + 0][col] = 1.0    # c(0,k)
        W[k2 * 2 + 1][col] = 1.0   # c(1,k2)
        col += 1
        
        # m2 = (a21+a22) * b1(k)
        U[2][col] = 1.0
        U[3][col] = 1.0
        V[k][col] = 1.0
        W[k * 2 + 1][col] = 1.0    # c(1,k)
        W[k2 * 2 + 1][col] = -1.0  # c(1,k2)
        col += 1
        
        # m3 = a11 * (b1(k2) - b2(k2))
        U[0][col] = 1.0
        V[k2][col] = 1.0
        V[15 + k2][col] = -1.0
        W[k2 * 2 + 0][col] = 1.0   # c(0,k2)
        W[k2 * 2 + 1][col] = 1.0   # c(1,k2)
        col += 1
        
        # m4 = a22 * (b2(k) - b1(k))
        U[3][col] = 1.0
        V[15 + k][col] = 1.0
        V[k][col] = -1.0
        W[k * 2 + 0][col] = 1.0    # c(0,k)
        W[k * 2 + 1][col] = 1.0    # c(1,k)
        col += 1
        
        # m5 = (a11+a12) * b2(k2)
        U[0][col] = 1.0
        U[1][col] = 1.0
        V[15 + k2][col] = 1.0
        W[k * 2 + 0][col] = -1.0   # c(0,k)
        W[k2 * 2 + 0][col] = 1.0   # c(0,k2)
        col += 1
        
        # m6 = (a21-a11) * (b1(k) + b1(k2))
        U[0][col] = -1.0
        U[2][col] = 1.0
        V[k][col] = 1.0
        V[k2][col] = 1.0
        W[k2 * 2 + 1][col] = 1.0   # c(1,k2)
        col += 1
        
        # m7 = (a12-a22) * (b2(k) + b2(k2))
        U[1][col] = 1.0
        U[3][col] = -1.0
        V[15 + k][col] = 1.0
        V[15 + k2][col] = 1.0
        W[k * 2 + 0][col] = 1.0    # c(0,k)
        col += 1
    
    # Single column 14
    k = 14
    # n1 = a11 * b1(14)
    U[0][col] = 1.0
    V[k][col] = 1.0
    W[k * 2 + 0][col] = 1.0
    col += 1
    
    # n2 = a12 * b2(14)
    U[1][col] = 1.0
    V[15 + k][col] = 1.0
    W[k * 2 + 0][col] = 1.0
    col += 1
    
    # n3 = a21 * b1(14)
    U[2][col] = 1.0
    V[k][col] = 1.0
    W[k * 2 + 1][col] = 1.0
    col += 1
    
    # n4 = a22 * b2(14)
    U[3][col] = 1.0
    V[15 + k][col] = 1.0
    W[k * 2 + 1][col] = 1.0
    col += 1
    
    # Verify with numpy
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check: sum_r U[i,r] * V[j,r] * W[k,r] should equal T[i,j,k]
    # T[i,j,k] = 1 if output entry (k) = A entry (i) * B entry (j)
    # i in 0..3 (A entries), j in 0..29 (B entries), k in 0..29 (C entries)
    
    # Build tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):  # row of A
        for j in range(m):  # col of A / row of B
            for k in range(p):  # col of B / col of C
                # A[i,j] * B[j,k] contributes to C[i,k]
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Compute reconstruction
    # R[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    R = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    if not np.allclose(R, T, atol=1e-10):
        # Fallback to schoolbook
        rank_s = n * m * p
        U_s = np.zeros((n * m, rank_s))
        V_s = np.zeros((m * p, rank_s))
        W_s = np.zeros((p * n, rank_s))
        c_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, c_idx] = 1
                    V_s[j * p + k, c_idx] = 1
                    W_s[k * n + i, c_idx] = 1
                    c_idx += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Check time
    if time.time() > deadline - 5:
        # Still return, we're done
        pass
    
    return {"U": U, "V": V, "W": W}
