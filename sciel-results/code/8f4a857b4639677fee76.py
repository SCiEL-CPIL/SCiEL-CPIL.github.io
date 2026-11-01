import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin

    # The matmul tensor T[i, a, j] = 1 for all i in [0,n), a in [0,m), j in [0,p)
    # We want: T[i,a,j] = sum_t U[i*m+a, t] * V[a*p+j, t] * W[j*n+i, t]
    
    # Strategy: Use 4 Strassen blocks (each rank 7) for columns 0-7, and schoolbook (rank 4) for column 8
    # Total rank = 4*7 + 4 = 32
    
    # Let's try to find a better construction.
    # The tensor is 2x2x9. 
    # We can view it as 3 blocks of 2x2x3.
    # Rank of 2x2x3 matmul:
    # Schoolbook: 2*2*3 = 12
    # Strassen-like: Can we do better?
    # R(2,2,3) <= 8 is known. Let's try to construct a rank-8 decomposition for 2x2x3.
    # If we can do 2x2x3 in rank 8, then 3 * 8 = 24.
    # If we can do 2x2x3 in rank 9, then 3 * 9 = 27.
    # If we can do 2x2x3 in rank 10, then 3 * 10 = 30.
    
    # Let's first verify the rank-32 solution.
    rank_32 = 32
    U = np.zeros((n*m, rank_32), dtype=float)
    V = np.zeros((m*p, rank_32), dtype=float)
    W = np.zeros((p*n, rank_32), dtype=float)
    
    # Strassen block for 2x2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-d)(g+h)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # In our tensor notation:
    # U index: i*m + a. A is 2x2, so i in {0,1}, a in {0,1}.
    # V index: a*p + j. B is 2x2 (subblock), so a in {0,1}, j in {0,1} (local).
    # W index: j*n + i. C is 2x2, so j in {0,1} (local), i in {0,1}.
    
    # For block k (k=0,1,2,3), columns are 2k, 2k+1.
    for block in range(4):
        k = block * 2
        base_t = block * 7
        
        # Local indices for B columns: 0 -> global 2k, 1 -> global 2k+1
        # Local indices for C columns: 0 -> global 2k, 1 -> global 2k+1
        
        # m1 = (a+d)(e+h)
        t = base_t
        U[0, t] = 1  # a
        U[3, t] = 1  # d
        V[0*p + k, t] = 1  # e
        V[1*p + (k+1), t] = 1  # h
        W[k*n + 0, t] = 1  # C00
        W[(k+1)*n + 1, t] = 1  # C11
        
        # m2 = (c+d)e
        t = base_t + 1
        U[2, t] = 1  # c
        U[3, t] = 1  # d
        V[0*p + k, t] = 1  # e
        W[k*n + 1, t] = 1  # C01
        W[(k+1)*n + 1, t] = -1  # C11
        
        # m3 = a(f-h)
        t = base_t + 2
        U[0, t] = 1  # a
        V[1*p + k, t] = 1  # f
        V[1*p + (k+1), t] = -1  # h
        W[(k+1)*n + 0, t] = 1  # C10
        W[(k+1)*n + 1, t] = 1  # C11
        
        # m4 = d(g-e)
        t = base_t + 3
        U[3, t] = 1  # d
        V[0*p + k, t] = -1  # e
        V[1*p + k, t] = 1  # g
        W[k*n + 0, t] = 1  # C00
        W[k*n + 1, t] = 1  # C01
        
        # m5 = (a+b)h
        t = base_t + 4
        U[0, t] = 1  # a
        U[1, t] = 1  # b
        V[1*p + (k+1), t] = 1  # h
        W[k*n + 0, t] = -1  # C00
        W[(k+1)*n + 0, t] = 1  # C10
        
        # m6 = (c-d)(f+h)
        t = base_t + 5
        U[2, t] = 1  # c
        U[0, t] = -1  # a? No, c-d. So U[2]=1, U[3]=-1? Wait, A is [[a,b],[c,d]].
        # c is index 2, d is index 3.
        U[2, t] = 1
        U[3, t] = -1
        V[1*p + k, t] = 1  # f
        V[1*p + (k+1), t] = 1  # h
        W[(k+1)*n + 1, t] = 1  # C11
        
        # m7 = (b-d)(g+h)
        t = base_t + 6
        U[1, t] = 1  # b
        U[3, t] = -1  # d
        V[0*p + k, t] = 1  # g
        V[1*p + (k+1), t] = 1  # h
        W[k*n + 0, t] = 1  # C00
    
    # Schoolbook for column 8 (k=8)
    base_t = 28
    # C00 += a*e
    t = base_t
    U[0, t] = 1
    V[0*p + 8, t] = 1
    W[8*n + 0, t] = 1
    
    # C00 += b*h
    t = base_t + 1
    U[1, t] = 1
    V[1*p + 8, t] = 1
    W[8*n + 0, t] = 1
    
    # C01 += c*e
    t = base_t + 2
    U[2, t] = 1
    V[0*p + 8, t] = 1
    W[8*n + 1, t] = 1
    
    # C01 += d*h
    t = base_t + 3
    U[3, t] = 1
    V[1*p + 8, t] = 1
    W[8*n + 1, t] = 1
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        n, m, p = DIMENSIONS
        T_decomp = np.zeros((n, m, p))
        for t in range(U_mat.shape[1]):
            u_t = U_mat[:, t].reshape(n, m)
            v_t = V_mat[:, t].reshape(m, p)
            w_t = W_mat[:, t].reshape(p, n)
            for i in range(n):
                for a in range(m):
                    for j in range(p):
                        T_decomp[i, a, j] += u_t[i, a] * v_t[a, j] * w_t[j, i]
        return np.allclose(T_decomp, 1.0, atol=tol)
    
    if not verify_decomposition(U, V, W):
        # Fallback: return schoolbook rank 36
        rank_school = n * m * p
        U_s = np.zeros((n*m, rank_school), dtype=float)
        V_s = np.zeros((m*p, rank_school), dtype=float)
        W_s = np.zeros((p*n, rank_school), dtype=float)
        for i in range(n):
            for a in range(m):
                for j in range(p):
                    t = i*m*p + a*p + j
                    U_s[i*m+a, t] = 1
                    V_s[a*p+j, t] = 1
                    W_s[j*n+i, t] = 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Now try to improve.
    # Can we do 2x2x3 in rank 8?
    # Let's try to find a rank-8 decomposition for 2x2x3.
    # If we find one, we can use 3 blocks of rank 8 = rank 24.
    
    # Let's try a random search for 2x2x3 rank 8.
    # We need to solve for U, V, W of shape (4,8), (6,8), (6,8) such that
    # sum_t U[:,t] * V[:,t] * W[:,t] = 1 (broadcasted to 2x2x3)
    
    # This is a hard problem. Let's try to see if we can merge some terms from the Strassen blocks.
    # For now, return the rank-32 solution.
    
    # Try to reduce rank by 1?
    # Maybe we can combine the last schoolbook term with the last Strassen block?
    # The last Strassen block handles columns 6,7. The schoolbook handles column 8.
    # They are disjoint in the j-index, so they are independent.
    # The rank is additive for disjoint supports.
    # So rank(2x2x9) <= rank(2x2x8) + rank(2x2x1).
    # rank(2x2x8) <= 4 * 7 = 28.
    # rank(2x2x1) = 2 (since C is 2x1, A is 2x2, B is 2x1. C0 = a*e + b*h, C1 = c*e + d*h. Rank 4? No, rank 2x2x1 is 4.
    # Wait, T[i,a,j] for j=0. T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1.
    # This is just the identity tensor on A and B.
    # Rank of 2x2x1 matmul is 4.
    # So 28 + 4 = 32.
    
    # What if we group as 2x2x4 + 2x2x4 + 2x2x1?
    # rank(2x2x4) <= 2 * 7 = 14.
    # 14 + 14 + 4 = 32.
    
    # What if we group as 2x2x3 + 2x2x3 + 2x2x3?
    # If rank(2x2x3) <= 8, then 3*8 = 24.
    # If rank(2x2x3) <= 9, then 3*9 = 27.
    # If rank(2x2x3) <= 10, then 3*10 = 30.
    
    # Let's try to find a rank-10 decomposition for 2x2x3.
    # Schoolbook is 12.
    # Can we do 10?
    
    # Let's try a random search for 2x2x3 rank 10.
    # We have 4*10 + 6*10 + 6*10 = 160 variables.
    # The tensor has 2*2*3 = 12 entries.
    # So we have 12 equations.
    # This is underdetermined. We can easily find a solution.
    # But we need integer/half-integer entries.
    
    # Let's just return the rank-32 solution for now.
    # If we have time, we can try to optimize.
    
    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a rank-31 decomposition.
    # This is unlikely to succeed quickly.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
