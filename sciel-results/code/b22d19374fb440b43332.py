import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition correctly represents matrix multiplication."""
    R = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute via decomposition: C[k*n+i, r] = sum_t U[i*m+k, t] * V[k*p+r, t] * W[t]
        # Actually: C[i,j] = sum_k A[i,k]*B[k,j]
        # The decomposition gives: C = sum_t (U[:,t] reshaped to (n,m)) * (V[:,t] reshaped to (m,p)) * W[:,t]
        # Wait, let me re-read the convention.
        # W[k*n+i, r] - so W is indexed by [k*n+i, r] where k is row of B / col of A, i is row of A / row of C
        # U is (n*m, rank): U[i*m+k, t] = coefficient for A[i,k]
        # V is (m*p, rank): V[k*p+j, t] = coefficient for B[k,j]  
        # W is (p*n, rank): W[j*n+i, t] = coefficient for C[i,j]
        # Wait, the task says W[k*n+i, r]. Let me re-read.
        # "The output factor follows the official trace convention W[k*n+i,r]"
        # So W has shape (p*n, rank) and W[j*n+i, t] gives coefficient for C[i,j]
        # where i is row index of C (and A), j is col index of C (and B)
        
        C_computed = np.zeros((n, p))
        for t in range(R):
            # U[:,t] reshaped to (n,m) gives the "A pattern"
            # V[:,t] reshaped to (m,p) gives the "B pattern"
            # W[:,t] reshaped to (p,n) gives the "C pattern"
            U_t = U[:, t].reshape(n, m)
            V_t = V[:, t].reshape(m, p)
            W_t = W[:, t].reshape(p, n)  # W[j*n+i, t]
            
            # The contribution is: sum_{i,k} U_t[i,k]*A[i,k] * sum_{k,j} V_t[k,j]*B[k,j] -> no
            # Actually the tensor decomposition means:
            # C[i,j] = sum_t W_t[j,i] * (sum_{i',k} U_t[i',k] * A[i',k]) * ... 
            # No, it's a bilinear decomposition:
            # C[i,j] = sum_t W[j*n+i, t] * (linear form in A)_t * (linear form in B)_t
            
            # (linear form in A)_t = sum_{i',k'} U[i'*m+k', t] * A[i',k']
            # (linear form in B)_t = sum_{k',j'} V[k'*p+j', t] * B[k',j']
            
            lin_A = np.sum(U[:, t].reshape(n, m) * A)
            lin_B = np.sum(V[:, t].reshape(m, p) * B)
            C_computed += W_t[j, i] * lin_A * lin_B  # This is wrong, need to be more careful
        
        # Let me redo this properly
        C_computed = np.zeros((n, p))
        for t in range(R):
            lin_A = np.sum(U[:, t].reshape(n, m) * A)
            lin_B = np.sum(V[:, t].reshape(m, p) * B)
            for i in range(n):
                for j in range(p):
                    C_computed[i, j] += W[j * n + i, t] * lin_A * lin_B
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Try to find a rank-6 decomposition for 2x2 * 2x3
    # If we can find one, we can use Kronecker with Strassen (rank 7) to get rank 42
    
    # First, let's try to solve for a rank-6 decomposition of M_{2,2,3}
    # The tensor has 4*6*6 = 144 entries
    # Each rank-1 term has (2*2)*(2*3)*(2*3) = 4*6*6 = 144 parameters but only 4*6+6*6-1 free after scaling
    # Actually each term is u ⊗ v ⊗ w where u ∈ R^4, v ∈ R^6, w ∈ R^6
    # With 6 terms, we have 6*(4+6+6) = 84 parameters (up to scaling)
    # The tensor is determined by its 144 entries but has structure
    
    # Let me try a specific ansatz for rank-6 M_{2,2,3}
    # X (2x2) * Y (2x3) = Z (2x3)
    # 
    # Key observation: The first row of Z depends only on the first row of X and all of Y
    # The second row of Z depends only on the second row of X and all of Y
    # 
    # So we can separate:
    # Z[0,:] = X[0,0]*Y[0,:] + X[0,1]*Y[1,:]
    # Z[1,:] = X[1,0]*Y[0,:] + X[1,1]*Y[1,:]
    #
    # For Z[0,:], we need to compute a 1x3 vector from two 1x2 vectors (rows of X) and 2x3 matrix Y
    # This is: z = x0 * Y[0,:] + x1 * Y[1,:] where x0=X[0,0], x1=X[0,1]
    # This can be done with 3 multiplications: x0*Y[0,0], x0*Y[0,1], x0*Y[0,2], x1*Y[1,0], x1*Y[1,1], x1*Y[1,2]
    # That's 6 mults for the first row alone. So rank >= 6 just for one row.
    # Total rank >= 12? No wait, we can share.
    
    # Actually, let me think again. The rank of the 2x2*2x3 tensor:
    # Lower bound: The tensor, when viewed as a map R^4 x R^6 -> R^6, 
    # each output entry Z[i,j] = sum_k X[i,k]*Y[k,j]
    # For fixed i, Z[i,:] is a linear map from (X[i,:], Y) to R^3
    # The two rows are independent, so rank >= 2 * rank(1x2 * 2x3)
    # rank(1x2 * 2x3): z = x0*y0 + x1*y1 where x0,x1 are scalars, y0,y1 are 1x3 vectors
    # This requires 3 mults (can't do better since output is 3D)
    # So rank >= 2*3 = 6
    
    # Can we achieve rank 6? Let's try:
    # p1 = X[0,0] * Y[0,0]  -> contributes to Z[0,0]
    # p2 = X[0,0] * Y[0,1]  -> contributes to Z[0,1]
    # p3 = X[0,0] * Y[0,2]  -> contributes to Z[0,0]... no
    #
    # Let me try:
    # p1 = X[0,0] * Y[0,0]
    # p2 = X[0,0] * Y[0,1]
    # p3 = X[0,0] * Y[0,2]
    # p4 = X[1,0] * Y[1,0]
    # p5 = X[1,0] * Y[1,1]
    # p6 = X[1,0] * Y[1,2]
    # But this misses X[0,1]*Y[1,:] and X[1,1]*Y[0,:]
    
    # I don't think rank 6 is achievable. Let me try rank 12 (schoolbook) and see
    # if the Kronecker construction gives a valid result, then try to optimize.
    
    # Actually, let me just implement the current rank-77 approach cleanly and verify it works.
    # Then if time permits, try to find better decompositions.
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # M1: A00+A11
        {0: 0, 1: 0, 2: 1, 3: 1},   # M2: A10+A11
        {0: 1, 1: 0, 2: 0, 3: 0},   # M3: A00
        {0: 0, 1: 0, 2: 0, 3: 1},   # M4: A11
        {0: 1, 1: 1, 2: 0, 3: 0},   # M5: A00+A01
        {0: -1, 1: 0, 2: 1, 3: 0},  # M6: A10-A00
        {0: 0, 1: 1, 2: 0, 3: -1}   # M7: A01-A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # M1: B00+B11
        {0: 1, 1: 0, 2: 0, 3: 0},   # M2: B00
        {0: 0, 1: 1, 2: 0, 3: -1},  # M3: B01-B11
        {0: -1, 1: 0, 2: 1, 3: 0},  # M4: B10-B00
        {0: 0, 1: 0, 2: 0, 3: 1},   # M5: B11
        {0: 1, 1: 1, 2: 0, 3: 0},   # M6: B00+B01
        {0: 0, 1: 0, 2: 1, 3: 1}    # M7: B10+B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # C00 from M1
        [(1, 0, 1), (1, 1, -1)],    # C01 from M1? No...
        # Let me redo: C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
        # recon[s] = list of (row, col, coeff) for output C
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # M1 -> C00 += 1, C11 += 1
        [(1, 0, 1)],                # M2 -> C10 += 1
        [(0, 1, 1), (1, 1, 1)],     # M3 -> C01 += 1, C11 += 1
        [(0, 0, 1), (1, 0, 1)],     # M4 -> C00 += 1, C10 += 1
        [(0, 0, -1), (0, 1, 1)],    # M5 -> C00 += -1, C01 += 1
        [(1, 1, 1)],                # M6 -> C11 += 1
        [(0, 0, 1)]                 # M7 -> C00 += 1
    ]
    
    col = 0
    for s in range(7):
        for si in range(7):
            # U part: alpha[s][outer_block] * alpha[si][inner_2x2]
            for r in range(4):
                if alpha[s][r] != 0:
                    ar, br = r // 2, r % 2  # outer block row/col
                    for a in range(4):
                        if alpha[si][a] != 0:
                            x_row, x_col = a // 2, a % 2  # inner 2x2 position
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[s][r] * alpha[si][a]
            
            # V part: beta[s][outer_block] * beta[si][inner_2x2]
            for c in range(4):
                if beta[s][c] != 0:
                    ac, bc = c // 2, c % 2  # outer block row/col
                    for b in range(4):
                        if beta[si][b] != 0:
                            y_row, y_inner_col = b // 2, b % 2  # inner position
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[s][c] * beta[si][b]
            
            # W part: recon[s][outer] * recon[si][inner]
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1
        
        # Extra 4 terms for the 3rd column of B (2x2 * 2x1 part)
        for u in range(2):
            for v in range(2):
                # U part
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V part
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                col += 1
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    else:
        # Fallback: schoolbook rank 96
        R2 = n * m * p
        U2 = np.zeros((n * m, R2))
        V2 = np.zeros((m * p, R2))
        W2 = np.zeros((p * n, R2))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    t = i * m * p + k * p + j
                    U2[i * m + k, t] = 1
                    V2[k * p + j, t] = 1
                    W2[j * n + i, t] = 1
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
