import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _strassen_2x2_2x2():
    """Return U, V, W for 2x2 x 2x2 -> 2x2, rank 7.
    U: (4,7), V: (4,7), W: (4,7)
    Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 0: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # Term 1: (A10+A11)B00 -> C10+=1, C11-=1
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    
    # Term 2: A00(B01-B11) -> C01+=1, C11+=1
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # Term 3: A11(B10-B00) -> C10+=1, C00+=1
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[2,3]=1; W[0,3]=1
    
    # Term 4: (A00+A01)B11 -> C01-=1, C11+=1
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=-1; W[3,4]=1
    
    # Term 5: (A10-A00)(B00+B01) -> C11+=1
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # Term 6: (A01-A11)(B10+B11) -> C00+=1
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _strassen_2x2_2x3():
    """Return U, V, W for 2x2 x 2x3 -> 2x3, rank 14 (two Strassen 2x2x2).
    U: (4,14), V: (6,14), W: (6,14)
    Indexing: U[i*2+j, t], V[j*3+k, t], W[k*2+i, t]
    """
    U1, V1, W1 = _strassen_2x2_2x2()
    
    U = np.zeros((4, 14))
    V = np.zeros((6, 14))
    W = np.zeros((6, 14))
    
    # First 2x2 x 2x2: columns 0,1 of B -> columns 0,1 of C
    U[:, :7] = U1
    V[:4, :7] = V1  # B00, B01, B10, B11 -> V[0..3, :] for 2x2 part
    W[:4, :7] = W1  # C00, C01, C10, C11 -> W[0..3, :] for 2x2 part
    
    # Second 2x2 x 2x2: column 2 of B -> column 2 of C
    # B has shape 2x3, so B[:, 2] is the third column
    # V index for column 2: j*3+2 for j=0,1
    # W index for column 2: k*2+2 for k=0,1
    U[:, 7:] = U1
    V[2, 7:14] = V1[0, :]  # B[0,2]
    V[5, 7:14] = V1[2, :]  # B[1,2]
    W[2, 7:14] = W1[0, :]  # C[0,2]
    W[5, 7:14] = W1[2, :]  # C[1,2]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def _verify_decomposition(U, V, W, n, m, p):
        r = U.shape[1]
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i,j,k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
        return np.allclose(res, 1.0, atol=1e-9)
    
    # Try Strassen-based composition
    # Partition: A (4x4) into 2x2 blocks of 2x2
    # B (4x5) into 2x2 blocks: top 2 rows x left 2 cols (2x2), top 2 rows x right 3 cols (2x3), etc.
    # Actually, let's partition B as 4x5 into blocks:
    # B = [[B00 (2x2), B01 (2x3)], [B10 (2x2), B11 (2x3)]]
    # C = [[C00 (2x2), C01 (2x3)], [C10 (2x2), C11 (2x3)]]
    
    # Strassen terms for 2x2 block structure:
    # M1 = (A00+A11)(B00+B11): 2x2 x (2x2 + 2x3) -> need to handle mixed sizes
    # This is tricky because B00 is 2x2 and B11 is 2x3
    
    # Alternative: pad B to 4x6 with zeros, then use 2x2 x 2x2 blocks
    # B_padded (4x6): [[B00(2x2), B01(2x2), B01_pad(2x2)], [B10(2x2), B11(2x2), B11_pad(2x2)]]
    # Where B01_pad has the 3rd column of original B01, B11_pad has 3rd column of B11
    # C_padded (4x6): [[C00(2x2), C01(2x2), C01_pad(2x2)], [C10(2x2), C11(2x2), C11_pad(2x2)]]
    
    # Use Strassen on 2x2 blocks where each "multiplication" is 2x2 x 2x2 (rank 7)
    # 7 terms * 7 rank = 49 total rank
    # But we only need 4x5 output, so we can drop the padded column
    
    U_s = np.zeros((16, 49))
    V_s = np.zeros((20, 49))  # 4x5 = 20
    W_s = np.zeros((20, 49))  # 5x4 = 20
    
    U2, V2, W2 = _strassen_2x2_2x2()
    
    # Strassen terms for outer 2x2:
    # Each term is a 2x2 x 2x2 multiplication at the block level
    # Block indices: (bi, bj) where bi, bj in {0,1}
    # A block (bi, bj) is A[2*bi:2*bi+2, 2*bj:2*bj+2]
    # B block (bi, bj) is B[2*bi:2*bi+2, 2*bj:2*bj+2] but B is 4x5 so columns 0-3 are 2x2, col 4 is separate
    
    # Let's handle this differently. Use the 2x2 x 2x3 decomposition for mixed cases.
    
    # Actually, let me just try the direct approach with the 2x2x2 and 2x2x3 decompositions.
    
    # Outer Strassen terms:
    # T0: (A00+A11)(B00+B11) where B00 is 2x2, B11 is 2x3 -> result is 2x3
    # T1: (A10+A11)B00 where B00 is 2x2 -> result is 2x2
    # T2: A00(B01-B11) where B01, B11 are 2x3 -> result is 2x3
    # T3: A11(B10-B00) where B10, B00 are 2x2 -> result is 2x2
    # T4: (A00+A01)B11 where B11 is 2x3 -> result is 2x3
    # T5: (A10-A00)(B00+B01) where B00 is 2x2, B01 is 2x3 -> result is 2x3
    # T6: (A01-A11)(B10+B11) where B10 is 2x2, B11 is 2x3 -> result is 2x3
    
    # For each term, the "multiplication" is either 2x2 x 2x2 (rank 7) or 2x2 x 2x3 (rank 14)
    
    # T0: 2x2 x (2x2+2x3) - this is a sum, so we need to decompose (B00+B11) which is 2x3
    # B00+B11: B00 is 2x2 (cols 0-1), B11 is 2x3 (cols 2-4). So B00+B11 = [[B00[0,0]+0, B00[0,1]+0, B11[0,0]], [B00[1,0]+0, B00[1,1]+0, B11[1,0]]]
    # Wait, this doesn't make sense dimensionally. B00 is top-left 2x2, B11 is bottom-right 2x3.
    # B = [[B00(2x2), B01(2x3)], [B10(2x2), B11(2x3)]]
    # So B00 is rows 0-1, cols 0-1. B11 is rows 2-3, cols 2-4.
    # B00+B11 doesn't make sense as a single matrix since they're in different positions.
    
    # I think the Strassen algorithm applies to square matrices. For rectangular, we need a different approach.
    
    # Let me just use the schoolbook and try to optimize numerically, but with a better strategy.
    
    # Try rank 60-75 with structured initialization
    for r in range(60, 80):
        if time.time() > deadline - 30:
            break
        if r >= best_rank:
            continue
        
        found = False
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(12345 + r * 100 + trial)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Try to refine with a few iterations of gradient descent
            lr = 0.01
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # d/dU[i*m+j, t] of (U[i*m+j,:] . (V[j*p+k,:] * W[k*n+i,:]))
                            # = V[j*p+k,:] * W[k*n+i,:]
                            uw = V[j*p+k, :] * W[k*n+i, :]
                            grad_U[i*m+j, :] += uw
                            grad_V[j*p+k, :] += U[i*m+j, :] * W[k*n+i, :]
                            grad_W[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if it % 10 == 0:
                    if _verify_decomposition(U, V, W, n, m, p):
                        # Round to half-integers
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                            best_rank = r
                            best_U = U_r
                            best_V = V_r
                            best_W = W_r
                            found = True
                            break
            if found:
                break
        
        if found:
            break
    
    def to_list(mat):
        return [[float(val) for val in row] for row in mat]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
