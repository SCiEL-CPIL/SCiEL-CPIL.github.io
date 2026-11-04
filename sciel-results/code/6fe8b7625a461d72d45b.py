import numpy as np
import time

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18
    
    # Pre-compute target tensor for fast verification
    # Shape: (nm, mp, pn)
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """
        Efficiently verify if U, V, W decompose the matrix multiplication tensor.
        Uses tensor contraction: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
        """
        rank = U.shape[1]
        T_comp = np.zeros((nm, mp, pn))
        for r in range(rank):
            T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        # Check against target
        if not np.allclose(T_comp, T_target, atol=tol, rtol=0):
            return False
        return True
    
    def snap_to_half_integers(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap
    
    def strassen_2x2x2():
        """
        Strassen algorithm for 2x2 matrix multiplication.
        Returns U, V, W matrices for the 2x2x2 tensor.
        U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
        """
        # Strassen's 7 products:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Index: U[i,j] where i is row of A (0,1), j is col of A (0,1)
        # V[i,j] where i is row of B (0,1), j is col of B (0,1)
        # W[i,j] where i is row of C (0,1), j is col of C (0,1)
        
        # M1: (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 3] = 1
        
        # M2: (A21+A22)B11
        U[2, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1
        W[2, 0] = 1; W[3, 3] = -1
        
        # M3: A11(B12-B22)
        U[0, 0] = 1
        V[1, 0] = 1; V[3, 0] = -1
        W[0, 1] = 1; W[3, 3] = 1
        
        # M4: A22(B21-B11)
        U[3, 0] = 1
        V[2, 0] = 1; V[0, 0] = -1
        W[1, 0] = 1; W[2, 0] = 1
        
        # M5: (A11+A12)B22
        U[0, 0] = 1; U[1, 0] = 1
        V[3, 0] = 1
        W[0, 0] = -1; W[0, 1] = 1
        
        # M6: (A21-A11)(B11+B12)
        U[2, 0] = 1; U[0, 0] = -1
        V[0, 0] = 1; V[1, 0] = 1
        W[3, 3] = 1
        
        # M7: (A12-A22)(B21+B22)
        U[1, 0] = 1; U[3, 0] = -1
        V[2, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1
        
        return U, V, W
    
    def laderman_3x3x3():
        """
        Laderman algorithm for 3x3 matrix multiplication (rank 23).
        Returns U, V, W matrices for the 3x3x3 tensor.
        U shape: (9, 23), V shape: (9, 23), W shape: (9, 23)
        """
        # Laderman's 23 products for 3x3 matrix multiplication
        # This is a known result; I'll construct it from the standard formulation
        
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
        # Laderman's algorithm uses 23 rank-1 tensors
        # I'll use a known construction
        
        # Products 1-9: diagonal-like
        # M1 = A11*B11
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        
        # M2 = A12*B21
        U[1, 1] = 1; V[3, 1] = 1; W[1, 0] = 1
        
        # M3 = A13*B31
        U[2, 2] = 1; V[6, 2] = 1; W[2, 0] = 1
        
        # M4 = A21*B12
        U[3, 3] = 1; V[1, 3] = 1; W[0, 1] = 1
        
        # M5 = A22*B22
        U[4, 4] = 1; V[4, 4] = 1; W[1, 1] = 1
        
        # M6 = A23*B32
        U[5, 5] = 1; V[7, 5] = 1; W[2, 1] = 1
        
        # M7 = A31*B13
        U[6, 6] = 1; V[2, 6] = 1; W[0, 2] = 1
        
        # M8 = A32*B23
        U[7, 7] = 1; V[5, 7] = 1; W[1, 2] = 1
        
        # M9 = A33*B33
        U[8, 8] = 1; V[8, 8] = 1; W[2, 2] = 1
        
        # Products 10-23: off-diagonal combinations
        # M10 = (A11+A12+A13+A21+A22+A23+A31+A32+A33)*(B11+B12+B13+B21+B22+B23+B31+B32+B33)
        # This is getting complex; let me use a different approach
        
        # Actually, let me use a known rank-23 decomposition for 3x3
        # I'll construct it more carefully
        
        # Clear and redo
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
        # Laderman's 23 factors
        # Factor 1: A11*B11 -> C11
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # Factor 2: A12*B21 -> C11
        U[1,1]=1; V[3,1]=1; W[0,0]+=1
        # Factor 3: A13*B31 -> C11
        U[2,2]=1; V[6,2]=1; W[0,0]+=1
        # Factor 4: A21*B12 -> C21
        U[3,3]=1; V[1,3]=1; W[3,0]=1
        # Factor 5: A22*B22 -> C22
        U[4,4]=1; V[4,4]=1; W[4,1]=1
        # Factor 6: A23*B32 -> C22
        U[5,5]=1; V[7,5]=1; W[4,1]+=1
        # Factor 7: A31*B13 -> C31
        U[6,6]=1; V[2,6]=1; W[6,0]=1
        # Factor 8: A32*B23 -> C32
        U[7,7]=1; V[5,7]=1; W[7,1]=1
        # Factor 9: A33*B33 -> C33
        U[8,8]=1; V[8,8]=1; W[8,2]=1
        # Factor 10: (A11+A12)*(B21+B22) -> C11,C12
        U[0,9]=1; U[1,9]=1; V[3,9]=1; V[4,9]=1; W[0,0]+=1; W[1,0]=1
        # Factor 11: (A11+A13)*(B31+B32) -> C11,C12
        U[0,10]=1; U[2,10]=1; V[6,10]=1; V[7,10]=1; W[0,0]+=1; W[1,0]+=1
        # Factor 12: (A21+A22)*(B12+B13) -> C21,C22
        U[3,11]=1; U[4,11]=1; V[1,11]=1; V[2,11]=1; W[3,0]+=1; W[4,1]=1
        # Factor 13: (A21+A23)*(B32+B33) -> C21,C22
        U[3,12]=1; U[5,12]=1; V[7,12]=1; V[8,12]=1; W[3,0]+=1; W[4,1]+=1
        # Factor 14: (A31+A32)*(B13+B12) -> C31,C32
        U[6,13]=1; U[7,13]=1; V[2,13]=1; V[1,13]=1; W[6,0]+=1; W[7,1]=1
        # Factor 15: (A31+A33)*(B33+B32) -> C31,C32
        U[6,14]=1; U[8,14]=1; V[8,14]=1; V[7,14]=1; W[6,0]+=1; W[7,1]+=1
        # Factor 16: (A11+A21)*(B11+B21) -> C11,C21
        U[0,15]=1; U[3,15]=1; V[0,15]=1; V[3,15]=1; W[0,0]+=1; W[3,0]=1
        # Factor 17: (A11+A21)*(B12+B22) -> C12,C22
        U[0,16]=1; U[3,16]=1; V[1,16]=1; V[4,16]=1; W[1,0]+=1; W[4,1]=1
        # Factor 18: (A12+A22)*(B11+B21) -> C12,C22
        U[1,17]=1; U[4,17]=1; V[0,17]=1; V[3,17]=1; W[1,0]+=1; W[4,1]+=1
        # Factor 19: (A12+A22)*(B13+B23) -> C13,C23
        U[1,18]=1; U[4,18]=1; V[2,18]=1; V[5,18]=1; W[2,0]=1; W[5,1]=1
        # Factor 20: (A13+A23)*(B11+B21) -> C13,C23
        U[2,19]=1; U[5,19]=1; V[0,19]=1; V[3,19]=1; W[2,0]+=1; W[5,1]=1
        # Factor 21: (A13+A23)*(B12+B22) -> C13,C23
        U[2,20]=1; U[5,20]=1; V[1,20]=1; V[4,20]=1; W[2,0]+=1; W[5,1]+=1
        # Factor 22: (A11+A12+A13)*(B21+B22+B23) -> C12,C13
        U[0,21]=1; U[1,21]=1; U[2,21]=1; V[3,21]=1; V[4,21]=1; V[5,21]=1; W[1,0]+=1; W[2,0]=1
        # Factor 23: (A21+A22+A23)*(B11+B12+B13) -> C21,C23
        U[3,22]=1; U[4,22]=1; U[5,22]=1; V[0,22]=1; V[1,22]=1; V[2,22]=1; W[3,0]+=1; W[5,1]=1
        
        return U, V, W
    
    def tensor_slicing_decomposition():
        """
        Construct the tensor slicing decomposition for 3x5x6.
        Partition m=5 into 2+3.
        Use Strassen (rank 7) for the 2x2x2 part.
        Use Laderman (rank 23) for the 3x3x3 part.
        Total rank: 7 + 23 = 30
        """
        total_rank = 30
        U = np.zeros((nm, total_rank))
        V = np.zeros((mp, total_rank))
        W = np.zeros((pn, total_rank))
        
        # Part 1: 2x2x2 using Strassen (uses j=0,1 for the middle dimension)
        U_s, V_s, W_s = strassen_2x2x2()
        
        # Map Strassen indices to our indices
        # Strassen: A is 2x2, B is 2x2, C is 2x2
        # In our problem: A[i,j] where i in [0,2], j in [0,1] (first 2 rows of middle)
        # B[j,k] where j in [0,1], k in [0,5]
        # C[i,k] where i in [0,2], k in [0,5]
        
        # For the 2x2x2 part:
        # A indices: i in [0,2], j in [0,1] -> U index: i*m + j = i*5 + j
        # B indices: j in [0,1], k in [0,5] -> V index: j*p + k = j*6 + k
        # C indices: i in [0,2], k in [0,5] -> W index: k*n + i = k*3 + i
        
        # Strassen U_s has shape (4, 7) where U_s[ai, r] with ai in [0,3]
        # ai = 0: A[0,0], ai = 1: A[0,1], ai = 2: A[1,0], ai = 3: A[1,1]
        # But our A has 3 rows, so we need to map:
        # Strassen row 0 -> our row 0 (i=0)
        # Strassen row 1 -> our row 1 (i=1)
        # We ignore Strassen row 2,3 (they correspond to i=1 in a 2x2, but we have 3 rows)
        
        # Actually, for the 2x2x2 sub-problem, we use i in [0,1] (first 2 rows)
        # So Strassen U_s[0, r] -> U[0*5 + 0, r] = U[0, r]
        # Strassen U_s[1, r] -> U[1*5 + 0, r] = U[5, r]  (i=1, j=0)
        # Strassen U_s[2, r] -> U[0*5 + 1, r] = U[1, r]  (i=0, j=1)
        # Strassen U_s[3, r] -> U[1*5 + 1, r] = U[6, r]  (i=1, j=1)
        
        # Wait, I need to be more careful. In the 2x2x2 sub-problem:
        # A is 2x2 with indices (i,j) where i in [0,1], j in [0,1]
        # B is 2x2 with indices (j,k) where j in [0,1], k in [0,1]
        # C is 2x2 with indices (i,k) where i in [0,1], k in [0,1]
        
        # But our full problem has:
        # A[i,j] with i in [0,2], j in [0,4]
        # B[j,k] with j in [0,4], k in [0,5]
        # C[i,k] with i in [0,2], k in [0,5]
        
        # For the 2x2x2 part, we restrict:
        # i in [0,1], j in [0,1], k in [0,1]
        
        # So:
        # U index for A[i,j]: i*5 + j
        # V index for B[j,k]: j*6 + k
        # W index for C[i,k]: k*3 + i
        
        # Strassen U_s[ai, r]:
        # ai=0: i=0,j=0 -> U[0*5+0, r] = U[0, r]
        # ai=1: i=0,j=1 -> U[0*5+1, r] = U[1, r]
        # ai=2: i=1,j=0 -> U[1*5+0, r] = U[5, r]
        # ai=3: i=1,j=1 -> U[1*5+1, r] = U[6, r]
        
        # Strassen V_s[bi, r]:
        # bi=0: j=0,k=0 -> V[0*6+0, r] = V[0, r]
        # bi=1: j=0,k=1 -> V[0*6+1, r] = V[1, r]
        # bi=2: j=1,k=0 -> V[1*6+0, r] = V[6, r]
        # bi=3: j=1,k=1 -> V[1*6+1, r] = V[7, r]
        
        # Strassen W_s[ci, r]:
        # ci=0: i=0,k=0 -> W[0*3+0, r] = W[0, r]
        # ci=1: i=0,k=1 -> W[1*3+0, r] = W[3, r]
        # ci=2: i=1,k=0 -> W[0*3+1, r] = W[1, r]
        # ci=3: i=1,k=1 -> W[1*3+1, r] = W[4, r]
        
        U_map = [0, 1, 5, 6]
        V_map = [0, 1, 6, 7]
        W_map = [0, 3, 1, 4]
        
        for r in range(7):
            for ai in range(4):
                U[U_map[ai], r] = U_s[ai, r]
            for bi in range(4):
                V[V_map[bi], r] = V_s[bi, r]
            for ci in range(4):
                W[W_map[ci], r] = W_s[ci, r]
        
        # Part 2: 3x3x3 using Laderman (uses j=2,3,4 for the middle dimension)
        U_l, V_l, W_l = laderman_3x3x3()
        
        # For the 3x3x3 sub-problem:
        # i in [0,2], j in [2,4], k in [0,5]
        # But Laderman is for 3x3x3, so we need to restrict k to 3 values
        # Let's use k in [0,2] for the 3x3x3 part
        
        # Actually, this is getting complicated. Let me reconsider.
        
        # The tensor slicing approach partitions the middle dimension.
        # For m=5, we split into m1=2 and m2=3.
        # Part 1: i in [0,2], j in [0,1], k in [0,5] -> but this is 3x2x6, not 2x2x2
        # Part 2: i in [0,2], j in [2,4], k in [0,5] -> this is 3x3x6, not 3x3x3
        
        # So the simple tensor slicing doesn't directly give 2x2x2 and 3x3x3.
        # I need to also partition k.
        
        # Let me use a different approach: partition both j and k.
        # For the 2x2x2 part: i in [0,1], j in [0,1], k in [0,1]
        # For the 3x3x3 part: i in [0,2], j in [2,4], k in [2,5]
        
        # This leaves some terms unaccounted for, but let's see if this works.
        
        # Actually, I realize the tensor slicing bound is more subtle.
        # Let me just construct a valid rank-30 decomposition directly.
        
        # For now, let me return a schoolbook decomposition as fallback
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        return U, V, W, rank
    
    def try_rank(r, max_iters=50, seed=42):
        if time.time() > deadline:
            return None
        
        # Use structured initialization
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r) * 0.1
        V = rng.randn(mp, r) * 0.1
        W = rng.randn(pn, r) * 0.1
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.outer(V[j*p:(j+1)*p, :], W[i:pn:n, :])
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.outer(U[:nm:n, :], W[k:pn:n, :])
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for i in range(n):
                for k in range(p):
                    M = np.outer(U[i*m:(i+1)*m, :], V[k:mp:p, :])
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err += (s - 1.0) ** 2
            err = np.sqrt(err / (n * m * p))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
        
        # Final verification with strict tolerance
        if not verify_decomposition(best_U, best_V, best_W, tol=1e-9):
            return None
        
        # Snap to half-integers
        U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
        
        # Verify snapped version
        if not verify_decomposition(U_snap, V_snap, W_snap, tol=1e-9):
            return None
        
        return U_snap, V_snap, W_snap
    
    def schoolbook_decomposition():
        """Schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        return U, V, W, rank
    
    # Try to find a rank-30 decomposition using the tensor slicing approach
    # If that fails, fall back to schoolbook rank 90
    
    best_result = None
    
    # Try rank 30 first
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        result = try_rank(30, max_iters=100, seed=30 * 100 + restart)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W, 30)
            break
    
    if best_result is None:
        # Fallback to schoolbook rank 90
        U, V, W, r = schoolbook_decomposition()
        best_result = (U, V, W, r)
    
    U, V, W, r = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
