import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    size_u = n * m   # 4
    size_v = m * p   # 18
    size_w = p * n   # 18

    def verify(U, V, W, tol=1e-8):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def build_schoolbook():
        R = n * m * p
        U = np.zeros((size_u, R))
        V = np.zeros((size_v, R))
        W = np.zeros((size_w, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Build a verified Strassen 2x2x2 decomposition (rank 7)
    # Using a known valid set:
    # We need U[i,j], V[j,k], W[k,i] such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k in {0,1}
    
    def build_strassen_2x2x2():
        # Verified Strassen decomposition with rank 7
        # Local indices: i,j in {0,1} for A and C, k in {0,1} for B
        # U: (4, 7) where row = i*2+j
        # V: (4, 7) where row = j*2+k
        # W: (4, 7) where row = k*2+i
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 0: (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # C00 += M0, C11 += M0
        
        # Term 1: (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        # C10 += M1
        
        # Term 2: A00(B10-B00)
        U[0, 2] = 1
        V[2, 2] = 1; V[0, 2] = -1
        W[0, 2] = 1; W[2, 2] = 1
        # C00 += M2, C10 += M2
        
        # Term 3: A11(B01-B11)
        U[3, 3] = 1
        V[1, 3] = 1; V[3, 3] = -1
        W[1, 3] = 1; W[3, 3] = 1
        # C01 += M3, C11 += M3
        
        # Term 4: (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        # C00 -= M4, C01 += M4
        
        # Term 5: (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[1, 5] = -1
        # C01 -= M5  (this is unusual, let me verify)
        
        # Term 6: (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[3, 6] = 1
        # C00 -= M6, C11 += M6
        
        # Verify: C00 = M0 + M2 - M4 - M6
        # C01 = M3 + M4 - M5
        # C10 = M1 + M2
        # C11 = M0 + M3 + M6
        
        # Let's check C00:
        # T[0,0,0] = U[0,0]*V[0,0]*W[0,0] + U[0,2]*V[0,2]*W[0,2] + U[0,4]*V[0,4]*W[0,4] + U[0,5]*V[0,5]*W[0,5] + U[0,6]*V[0,6]*W[0,6]
        # = 1*1*1 + 1*(-1)*1 + 1*0*(-1) + (-1)*1*0 + 0*0*(-1) = 1 - 1 + 0 + 0 + 0 = 0
        # That's wrong! We need T[0,0,0] = 1.
        
        # Let me try a different, verified Strassen decomposition.
        # I'll use the one from the literature more carefully.
        
        # Actually, let me just brute-force verify a known set.
        # The standard Strassen formulas give:
        # C00 = M1 + M3 - M5 - M7
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M4 + M7
        # where:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # So:
        # W[0,0]=1 (M1->C00), W[0,2]=1 (M3->C00), W[0,4]=-1 (M5->C00), W[0,6]=-1 (M7->C00)
        # W[1,3]=1 (M4->C01), W[1,4]=1 (M5->C01)
        # W[2,1]=1 (M2->C10), W[2,2]=1 (M3->C10)
        # W[3,0]=1 (M1->C11), W[3,1]=-1 (M2->C11), W[3,3]=1 (M4->C11), W[3,6]=1 (M7->C11)
        # M6 is not used in the C formula, which means it's not needed for the rank-7 decomposition
        # of the 2x2x2 tensor. But wait, that gives only 6 terms used in C. Let me recheck.
        
        # Actually, looking at this more carefully, the standard Strassen uses 7 multiplications
        # but the way they combine to form C uses all 7. Let me re-derive.
        
        # C00 = M1 + M3 - M5 - M7  (uses M1, M3, M5, M7)
        # C01 = M4 + M5            (uses M4, M5)
        # C10 = M2 + M3            (uses M2, M3)
        # C11 = M1 - M2 + M4 + M7  (uses M1, M2, M4, M7)
        
        # So M6 is not used! That means we only need 6 terms? No, that can't be right.
        # The rank of 2x2x2 is 7, so we need 7 terms.
        
        # I think the issue is that the standard Strassen formulas I'm recalling are slightly off.
        # Let me just use a computationally verified set.
        
        # For safety, I'll skip the Strassen part and use a different approach.
        # Instead of Strassen for 2x2x2 + schoolbook for 2x2x7, let me try to find
        # a lower-rank decomposition using the structure of the problem.
        
        # For 2x2 * 2x9, the tensor T[i,j,k] = 1 for all i,j in {0,1}, k in {0,...,8}.
        # This is a "slice" tensor where each slice k is the 2x2x2 all-ones tensor.
        
        # The rank of the 2x2x2 all-ones tensor is 4 (schoolbook), not 7.
        # Wait, no. The 2x2x2 matrix multiplication tensor has rank 7, but the
        # all-ones tensor (where every entry is 1) has rank 1!
        
        # I'm confusing two different tensors. The matrix multiplication tensor
        # T_matmul[i,j,k] = 1 means C[i,k] = sum_j A[i,j] B[j,k].
        # The all-ones tensor T_ones[i,j,k] = 1 for all i,j,k has rank 1.
        
        # So the tensor we're decomposing IS the matrix multiplication tensor,
        # and its rank for 2x2x2 is 7.
        
        # Let me just give up on Strassen and use schoolbook (rank 36) as a safe baseline.
        # The key insight is that we need to find ANY valid decomposition, and rank 36
        # is always valid.
        
        return None  # Skip Strassen, use schoolbook

    # Use schoolbook as the safe baseline
    U, V, W = build_schoolbook()
    
    # Try to find a better decomposition using iterative optimization
    # We'll try to find a decomposition with rank less than 36
    
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try rank 35, 34, ..., down to some minimum
    # Use a simple iterative refinement approach
    for target_rank in range(35, 24, -1):
        R = target_rank
        U_r = np.zeros((size_u, R))
        V_r = np.zeros((size_v, R))
        W_r = np.zeros((size_w, R))
        
        # Initialize with schoolbook for the first R terms
        for col in range(R):
            i = col // (m * p)
            j = (col // p) % m
            k = col % p
            U_r[i * m + j, col] = 1
            V_r[j * p + k, col] = 1
            W_r[k * n + i, col] = 1
        
        # Iteratively try to reduce the error
        for iteration in range(200):
            # Compute residuals
            res = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                        res[i, j, k] = val - 1.0
            
            max_res = np.max(np.abs(res))
            if max_res < 1e-10:
                break
            
            # Update one factor at a time (Gauss-Seidel style)
            # Update U
            for r in range(R):
                for i in range(n):
                    for j in range(m):
                        # d/dU[i*m+j, r] of sum_{i,j,k} (sum_r' U[i*m+j,r'] * V[j*p+k,r'] * W[k*n+i,r'] - 1)^2
                        # = 2 * sum_k (V[j*p+k, r] * W[k*n+i, r]) * (sum_r' U[i*m+j,r'] * V[j*p+k,r'] * W[k*n+i,r'] - 1)
                        grad = 0.0
                        for k in range(p):
                            val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                            grad += V_r[j * p + k, r] * W_r[k * n + i, r] * (val - 1.0)
                        U_r[i * m + j, r] -= 0.1 * grad
            
            # Update V
            for r in range(R):
                for j in range(m):
                    for k in range(p):
                        grad = 0.0
                        for i in range(n):
                            val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                            grad += U_r[i * m + j, r] * W_r[k * n + i, r] * (val - 1.0)
                        V_r[j * p + k, r] -= 0.1 * grad
            
            # Update W
            for r in range(R):
                for k in range(p):
                    for i in range(n):
                        grad = 0.0
                        for j in range(m):
                            val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                            grad += U_r[i * m + j, r] * V_r[j * p + k, r] * (val - 1.0)
                        W_r[k * n + i, r] -= 0.1 * grad
        
        # Check if we found a valid decomposition
        if verify(U_r, V_r, W_r):
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = R
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
