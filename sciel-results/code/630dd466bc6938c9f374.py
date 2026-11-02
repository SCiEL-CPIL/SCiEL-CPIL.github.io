import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # M_{1,1,2} rank-2 decomposition:
    # U_112 = [[1, 1]]  (shape 1x2)
    # V_112 = [[1, 0], [0, 1]]  (shape 2x2)
    # W_112 = [[1, 0], [0, 1]]  (shape 2x2)
    
    # For M_{3,4,4}: A(3x4) * B(4x4) -> C(3x4)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Split j into two 2s: j1 in {0,1}, j2 in {2,3}
    # C[i,k] = sum_{j1} A[i,j1]*B[j1,k] + sum_{j2} A[i,j2+2]*B[j2+2,k]
    #
    # For each half, we have M_{3,2,4}
    # M_{3,2,4} = M_{3,2,2} ⊗ M_{1,1,2}
    # M_{3,2,2}: 3x2 * 2x2 -> 3x2, schoolbook rank 12
    # 
    # Actually, let me try a direct approach for M_{3,4,4}.
    # 
    # The key observation: M_{3,4,4} can be embedded in M_{4,4,4}.
    # If we have a rank-r decomposition of M_{4,4,4}, we can restrict it to get
    # a rank-r decomposition of M_{3,4,4} (just ignore the 4th output row).
    #
    # Best known rank for M_{4,4,4}: 49 (recursive Strassen)
    # But there are better bounds. Let me try to construct a good one.
    #
    # Actually, for practical purposes, let me try a different approach:
    # Use the fact that M_{3,4,4} = sum_{i=0}^{2} M_{1,4,4} (with different i)
    # Each M_{1,4,4} has schoolbook rank 16.
    # So M_{3,4,4} has schoolbook rank 48.
    # 
    # Can we do better for M_{1,4,4}?
    # M_{1,4,4} = M_{1,2,2} ⊗ M_{1,2,2}
    # M_{1,2,2}: 1x2 * 2x2 -> 1x2, rank 4 (schoolbook)
    # So M_{1,4,4} rank = 4*4 = 16. No improvement from tensor product.
    #
    # Let me try a randomized search with the tensor product structure.
    # 
    # Strategy: 
    # 1. Find a good rank-r decomposition of M_{3,4,4}
    # 2. Combine with M_{1,1,2} rank-2 to get rank-2r decomposition of M_{3,4,8}
    
    def verify_decomposition(U, V, W, rank, tol=1e-8):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    best_rank = n * m * p  # 96
    best_result = None
    
    # Schoolbook decomposition as fallback
    def schoolbook_decomp(nn, mm, pp):
        U = np.zeros((nn*mm, nn*mm*pp))
        V = np.zeros((mm*pp, nn*mm*pp))
        W = np.zeros((pp*nn, nn*mm*pp))
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i*mm+j, col] = 1
                    V[j*pp+k, col] = 1
                    W[k*nn+i, col] = 1
                    col += 1
        return U, V, W
    
    # First, try to find a good decomposition of M_{3,4,4}
    # Then combine with M_{1,1,2}
    
    n3, m4, p4 = 3, 4, 4
    best_rank_344 = n3 * m4 * p4  # 48
    best_344 = None
    
    # Try ranks for M_{3,4,4}
    for target_rank in [24, 20, 18, 16, 14, 12, 10, 8]:
        if time.time() - start_time > deadline - 10:
            break
        
        found = False
        for attempt in range(50):
            if time.time() - start_time > deadline - 10:
                break
            
            np.random.seed(attempt * 7919 + target_rank)
            
            # Random initialization
            U = np.random.randint(-2, 3, size=(n3*m4, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m4*p4, target_rank)).astype(float)
            
            # Compute M[i,k,c] = sum_j U[i*m4+j,c] * V[j*p4+k,c]
            M = np.zeros((n3, p4, target_rank))
            for i in range(n3):
                for k in range(p4):
                    for c in range(target_rank):
                        M[i, k, c] = np.dot(U[i*m4:(i+1)*m4, c], V[(0*p4+k):(m4*p4+k):p4, c])
            
            # Solve for W: M[i,k,:] dot W[k*n3+i,:] = 1
            W = np.zeros((p4*n3, target_rank))
            valid = True
            for i in range(n3):
                for k in range(p4):
                    mv = M[i, k, :]
                    norm_sq = np.dot(mv, mv)
                    if norm_sq < 1e-10:
                        valid = False
                        break
                    W[k*n3+i, :] = mv / norm_sq
            if not valid:
                continue
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify
            ok = True
            for i in range(n3):
                for j in range(m4):
                    for k in range(p4):
                        s = np.dot(U_r[i*m4+j, :], V_r[j*p4+k, :] * W_r[k*n3+i, :])
                        if abs(s - 1.0) > 1e-8:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                best_rank_344 = target_rank
                best_344 = (U_r, V_r, W_r)
                found = True
                break
        
        if found:
            break
    
    # If we found a good M_{3,4,4} decomposition, combine with M_{1,1,2}
    if best_344 is not None and best_rank_344 < n3*m4*p4:
        U344, V344, W344 = best_344
        r344 = best_rank_344
        
        # M_{1,1,2} rank-2 decomposition
        U_112 = np.array([[1.0, 1.0]])  # shape (1, 2)
        V_112 = np.array([[1.0, 0.0], [0.0, 1.0]])  # shape (2, 2)
        W_112 = np.array([[1.0, 0.0], [0.0, 1.0]])  # shape (2, 2)
        
        # Tensor product: M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
        # U shape: (3*4, r344*2) = (12, 2*r344)
        # V shape: (4*8, r344*2) = (32, 2*r344)
        # W shape: (8*3, r344*2) = (24, 2*r344)
        
        total_rank = r344 * 2
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        # For each (c1, c2) where c1 in [0,r344), c2 in [0,2):
        # U[i*4+j, c1*2+c2] = U344[i*4+j, c1] * U_112[0, c2]
        # V[j*8+k, c1*2+c2] = V344[j*4+k//2, c1] * V_112[k%2, c2]
        # W[k*3+i, c1*2+c2] = W344[k//2*3+i, c1] * W_112[k%2, c2]
        
        for i in range(n3):
            for j in range(m4):
                for c1 in range(r344):
                    for c2 in range(2):
                        U[i*m4+j, c1*2+c2] = U344[i*m4+j, c1] * U_112[0, c2]
        
        for j in range(m4):
            for k in range(p):
                k4 = k // 2  # which 4-block
                k2 = k % 2   # within the 2-block
                for c1 in range(r344):
                    for c2 in range(2):
                        V[j*p4+k4, c1*2+c2] = V344[j*m4+k4, c1] * V_112[k2, c2]
        
        for k in range(p):
            k4 = k // 2
            k2 = k % 2
            for i in range(n3):
                for c1 in range(r344):
                    for c2 in range(2):
                        W[k*n3+i, c1*2+c2] = W344[k4*n3+i, c1] * W_112[k2, c2]
        
        # Verify
        if verify_decomposition(U, V, W, total_rank):
            best_rank = total_rank
            best_result = (U, V, W)
    
    # If the tensor product approach didn't work or wasn't better, try direct search
    if best_result is None or best_rank >= n*m*p:
        for target_rank in [48, 40, 36, 32, 28, 24, 20, 16, 14, 12, 10, 8]:
            if time.time() - start_time > deadline:
                break
            
            found = False
            for attempt in range(30):
                if time.time() - start_time > deadline:
                    break
                
                np.random.seed(attempt * 1000 + target_rank + 42)
                
                U = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
                V = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
                
                M = np.zeros((n, p, target_rank))
                for i in range(n):
                    for k in range(p):
                        for c in range(target_rank):
                            M[i, k, c] = np.dot(U[i*m:(i+1)*m, c], V[(0*p+k):(m*p+k):p, c])
                
                W = np.zeros((p*n, target_rank))
                valid = True
                for i in range(n):
                    for k in range(p):
                        mv = M[i, k, :]
                        norm_sq = np.dot(mv, mv)
                        if norm_sq < 1e-10:
                            valid = False
                            break
                        W[k*n+i, :] = mv / norm_sq
                if not valid:
                    continue
                
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify_decomposition(U_r, V_r, W_r, target_rank):
                    best_rank = target_rank
                    best_result = (U_r, V_r, W_r)
                    found = True
                    break
            
            if found:
                break
    
    # Fallback to schoolbook
    if best_result is None or best_rank == n*m*p:
        best_rank = n * m * p
        U, V, W = schoolbook_decomp(n, m, p)
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
