import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def _build_strassen_block():
    """
    Build a 2x2x2 Strassen-style decomposition embedded into the 3x4x5 problem.
    
    For the 2x2x2 sub-problem (rows 0,1 of A; cols 0,1 of B; cols 0,1 of C),
    Strassen uses 7 multiplications instead of 8.
    
    The 7 Strassen products for 2x2x2:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B01-B11)
      M4 = A11*(B10-B00)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B01)
      M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # We're building tensors for the full 3x4x5 problem
    n, m, p = 3, 4, 5
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices for the 2x2x2 sub-problem:
    # A rows: 0,1 (i=0,1)
    # A cols / B rows: 0,1 (j=0,1)
    # B cols / C cols: 0,1 (k=0,1)
    # C rows: 0,1 (i=0,1)
    
    # U indices: i*m + j = i*4 + j
    # V indices: j*p + k = j*5 + k
    # W indices: k*n + i = k*3 + i
    
    # M1 = (A00+A11)*(B00+B11)
    # A00: i=0,j=0 -> U[0]; A11: i=1,j=1 -> U[5]
    # B00: j=0,k=0 -> V[0]; B11: j=1,k=1 -> V[6]
    # C00: k=0,i=0 -> W[0]; C11: k=1,i=1 -> W[4]
    U[0, 0] += 1; U[5, 0] += 1
    V[0, 0] += 1; V[6, 0] += 1
    W[0, 0] += 1; W[4, 0] += 1
    
    # M2 = (A10+A11)*B00
    # A10: i=1,j=0 -> U[4]; A11: i=1,j=1 -> U[5]
    # B00: j=0,k=0 -> V[0]
    # C10: k=0,i=1 -> W[1]
    U[4, 1] += 1; U[5, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1
    
    # M3 = A00*(B01-B11)
    # A00: i=0,j=0 -> U[0]
    # B01: j=0,k=1 -> V[1]; B11: j=1,k=1 -> V[6]
    # C01: k=1,i=0 -> W[3]
    U[0, 2] += 1
    V[1, 2] += 1; V[6, 2] += -1
    W[3, 2] += 1
    
    # M4 = A11*(B10-B00)
    # A11: i=1,j=1 -> U[5]
    # B10: j=1,k=0 -> V[5]; B00: j=0,k=0 -> V[0]
    # C00: k=0,i=0 -> W[0]; C10: k=0,i=1 -> W[1]
    U[5, 3] += 1
    V[5, 3] += 1; V[0, 3] += -1
    W[0, 3] += 1; W[1, 3] += 1
    
    # M5 = (A00+A01)*B11
    # A00: i=0,j=0 -> U[0]; A01: i=0,j=1 -> U[1]
    # B11: j=1,k=1 -> V[6]
    # C00: k=0,i=0 -> W[0]; C01: k=1,i=0 -> W[3]
    U[0, 4] += 1; U[1, 4] += 1
    V[6, 4] += 1
    W[0, 4] += -1; W[3, 4] += 1
    
    # M6 = (A10-A00)*(B00+B01)
    # A10: i=1,j=0 -> U[4]; A00: i=0,j=0 -> U[0]
    # B00: j=0,k=0 -> V[0]; B01: j=0,k=1 -> V[1]
    # C11: k=1,i=1 -> W[4]
    U[4, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    W[4, 5] += 1
    
    # M7 = (A01-A11)*(B10+B11)
    # A01: i=0,j=1 -> U[1]; A11: i=1,j=1 -> U[5]
    # B10: j=1,k=0 -> V[5]; B11: j=1,k=1 -> V[6]
    # C00: k=0,i=0 -> W[0]; C11: k=1,i=1 -> W[4]
    U[1, 6] += 1; U[5, 6] += -1
    V[5, 6] += 1; V[6, 6] += 1
    W[0, 6] += 1; W[4, 6] += 1
    
    return U, V, W

def _build_schoolbook_remaining():
    """
    Build schoolbook decomposition for the parts not covered by Strassen block.
    
    The Strassen block covers:
    - A rows 0,1; A cols 0,1; B cols 0,1; C cols 0,1; C rows 0,1
    
    Remaining parts:
    1. A rows 0,1; A cols 2,3; B cols 0,1; C cols 0,1; C rows 0,1
       (4*2*2 = 16 terms)
    2. A rows 0,1; A cols 0,1; B cols 2,3,4; C cols 2,3,4; C rows 0,1
       (2*2*3*2 = 24 terms... wait)
    
    Actually, let me think more carefully. The full problem is 3x4x5.
    The Strassen block handles the (2x2)x(2x2) sub-problem for rows 0,1 of A.
    
    Remaining:
    - For rows 0,1 of A: columns 2,3 of A (which are rows 2,3 of B)
      times all 5 columns of B -> contributes to all 5 columns of C for rows 0,1
      That's 2*2*5 = 20 terms
    - For row 2 of A: all 4 columns of A times all 5 columns of B
      That's 1*4*5 = 20 terms
    
    Total remaining: 20 + 20 = 40 terms
    Total rank: 7 + 40 = 47
    """
    n, m, p = 3, 4, 5
    rank = 40
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Part 1: A rows 0,1; A cols 2,3; B cols 0,1,2,3,4; C cols 0,1,2,3,4; C rows 0,1
    for i in range(2):
        for j in range(2, 4):
            for k in range(5):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
    
    # Part 2: A row 2; A cols 0,1,2,3; B cols 0,1,2,3,4; C cols 0,1,2,3,4; C row 2
    for j in range(4):
        for k in range(5):
            i = 2
            u = i * m + j
            v = j * p + k
            w = k * n + i
            U[u, col] = 1
            V[v, col] = 1
            W[w, col] = 1
            col += 1
    
    return U, V, W

def _combine_and_verify(U1, V1, W1, U2, V2, W2):
    """Combine two decompositions and verify correctness."""
    n, m, p = 3, 4, 5
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    U[:, :rank1] = U1
    V[:, :rank1] = V1
    W[:, :rank1] = W1
    U[:, rank1:] = U2
    V[:, rank1:] = V2
    W[:, rank1:] = W2
    
    # Verify: sum_r U[:,r] * V[:,r]^T * W[:,r]^T should give the multiplication tensor
    # The tensor T[i,j,k] = delta (i.e., T[i,j,k] = 1 if the (i,j) element of A times (j,k) element of B gives (i,k) element of C)
    # More precisely: the bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor for this is: T[i,j,k] = 1 for all valid i,j,k (since each product A[i,j]*B[j,k] contributes to C[i,k])
    
    # Verify by checking: for each (i,k), sum_j A[i,j]*B[j,k] = C[i,k]
    # In tensor form: sum_r U[i,r]*V[j,r]*W[k,r] should equal 1 if j is the connecting index
    # Actually the standard form: the multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # and the decomposition satisfies: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    # Let's verify numerically
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                computed = 0.0
                for r in range(total_rank):
                    computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                error += abs(computed - expected)
    
    return U, V, W, error

def _try_als_refinement(U, V, W, max_iters=200, time_budget=30):
    """
    Try ALS (Alternating Least Squares) to reduce the effective rank
    by finding a lower-rank approximation, then check if it's exact.
    """
    n, m, p = 3, 4, 5
    current_rank = U.shape[1]
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to reduce rank by 1, 2, ...
    for new_rank in range(current_rank - 1, max(50, current_rank - 10), -1):
        if time.time() - _start_time > time_budget:
            break
            
        # Initialize with random projection
        np.random.seed(42)
        U_r = np.random.randn(n*m, new_rank)
        V_r = np.random.randn(m*p, new_rank)
        W_r = np.random.randn(p*n, new_rank)
        
        # Normalize
        for r in range(new_rank):
            n_u = np.linalg.norm(U_r[:, r])
            n_v = np.linalg.norm(V_r[:, r])
            n_w = np.linalg.norm(W_r[:, r])
            if n_u > 1e-10 and n_v > 1e-10 and n_w > 1e-10:
                scale = (n_u * n_v * n_w) ** (1/3)
                U_r[:, r] /= n_u * scale
                V_r[:, r] /= n_v * scale
                W_r[:, r] /= n_w * scale
        
        best_error = float('inf')
        for it in range(max_iters):
            # Update U
            for i in range(n*m):
                # U[i, r] = T[i, :, :] contracted with V[:, r] and W[:, r]
                # T[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
                # Fix V and W, solve for U[i, :]
                # T[i, :, :] = sum_r U[i,r] outer(V[:,r], W[:,r])
                # This is a linear system: sum_r U[i,r] * V[:,r] * W[:,r]^T = T[i,:,:]
                # Reshape: for each r, the contribution is U[i,r] * (V[:,r] * W[:,r]^T)
                # So we need: sum_r U[i,r] * vec(V[:,r] * W[:,r]^T) = vec(T[i,:,:])
                
                # Build the system
                A_sys = np.zeros((m*p*p*n, new_rank))
                for r in range(new_rank):
                    A_sys[:, r] = np.outer(V_r[:, r], W_r[:, r]).reshape(-1)
                b = T[i, :, :].reshape(-1)
                
                sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b, rcond=None)
                U_r[i, :] = sol
            
            # Update V
            for j in range(m*p):
                A_sys = np.zeros((n*m*p*n, new_rank))
                for r in range(new_rank):
                    A_sys[:, r] = np.outer(U_r[:, r], W_r[:, r]).reshape(-1)
                b = np.zeros(n*m*p*n)
                for i in range(n):
                    for k in range(p):
                        b[(i*m + j)*p*n + k*n + i] = T[i*m+j, j*p+k, k*n+i]
                
                sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b, rcond=None)
                V_r[j, :] = sol
            
            # Update W
            for k in range(p*n):
                A_sys = np.zeros((n*m*m*p, new_rank))
                for r in range(new_rank):
                    A_sys[:, r] = np.outer(U_r[:, r], V_r[:, r]).reshape(-1)
                b = np.zeros(n*m*m*p)
                for i in range(n):
                    for j in range(m):
                        b[i*m*m*p + j*p + k] = T[i*m+j, j*p+k, k*n+i]
                
                sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b, rcond=None)
                W_r[k, :] = sol
            
            # Compute error
            err = 0.0
            for i in range(n*m):
                for j in range(m*p):
                    for kk in range(p*n):
                        expected = T[i, j, kk]
                        computed = np.dot(U_r[i, :], V_r[j, :] * W_r[kk, :])
                        err += (computed - expected) ** 2
            
            if err < best_error:
                best_error = err
                if err < 1e-10:
                    # Found an exact decomposition!
                    return U_r, V_r, W_r
            if time.time() - _start_time > time_budget:
                break
        
        if best_error < 1e-10:
            return U_r, V_r, W_r
    
    return None

def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the structured decomposition: Strassen block + schoolbook
    U1, V1, W1 = _build_strassen_block()
    U2, V2, W2 = _build_schoolbook_remaining()
    
    U, V, W, error = _combine_and_verify(U1, V1, W1, U2, V2, W2)
    
    if error > 1e-10:
        # Fallback to pure schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    current_rank = U.shape[1]
    
    # Try to refine with ALS if we have time
    if time.time() < deadline - 40:
        refined = _try_als_refinement(U, V, W, max_iters=100, time_budget=40)
        if refined is not None:
            U, V, W = refined
    
    # Final check
    if time.time() < deadline - 10:
        # Verify final result
        final_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    computed = 0.0
                    for r in range(U.shape[1]):
                        computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    final_error += abs(computed - expected)
        
        if final_error > 1e-10:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        U[u, col] = 1
                        V[v, col] = 1
                        W[w, col] = 1
                        col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
