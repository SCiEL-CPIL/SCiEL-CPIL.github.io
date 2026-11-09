import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 115  # 150s limit, keep 35s safety margin
    
    # Schoolbook decomposition (rank 54)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_round_and_verify(U, V, W, tol=1e-9):
        """Round to nearest 0.5 and verify"""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, tol):
            return U_r, V_r, W_r
        # Try rounding to integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify(U_i, V_i, W_i, tol):
            return U_i, V_i, W_i
        return None
    
    def compute_tensor(U, V, W):
        """Compute T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        U3 = U.reshape(n, m, U.shape[1])
        V3 = V.reshape(m, p, V.shape[1])
        W3 = W.reshape(p, n, W.shape[1])
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
        return T
    
    def gradient_descent(U, V, W, max_steps=2000, lr=0.05, tol=1e-10):
        """Gradient descent to minimize sum (T[i,j,k]-1)^2"""
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            T = compute_tensor(U, V, W)
            error = T - 1.0
            if np.max(np.abs(error)) < tol:
                return U, V, W
            
            U3 = U.reshape(n, m, U.shape[1])
            V3 = V.reshape(m, p, V.shape[1])
            W3 = W.reshape(p, n, W.shape[1])
            r = U.shape[1]
            
            grad_U = np.zeros((n, m, r))
            grad_V = np.zeros((m, p, r))
            grad_W = np.zeros((p, n, r))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = error[i,j,k]
                        if abs(e) > 1e-15:
                            grad_U[i,j,:] += 2 * e * V3[j,k,:] * W3[k,i,:]
                            grad_V[j,k,:] += 2 * e * U3[i,j,:] * W3[k,i,:]
                            grad_W[k,i,:] += 2 * e * U3[i,j,:] * V3[j,k,:]
            
            U = U - lr * grad_U.reshape(n*m, r)
            V = V - lr * grad_V.reshape(m*p, r)
            W = W - W - 0  # placeholder, fix below
            W = W - lr * grad_W.reshape(p*n, r)
        
        return U, V, W
    
    # Strategy: Use block structure. The 2x3x9 multiplication can be seen as
    # multiplying a 2x3 matrix by three 3x3 blocks (columns 0-2, 3-5, 6-8).
    # If we have a rank-r decomposition for 2x3x3, we can use rank 3*r for 2x3x9.
    # Known: rank of 2x3x3 is at most 11 (from literature).
    # So rank 33 should be achievable.
    
    # First, try to find a rank-11 decomposition for 2x3x3
    def find_2x3x3_decomposition(max_rank=11, time_budget=30):
        """Search for a low-rank decomposition of 2x3x3 matrix multiplication"""
        n2, m2, p2 = 2, 3, 3
        best_r = None
        best_U2 = None
        best_V2 = None
        best_W2 = None
        
        for r in range(max_rank, 7, -1):  # Try ranks 11 down to 8
            if time.time() > deadline - time_budget:
                break
            for restart in range(50):
                if time.time() > deadline - time_budget:
                    break
                U2 = np.random.randn(n2*m2, r)
                V2 = np.random.randn(m2*p2, r)
                W2 = np.random.randn(p2*n2, r)
                
                # Gradient descent
                U2, V2, W2 = gradient_descent(U2, V2, W2, max_steps=3000, lr=0.02)
                
                # Try rounding
                result = try_round_and_verify(U2, V2, W2)
                if result is not None:
                    U2_r, V2_r, W2_r = result
                    # Verify for 2x3x3
                    U3 = U2_r.reshape(n2, m2, r)
                    V3 = V2_r.reshape(m2, p2, r)
                    W3 = W2_r.reshape(p2, n2, r)
                    valid = True
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                val = np.dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        best_r = r
                        best_U2 = U2_r
                        best_V2 = V2_r
                        best_W2 = W2_r
                        break  # Found for this rank, try lower
            if best_r is not None and best_r <= 10:
                break
        
        return best_r, best_U2, best_V2, best_W2
    
    # Try to find 2x3x3 decomposition
    r2, U2, V2, W2 = find_2x3x3_decomposition(max_rank=11, time_budget=25)
    
    if r2 is not None:
        # Construct 2x3x9 decomposition using block structure
        # The 3x9 matrix B has three 3x3 blocks: B[:,0:3], B[:,3:6], B[:,6:9]
        # C[:,0:3] = A * B[:,0:3], C[:,3:6] = A * B[:,3:6], C[:,6:9] = A * B[:,6:9]
        # So we need three copies of the 2x3x3 decomposition.
        r_total = 3 * r2
        U_total = np.zeros((n*m, r_total))
        V_total = np.zeros((m*p, r_total))
        W_total = np.zeros((p*n, r_total))
        
        for block in range(3):
            offset = block * r2
            # U is the same for all blocks (A is the same)
            U_total[:, offset:offset+r2] = U2
            # V: block b uses columns b*3 to (b+1)*3-1 of B
            # V2 has shape (m2*p2, r2) = (9, r2)
            # For block b, V[j, k] where k is in [b*3, (b+1)*3)
            # V_total[j*p + k, offset:offset+r2] = V2[j*p2 + (k - b*3), :]
            for j in range(m):
                for k in range(p):
                    k_local = k - block * 3
                    V_total[j*p + k, offset:offset+r2] = V2[j*3 + k_local, :]
            # W: W[k*n + i, :] for output C[i,k]
            # For block b, k is in [b*3, (b+1)*3)
            # W2 has shape (p2*n2, r2) = (6, r2)
            # W_total[k*n + i, offset:offset+r2] = W2[(k - b*3)*n2 + i, :]
            for i in range(n):
                for k in range(p):
                    k_local = k - block * 3
                    W_total[k*n + i, offset:offset+r2] = W2[k_local*n + i, :]
        
        # Verify the full decomposition
        if verify(U_total, V_total, W_total):
            best_U = U_total
            best_V = V_total
            best_W = W_total
            best_rank = r_total
    
    # If we didn't find a good 2x3x3, try direct search for 2x3x9 with lower rank
    if best_rank == 54:
        # Try direct gradient descent for ranks 45, 42, 39, 36
        for target_rank in [45, 42, 39, 36]:
            if time.time() > deadline - 10:
                break
            for restart in range(20):
                if time.time() > deadline - 5:
                    break
                U = np.random.randn(n*m, target_rank)
                V = np.random.randn(m*p, target_rank)
                W = np.random.randn(p*n, target_rank)
                
                U, V, W = gradient_descent(U, V, W, max_steps=1500, lr=0.01)
                
                result = try_round_and_verify(U, V, W)
                if result is not None:
                    U_r, V_r, W_r = result
                    if verify(U_r, V_r, W_r):
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        best_rank = target_rank
                        break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
