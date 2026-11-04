import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # Schoolbook solution: rank = n*m*p = 54
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U, V, W, best_rank = schoolbook()
    
    # Efficient verification using the tensor structure
    # Constraint: for all (i,j,j2,k2):
    #   sum_r U[i*m+j, r] * V[j2*p+k2, r] * W[k*n+i, r] 
    #     = 1 if (j==j2 and k2==j) else 0, for all i,k
    # Equivalently: for each (j,j2,k2), the matrix M[i,k] = sum_r U[i*m+j,r]*V[j2*p+k2,r]*W[k*n+i,r]
    #   should be delta(j,j2)*delta(k2,j) * I_n (identity in i, all k same)
    
    def check_solution(U_mat, V_mat, W_mat, tol=1e-6):
        rank = U_mat.shape[1]
        # Precompute W as (p, n, rank) for efficiency
        W_3d = W_mat.reshape(p, n, rank)
        
        for j in range(m):
            for j2 in range(m):
                for k2 in range(p):
                    # For each i, compute M[i,k] = sum_r U[i*m+j,r] * V[j2*p+k2,r] * W_3d[k,i,r]
                    u_col = U_mat[:, j]  # shape (n, rank) -- rows are i*m+j for fixed j? No.
                    # U has shape (n*m, rank). Row index i*m+j.
                    # For fixed j, the rows are 0*m+j, 1*m+j, ... = j, m+j, 2m+j
                    u_rows = U_mat[j::m, :]  # shape (n, rank), row i is U[i*m+j, :]
                    v_row = V_mat[j2*p + k2, :]  # shape (rank,)
                    
                    # M[i,k] = sum_r u_rows[i,r] * v_row[r] * W_3d[k,i,r]
                    # This is not a simple matrix product because W_3d[k,i,r] couples i and k.
                    
                    # Let's compute it directly
                    M = np.zeros((n, p))
                    for i in range(n):
                        # M[i,k] = sum_r u_rows[i,r] * v_row[r] * W_3d[k,i,r]
                        # = sum_r (u_rows[i,r] * v_row[r]) * W_3d[k,i,r]
                        w_i = u_rows[i] * v_row  # shape (rank,)
                        M[i] = (W_3d[:, i, :] * w_i).sum(axis=1)  # shape (p,)
                    
                    expected = 1.0 if (j == j2 and k2 == j) else 0.0
                    if expected == 0:
                        if np.any(np.abs(M) > tol):
                            return False
                    else:
                        # M[i,k] should be 1 if i==i2 for the correct i2... 
                        # Wait, re-reading the constraint:
                        # sum_r W[k*n+i, r] * U[i2*m+j, r] * V[j2*p+k2, r] = delta(i,i2)*delta(j,j2)*delta(k2,j)
                        # So for fixed (j,j2,k2), and varying (i,k):
                        # The value is 1 only when i=i2 AND j=j2 AND k2=j
                        # But i2 is the row index in U, which is fixed in our loop? 
                        # No, in the check we fix j (from U's column) and j2,k2 (from V's row)
                        # and we need to verify for ALL i (which is i2 in the original formula) and all k.
                        # 
                        # Actually, let me re-derive. The original constraint from the seed:
                        # sum_r W[k*n+i, r] * U[i2*m+j, r] * V[j2*p+k2, r] = delta(i,i2)*delta(j,j2)*delta(k2,j)
                        # 
                        # So for fixed (j, j2, k2), we compute for each (i, k):
                        #   S(i,k) = sum_r U[i2*m+j, r] * V[j2*p+k2, r] * W[k*n+i, r]
                        # Wait, i2 appears in U but i appears in W. These are different indices!
                        # 
                        # Let me re-read the seed's check_solution more carefully:
                        # "for i2 in range(n): for j in range(m): for j2 in range(m): for k2 in range(p):"
                        #   u_row = U_mat[i2*m+j, :]
                        #   v_row = V_mat[j2*p+k2, :]
                        #   M = (weights @ W_mat.T).reshape(p, n).T  # M[i,k]
                        #   expected_val = 1.0 if (j == j2 and k2 == j) else 0.0
                        #   if expected_val == 0: check M is all zeros
                        #   else: M[i,:] should be 1 if i==i2, 0 otherwise
                        #
                        # So the constraint is: for each (i2, j, j2, k2):
                        #   M[i,k] = sum_r U[i2*m+j,r] * V[j2*p+k2,r] * W[k*n+i,r]
                        #   M[i,k] = 1 if (i==i2 and j==j2 and k2==j) else 0
                        #
                        # This means: for fixed (j,j2,k2), the matrix M (size n x p) should have:
                        #   M[i,k] = delta(i,i2)*delta(j,j2)*delta(k2,j) for the specific i2 being checked
                        #   But i2 is a free variable in the outer loop!
                        #
                        # So actually, for fixed (j,j2,k2), we need:
                        #   For ALL i2: M[i2,k] = delta(j,j2)*delta(k2,j) for all k, and M[i,k]=0 for i!=i2
                        #   This means: M is a diagonal-like matrix where only row i2 is nonzero,
                        #   and that row is all delta(j,j2)*delta(k2=j).
                        #
                        # But wait, this must hold for ALL i2 simultaneously? That can't be right.
                        # Let me re-read: the loop is over i2, j, j2, k2. For each combination,
                        # we check a different constraint. So for a fixed (j,j2,k2), we have n different
                        # constraints (one for each i2), each saying:
                        #   "If we pick U's row i2*m+j, then the resulting M has 1s in row i2 (if j=j2,k2=j)
                        #    and 0s elsewhere."
                        #
                        # This is equivalent to saying: the rows of U (indexed by i2 for fixed j) 
                        # produce orthogonal "responses" in the W space.
                        #
                        # Let me just implement it as in the seed for correctness.
                        pass
        
        # Let me just use the seed's approach but vectorized
        for i2 in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k2 in range(p):
                        u_row = U_mat[i2*m+j, :]
                        v_row = V_mat[j2*p+k2, :]
                        weights = u_row * v_row  # shape (rank,)
                        # M[i,k] = sum_r weights[r] * W[k*n+i, r]
                        M = (weights @ W_mat.T).reshape(p, n).T  # shape (n, p)
                        
                        expected_val = 1.0 if (j == j2 and k2 == j) else 0.0
                        if expected_val == 0:
                            if np.any(np.abs(M) > tol):
                                return False
                        else:
                            for i in range(n):
                                if i == i2:
                                    if np.any(np.abs(M[i, :] - 1.0) > tol):
                                        return False
                                else:
                                    if np.any(np.abs(M[i, :]) > tol):
                                        return False
        return True
    
    # Verify schoolbook
    if not check_solution(U, V, W):
        raise ValueError("Schoolbook solution failed verification")
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 27, 6) = 27
    lower_bound = max(n*p, m*p, n*m)
    
    # Strategy: Try to find rank-27 solution using structured initialization
    # and gradient-based optimization
    
    def try_structured_search(target_rank, time_budget=60):
        """Try to find a rank-target_rank solution using structured initialization."""
        deadline_local = start_time + time_budget
        
        # Key insight: For rank 27 = m*p, we can try a "V-dominant" structure
        # where V is close to identity-like and U,W are derived accordingly.
        
        # Attempt 1: Initialize V with a structured pattern
        # For each (j2, k2), we want V[j2*p+k2, :] to be distinguishable
        # Let's try: V[j2*p+k2, r] = 1 if r == j2*p+k2 else 0 (identity-like)
        # Then U[i*m+j, r] and W[k*n+i, r] need to satisfy:
        # sum_r U[i2*m+j, r] * delta(r, j2*p+k2) * W[k*n+i, r]
        #   = U[i2*m+j, j2*p+k2] * W[k*n+i, j2*p+k2]
        # should equal delta(i,i2)*delta(j,j2)*delta(k2,j)
        
        # So we need: U[i2*m+j, j2*p+k2] * W[k*n+i, j2*p+k2] = delta(i,i2)*delta(j,j2)*delta(k2,j)
        # 
        # For this to work:
        # If j != j2 or k2 != j: U[i2*m+j, j2*p+k2] * W[k*n+i, j2*p+k2] = 0 for all i,i2
        #   => Either U[i2*m+j, j2*p+k2] = 0 for all i2, or W[k*n+i, j2*p+k2] = 0 for all i
        # If j == j2 and k2 == j: U[i2*m+j, j*p+j] * W[k*n+i, j*p+j] = delta(i,i2)
        #   => U[i2*m+j, j*p+j] * W[k*n+i, j*p+j] = delta(i,i2) for all i,i2,k
        #   This requires: U[i2*m+j, j*p+j] * W[k*n+i, j*p+j] = delta(i,i2)
        #   For this to hold for all k, W[k*n+i, j*p+j] must be independent of k,
        #   and U[i2*m+j, j*p+j] * W[i, j*p+j] = delta(i,i2)
        #   This is like: U_col * W_row^T = I, which means U_col and W_row must be
        #   inverses in some sense. But they're vectors, not matrices.
        #   U[i2*m+j, j*p+j] * W[k*n+i, j*p+j] = delta(i,i2)
        #   Let a_i = U[i*m+j, j*p+j] and b_i = W[k*n+i, j*p+j] (independent of k)
        #   Then a_i2 * b_i = delta(i,i2) for all i,i2
        #   This means: a_i2 * b_i = 1 if i==i2, 0 otherwise
        #   For i=i2: a_i * b_i = 1
        #   For i!=i2: a_i2 * b_i = 0
        #   From i!=i2: either a_i2=0 or b_i=0. But from i=i2: a_i*b_i=1, so neither is zero.
        #   Contradiction for n>1! So this simple structure doesn't work.
        
        # Attempt 2: Use a different initialization strategy
        # Let's try random initialization with gradient descent on the constraint violation
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline_local:
                break
            
            # Initialize with random values scaled appropriately
            U_try = np.random.randn(n*m, target_rank) * 0.5
            V_try = np.random.randn(m*p, target_rank) * 0.5
            W_try = np.random.randn(p*n, target_rank) * 0.5
            
            # Gradient descent with learning rate schedule
            lr = 0.01
            for it in range(500):
                if time.time() > deadline_local:
                    break
                
                # Compute error: sum of squared violations
                error = 0.0
                grad_U = np.zeros_like(U_try)
                grad_V = np.zeros_like(V_try)
                grad_W = np.zeros_like(W_try)
                
                # For each (i2, j, j2, k2):
                # M[i,k] = sum_r U[i2*m+j,r] * V[j2*p+k2,r] * W[k*n+i,r]
                # Expected: delta(i,i2)*delta(j,j2)*delta(k2,j)
                
                for i2 in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                u_row = U_try[i2*m+j, :]
                                v_row = V_try[j2*p+k2, :]
                                weights = u_row * v_row
                                M = (weights @ W_try.T).reshape(p, n).T
                                
                                expected_val = 1.0 if (j == j2 and k2 == j) else 0.0
                                if expected_val == 0:
                                    residual = M
                                else:
                                    expected_M = np.zeros((n, p))
                                    expected_M[i2, :] = 1.0
                                    residual = M - expected_M
                                
                                err = np.sum(residual**2)
                                error += err
                                
                                # Gradients
                                # dM[i,k]/dU[i2*m+j,r] = V[j2*p+k2,r] * W[k*n+i,r]
                                # dM[i,k]/dV[j2*p+k2,r] = U[i2*m+j,r] * W[k*n+i,r]
                                # dM[i,k]/dW[k*n+i,r] = U[i2*m+j,r] * V[j2*p+k2,r]
                                
                                # grad_U[i2*m+j, r] += 2 * sum_{i,k} residual[i,k] * V[j2*p+k2,r] * W[k*n+i,r]
                                # = 2 * V[j2*p+k2,r] * sum_{i,k} residual[i,k] * W[k*n+i,r]
                                # 
                                # Let's compute this more carefully
                                # For each r:
                                # grad_U[i2*m+j, r] = 2 * V[j2*p+k2,r] * sum_{i,k} residual[i,k] * W[k*n+i,r]
                                
                                for r in range(target_rank):
                                    # sum_{i,k} residual[i,k] * W[k*n+i,r]
                                    w_col = W_try[:, r].reshape(p, n)  # W[k*n+i, r] -> w_col[k,i]
                                    s = np.sum(residual * w_col.T)  # sum_{i,k} residual[i,k] * w_col[k,i]
                                    grad_U[i2*m+j, r] += 2 * v_row[r] * s
                                    grad_V[j2*p+k2, r] += 2 * u_row[r] * s
                                    # For W:
                                    # grad_W[k*n+i, r] += 2 * residual[i,k] * u_row[r] * v_row[r]
                                    grad_W[:, r] += 2 * (u_row[r] * v_row[r]) * residual.reshape(p*n)
                                
                                if error > 1e6:
                                    break
                        if error > 1e6:
                            break
                    if error > 1e6:
                        break
                if error > 1e6:
                    break
                
                # Update
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W
                
                # Decay learning rate
                lr *= 0.99
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                
                if error < 1e-8:
                    break
        
        # Check if we found a valid solution
        if best_U is not None and best_error < 1e-6:
            # Round to half-integers
            U_h = np.round(best_U * 2) / 2
            V_h = np.round(best_V * 2) / 2
            W_h = np.round(best_W * 2) / 2
            if check_solution(U_h, V_h, W_h):
                return U_h, V_h, W_h
            
            # Try with smaller tolerance
            if check_solution(U_h, V_h, W_h, tol=1e-4):
                return U_h, V_h, W_h
        
        return None
    
    # Try rank 27 first
    if time.time() < deadline - 40:
        result = try_structured_search(27, time_budget=50)
        if result is not None:
            U, V, W = result
            best_rank = 27
    
    # If rank 27 didn't work, try a more targeted approach
    # Use the fact that the tensor has symmetry
    
    # Alternative: try to use a known low-rank decomposition structure
    # For matrix multiplication, the rank is related to the bilinear complexity.
    # For 2x3x9, the lower bound is 27. Let's try a few more structured approaches.
    
    if best_rank > 27 and time.time() < deadline - 30:
        # Try a different initialization: use the structure of the problem
        # Group the 9 columns of B into 3 groups of 3
        # For each group t, the relevant subproblem is 2x3x3
        
        # Actually, let's try a direct construction for rank 27
        # Based on the idea that V can be "diagonal" in some basis
        
        U_try = np.zeros((n*m, 27))
        V_try = np.zeros((m*p, 27))
        W_try = np.zeros((p*n, 27))
        
        # Assign each rank r to a specific (j, k) pair where j in [0,3), k in [0,9)
        # But we only have 27 ranks, and there are 3*9=27 (j,k) pairs
        # So rank r corresponds to (j, k) where r = j*9 + k
        
        for j in range(m):
            for k in range(p):
                r = j * p + k  # rank index
                # For this rank, we want:
                # U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i', r] 
                #   contributes to the constraint
                # 
                # The constraint requires:
                # sum_r U[i2*m+j, r] * V[j2*p+k2, r] * W[k*n+i, r] 
                #   = delta(i,i2)*delta(j,j2)*delta(k2,j)
                #
                # If we make V[j2*p+k2, r] = delta(j2, j)*delta(k2, k) (i.e., V is identity-like),
                # then only r = j2*p+k2 contributes, and:
                # U[i2*m+j2, j2*p+k2] * W[k*n+i, j2*p+k2] = delta(i,i2)*delta(j2,j2)*delta(k2,j2)
                # = delta(i,i2)*delta(k2,j2)
                #
                # So for r = j2*p+k2:
                # U[i2*m+j2, j2*p+k2] * W[k*n+i, j2*p+k2] = delta(i,i2)*delta(k2,j2)
                #
                # For k2 != j2: U[i2*m+j2, j2*p+k2] * W[k*n+i, j2*p+k2] = 0
                # For k2 == j2: U[i2*m+j2, j2*p+j2] * W[k*n+i, j2*p+j2] = delta(i,i2)
                #
                # This has the same issue as before. Let's try a different V structure.
                
                pass
        
        # Let me try yet another approach: use the bilinear structure more directly
        # The matrix multiplication tensor for n x m x p has a natural rank of n*m*p.
        # To reduce rank, we need to find linear dependencies.
        
        # For now, let's just return the best we have
        pass
    
    # Ensure we return valid half-integer matrices
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Final verification
    if not check_solution(U, V, W, tol=1e-4):
        # Fall back to schoolbook
        U, V, W, _ = schoolbook()
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
