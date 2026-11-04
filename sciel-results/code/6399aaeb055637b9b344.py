import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U, V, W, best_rank = schoolbook()
    
    # The constraint for matrix multiplication tensor:
    # For multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where T[i,j,k] = delta(j,j) which is always 1... 
    # 
    # Actually, the matrix multiplication tensor T is defined as:
    # T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # 
    # Wait, let me reconsider. The standard matrix multiplication tensor has:
    # T_{i,j,k} = 1 if the product A_{i,j} * B_{j,k} contributes to C_{i,k}
    # This means T_{i,j,k} = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k])
    # 
    # The decomposition is: T_{i,j,k} = sum_r U_{i,r} * V_{j,r} * W_{k,r}
    # But wait, the problem says shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # 
    # Let me re-read: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # 
    # So the tensor is indexed as:
    # T[i*m+j, j'*p+k, k'*n+i'] ... hmm this doesn't make sense directly.
    # 
    # Actually, looking at the check_solution in the original code:
    # W[k*n+i, r] * U[i2*m+j, r] * V[j2*p+k2, r]
    # expected = 1.0 if (i==i2 and j==j2 and k2==j) else 0.0
    # 
    # So the constraint is:
    # sum_r W[k*n+i, r] * U[i2*m+j, r] * V[j2*p+k2, r] = delta(i,i2) * delta(j,j2) * delta(k2,j)
    # 
    # This means: the "tensor" T has indices (k,i) for W, (i2,j) for U, (j2,k2) for V
    # And T[(k,i),(i2,j),(j2,k2)] = 1 if i=i2, j=j2, k2=j, else 0
    # 
    # So T is a 6-index tensor that is 1 only when i=i2, j=j2, and k2=j.
    # This is essentially the matrix multiplication structure where:
    # - W is indexed by (output_row, output_col) = (i, k) but stored as k*n+i
    # - U is indexed by (input_row, input_col) = (i2, j) stored as i2*m+j  
    # - V is indexed by (input_col, input_row_of_B) = (j2, k2) stored as j2*p+k2
    # 
    # The condition k2==j means: the column index of B must equal the row index of A
    # This is exactly the matrix multiplication constraint!
    
    def check_solution(U_mat, V_mat, W_mat):
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        # Check all constraints efficiently using matrix operations
        # For each (i2, j, j2, k2), compute sum_r W[k*n+i, r] * U[i2*m+j, r] * V[j2*p+k2, r]
        # This should equal delta(i,i2)*delta(j,j2)*delta(k2,j) for all i,k
        
        # Let's do this by constructing the full tensor and checking
        # For efficiency, we can use the fact that this is a bilinear form
        
        # Check: for all i2,j,j2,k2:
        # The matrix M_{i,k} = sum_r U[i2*m+j, r] * V[j2*p+k2, r] * W[k*n+i, r]
        # should equal delta(j,j2)*delta(k2,j) * I_n (identity in i)
        
        for i2 in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k2 in range(p):
                        # Compute the n x p matrix of values
                        # M[i,k] = sum_r U[i2*m+j, r] * V[j2*p+k2, r] * W[k*n+i, r]
                        u_row = U_mat[i2*m+j, :]  # shape (rank,)
                        v_row = V_mat[j2*p+k2, :]  # shape (rank,)
                        # W is (p*n, rank), so W[k*n+i, r] for all k,i
                        # We need to form the matrix where entry (i,k) = sum_r u_row[r] * v_row[r] * W[k*n+i, r]
                        # This is like an outer product weighted by W
                        # 
                        # Let's compute: for each r, contribution is u_row[r]*v_row[r] * W[:, r]
                        # W[:, r] has entries W[k*n+i, r] for all (k,i)
                        # So the matrix is sum_r (u_row[r]*v_row[r]) * reshape(W[:, r], (p, n)).T
                        # i.e., reshape W[:, r] to (p, n) and transpose to (n, p)
                        
                        # More directly:
                        # M = sum_r (u_row[r] * v_row[r]) * W_col_r reshaped
                        weights = u_row * v_row  # shape (rank,)
                        # W is (p*n, rank), so W.T is (rank, p*n)
                        # M[i,k] = sum_r weights[r] * W[k*n+i, r] = sum_r weights[r] * W.T[r, k*n+i]
                        # So M = weights @ W.T, reshaped to (p, n), then transposed to (n, p)
                        M = (weights @ W_mat.T).reshape(p, n).T  # shape (n, p)
                        
                        expected_val = 1.0 if (j == j2 and k2 == j) else 0.0
                        # M should be expected_val * I_n... but M is (n, p), not (n, n)
                        # Wait, the constraint says delta(i,i2)*delta(j,j2)*delta(k2,j)
                        # So for fixed i2,j,j2,k2, M[i,k] should be 1 if i=i2 and j=j2 and k2=j, else 0
                        # But M is (n, p), so it's a matrix, not a scalar.
                        
                        # Let me re-read: the sum is over r of W[k*n+i, r] * U[i2*m+j, r] * V[j2*p+k2, r]
                        # This gives a value for each (i, k) pair (since W is indexed by k*n+i)
                        # And this should equal delta(i,i2)*delta(j,j2)*delta(k2,j)
                        
                        # So for each (i,k): value = delta(i,i2)*delta(j,j2)*delta(k2,j)
                        # This is a scalar that depends on i through delta(i,i2)
                        # So M[i,k] = delta(i,i2) * delta(j,j2) * delta(k2,j) for all k
                        # This means M is a matrix where row i2 is all delta(j,j2)*delta(k2,j) and other rows are 0
                        
                        if expected_val == 0:
                            if np.any(np.abs(M) > 1e-6):
                                return False
                        else:
                            # M should have 1 in row i2, all columns, and 0 elsewhere
                            for i in range(n):
                                if i == i2:
                                    if np.any(np.abs(M[i, :] - 1.0) > 1e-6):
                                        return False
                                else:
                                    if np.any(np.abs(M[i, :]) > 1e-6):
                                        return False
        return True
    
    # Now let's try to find a lower rank solution using the Kronecker product structure
    # The insight: 2x3x9 matrix multiplication
    # We can view B as three 3x3 blocks (since p=9=3*3)
    # Let B = [B1, B2, B3] where each Bt is 3x3
    # Then C = A * B = [A*B1, A*B2, A*B3]
    # Each A*Bt is 2x3
    # 
    # If we had a rank-r decomposition for 2x3x3 matrix multiplication,
    # we could potentially reuse it for each block.
    # 
    # For 2x3x3, the schoolbook rank is 18.
    # But maybe we can do better.
    
    # Actually, let's try a different approach: use the structure to build a rank-27 solution
    # by treating the 9 columns of B as three independent 3-column groups.
    
    def try_kronecker_decomposition():
        """
        Try to build a decomposition using the block structure.
        View B as having 3 blocks of 3 columns each.
        For each block t (t=0,1,2), we need to compute A * B_t where B_t is 3x3.
        
        If we use a rank-r decomposition for 2x3x3, the total rank would be 3*r.
        But we need to be careful about how the indices work.
        """
        # For 2x3x3 matrix multiplication:
        # U has shape (2*3, r) = (6, r)
        # V has shape (3*3, r) = (9, r)  
        # W has shape (3*2, r) = (6, r)
        # 
        # The constraint: sum_r W[k*2+i, r] * U[i2*3+j, r] * V[j2*3+k2, r] = delta(i,i2)*delta(j,j2)*delta(k2,j)
        # for i,i2 in [0,2), j,j2 in [0,3), k,k2 in [0,3)
        
        # Schoolbook for 2x3x3 has rank 18.
        # Let's see if we can find something better, or just use the structure.
        
        # Actually, let's just try to directly search for a low-rank solution for the full problem.
        # Use a more systematic approach.
        
        # Lower bound: max(n*p, m*p, n*m) = max(18, 27, 6) = 27
        # So rank must be at least 27.
        
        # Let's try rank 27 first.
        target_rank = 27
        
        # Initialize with a structured guess based on the block decomposition
        U_try = np.zeros((n*m, target_rank))
        V_try = np.zeros((m*p, target_rank))
        W_try = np.zeros((p*n, target_rank))
        
        # For each block t in 0,1,2 (corresponding to columns 3t, 3t+1, 3t+2 of B):
        # We allocate 9 ranks for each block (3*3 schoolbook for 2x3x3)
        # Total: 3*9 = 27
        
        for t in range(3):
            # For block t, the relevant part of B is columns 3t to 3t+2
            # The constraint for this block:
            # For i in [0,2), j in [0,3), k in [0,3) (local k within the block):
            # The global k is 3t + k
            # 
            # U[i*m+j, r] should be nonzero only for ranks allocated to block t
            # V[j*p + (3t+k), r] should be nonzero only for ranks allocated to block t
            # W[(3t+k)*n + i, r] should be nonzero only for ranks allocated to block t
            
            block_start = t * 9
            for i in range(n):
                for j in range(m):
                    for k in range(3):
                        r = block_start + i * m * 3 + j * 3 + k  # 0 to 8 within block
                        # Wait, we need 9 ranks per block for 2x3x3 schoolbook
                        # Schoolbook for 2x3x3: rank = 2*3*3 = 18, not 9
                        # So we need 18 ranks per block, total 54. That's not better.
                        
                        # Let me reconsider. Maybe we can share some structure.
                        pass
        
        # This approach gives rank 54, which is worse than schoolbook.
        # Let's try a different strategy.
        
        return None
    
    # Let's try a more direct search approach with better initialization
    # and using the constraint structure more effectively.
    
    def random_search_with_constraints(target_rank, max_iterations=1000):
        """
        Random search with constraint satisfaction.
        """
        best_solution = None
        best_error = float('inf')
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random values
            U_try = np.random.randn(n*m, target_rank) * 0.1
            V_try = np.random.randn(m*p, target_rank) * 0.1
            W_try = np.random.randn(p*n, target_rank) * 0.1
            
            # Gradient descent on the constraint violation
            for it in range(max_iterations):
                if time.time() > deadline - 10:
                    break
                
                # Compute the error and gradient
                # This is expensive, so we'll do a simplified version
                
                # Check if solution is valid
                if check_solution(U_try, V_try, W_try):
                    return U_try, V_try, W_try
                
                # Simple perturbation
                # Pick a random variable and adjust it
                choice = random.randint(0, 2)
                if choice == 0:
                    idx = random.randint(0, n*m-1)
                    r = random.randint(0, target_rank-1)
                    U_try[idx, r] += random.gauss(0, 0.01)
                elif choice == 1:
                    idx = random.randint(0, m*p-1)
                    r = random.randint(0, target_rank-1)
                    V_try[idx, r] += random.gauss(0, 0.01)
                else:
                    idx = random.randint(0, p*n-1)
                    r = random.randint(0, target_rank-1)
                    W_try[idx, r] += random.gauss(0, 0.01)
        
        return None
    
    # Try ranks from lower bound up
    lower_bound = max(n*p, m*p, n*m)
    
    for target_rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        result = random_search_with_constraints(target_rank)
        if result is not None:
            U_try, V_try, W_try = result
            # Round to half-integers and verify
            U_h = np.round(U_try * 2) / 2
            V_h = np.round(V_try * 2) / 2
            W_h = np.round(W_try * 2) / 2
            if check_solution(U_h, V_h, W_h):
                best_rank = target_rank
                U, V, W = U_h, V_h, W_h
                break
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
