"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Schoolbook rank = n * m * p = 4 * 4 * 5 = 80
    schoolbook_rank = n * m * p
    
    # We need to find a decomposition with rank as low as possible.
    # The matrix multiplication tensor T[i, j, k] = 1 for all i in range(n), j in range(m), k in range(p)
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # Let's use the Strassen-like approach for the 4x4x4 part and handle the 5th column separately.
    # Actually, let's try a more direct approach: use the known rank-7 decomposition for 2x2x2
    # and build up to 4x4x5.
    
    # First, let's define the rank-7 decomposition for 2x2x2 matrix multiplication.
    # The standard Strassen decomposition has 7 multiplications.
    # For 2x2x2: A (2x2) * B (2x2) = C (2x2)
    # The indices are: i, j, k in {0, 1}
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Strassen's 7-term decomposition for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Let's encode this properly.
    # For 2x2x2, the factors have shapes (4, 7), (4, 7), (4, 7)
    
    U_2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],  # A00
        [0, 0, 0, 0, 0, 1, 1],  # A01
        [1, 1, 0, 0, 0, 0, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1], # A11
    ], dtype=float)
    
    V_2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 1, 1, 1],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1],  # B11
    ], dtype=float)
    
    W_2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],   # C00
        [0, 0, 1, 0, 0, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 0, 0, 1, 0, 0],   # C11
    ], dtype=float)
    
    # Verify: sum_r U_2[i*2+j, r] * V_2[j*2+k, r] * W_2[k*2+i, r] should be 1 for all i,j,k
    
    def verify_2x2(U, V, W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # Check if our Strassen decomposition is correct
    strassen_ok = verify_2x2(U_2, V_2, W_2)
    
    # If Strassen is correct, we can use it to build a 4x4x4 decomposition
    # by using the tensor product structure.
    # Actually, for 4x4x4, we can think of it as 2x2 blocks of 2x2 matrices.
    # But the indices are more complex. Let's use a different approach.
    
    # For 4x4x5, let's try to find a low-rank decomposition directly.
    # The schoolbook gives rank 80. We want to do better.
    
    # Let's use a random search with coordinate descent, but with a better initialization.
    # We'll start from the schoolbook and try to merge terms.
    
    # Actually, let's try a different strategy: use the fact that the all-ones tensor
    # has a known low-rank decomposition.
    
    # For the all-ones tensor of size n*m*p, the rank is at most n*m*p.
    # But we can do better by using the structure.
    
    # Let's try to use a rank-79 decomposition by modifying one term of the schoolbook.
    
    # First, create the schoolbook decomposition
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    # Now, let's try to find a rank-79 decomposition.
    # We'll remove one term and try to adjust the others.
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Try rank 79
    if time.time() < deadline - 20:
        U_79 = U_school[:, :79].copy()
        V_79 = V_school[:, :79].copy()
        W_79 = W_school[:, :79].copy()
        
        # The missing term is for (i=3, j=3, k=4) [last term]
        # We need to compensate for this.
        # Let's try to add a small perturbation to one of the existing terms.
        
        # Actually, let's use a more systematic approach: use scipy to solve for a low-rank decomposition.
        # We can set up a linear system where the unknowns are the entries of U, V, W
        # and the constraint is that the tensor equals the all-ones tensor.
        
        # This is a nonlinear system, so let's use a different approach.
        
        # Let's try to use the rank-7 decomposition for 2x2x2 to build up.
        # For 4x4x4, we can use 4 blocks of 2x2x2, but they interact.
        
        # Actually, let's just try a random search with gradient descent.
        
        pass
    
    # Let's try a different approach: use the known result that the rank of the 4x4x4
    # matrix multiplication tensor is 49 (Landsberg's bound).
    # But implementing this is complex. Let's stick with a simpler approach.
    
    # Let's try to find a rank-75 decomposition using random initialization and coordinate descent.
    
    def try_rank(target_rank, max_iter=50, n_attempts=3):
        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization
            U_t = np.random.randn(n * m, target_rank) * 0.5
            V_t = np.random.randn(m * p, target_rank) * 0.5
            W_t = np.random.randn(p * n, target_rank) * 0.5
            
            # Coordinate descent
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for r in range(target_rank):
                    # The contribution of term r to T[i,j,k] is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # We want sum_r to equal 1.
                    # The residual is R[i,j,k] = 1 - sum_{r' != r} U[i*m+j,r'] * V[j*p+k,r'] * W[k*n+i,r']
                    # We need U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = R[i,j,k]
                    # This is a rank-1 approximation problem.
                    
                    # Compute residual without term r
                    R = np.ones((n, m, p))
                    for r2 in range(target_rank):
                        if r2 == r:
                            continue
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    R[i,j,k] -= U_t[i*m+j, r2] * V_t[j*p+k, r2] * W_t[k*n+i, r2]
                    
                    # Find U[:,r] that minimizes ||U[:,r] * V[:,r] * W[:,r] - R||_F
                    # This is equivalent to finding the best rank-1 approximation of R.
                    # Use SVD: R = U_svd * s * V_svd^T, take the first component.
                    
                    # Reshape R to a matrix for SVD
                    R_mat = R.reshape(n*m, p)
                    try:
                        U_svd, s, Vt_svd = np.linalg.svd(R_mat, full_matrices=False)
                    except np.linalg.LinAlgError:
                        continue
                    
                    if s[0] > 0:
                        u_vec = U_svd[:, 0] * np.sqrt(s[0])
                        v_vec = Vt_svd[0, :] * np.sqrt(s[0])
                        
                        # Now we need to split u_vec * v_vec into U[i*m+j,r] * W[k*n+i,r]
                        # This is not straightforward. Let's use a simpler approach.
                        pass
            
            # Check if we got a good solution
            # (This is a placeholder; we need a proper check)
            
        return None
    
    # Given the complexity, let's just return the schoolbook for now.
    # But we can try to improve it by using the Strassen decomposition for the 2x2x2 blocks.
    
    # Actually, let's try a simpler improvement: use the fact that we can combine some terms.
    
    # For now, let's return the schoolbook decomposition.
    
    def to_list_with_ints(arr):
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i,j]
                if abs(val - round(val)) < 1e-8:
                    row.append(int(round(val)))
                elif abs(val - (round(val*2)/2)) < 1e-8:
                    row.append(round(val*2)/2)
                else:
                    row.append(float(val))
            result.append(row)
        return result
    
    U_list = to_list_with_ints(best_U)
    V_list = to_list_with_ints(best_V)
    W_list = to_list_with_ints(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
