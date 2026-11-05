"""Matrix multiplication 3,3,7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy: Matrix pencil / structured search approach.
    1. Start with schoolbook (rank 63)
    2. Try to find lower-rank decompositions by:
       a. Using known lower-rank sub-decompositions
       b. Iterative refinement with pruning
    3. Return the best (lowest rank) valid decomposition found
    """
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = 3, 3, 7
    
    def elapsed():
        return time.time() - start_time
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W form a valid tensor decomposition."""
        rank = U.shape[1]
        # Check: sum_r U[i1*m+j1, r] * V[j2*p+k2, r] * W[k3*n+i3, r]
        # should equal delta_{i1,i3} * delta_{j1,j2} * delta_{k2,k3}
        for i1 in range(n):
            for i3 in range(n):
                for j1 in range(m):
                    for j2 in range(m):
                        for k2 in range(p):
                            for k3 in range(p):
                                s = 0.0
                                for r in range(rank):
                                    s += U[i1*m+j1, r] * V[j2*p+k2, r] * W[k3*n+i3, r]
                                expected = (1.0 if i1==i3 and j1==j2 and k2==k3 else 0.0)
                                if abs(s - expected) > tol:
                                    return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using matrix operations."""
        rank = U.shape[1]
        # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # and check it equals the multiplication tensor
        # 
        # Alternative: check that for random A, B, the product matches
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            # Term: (A * u_col) elementwise? No...
            # The decomposition is: C[i,k] = sum_r (sum_{i',j'} U[i',j',r] A[i',j']) * (sum_{j',k'} V[j',k',r] B[j',k']) * W[k,i,r]
            # Wait, that's not right either. Let me re-derive.
            #
            # Actually for matrix mult tensor: T[i,j,k,l] where C[i,k] = sum_{j,l} A[i,j] B[l,k] * delta_{j,l}
            # No, C[i,k] = sum_j A[i,j] B[j,k]
            #
            # The tensor is T[i,j,k] = delta_{j,k} for the "multiplication" part
            # But we have 3 matrices: A (n x m), B (m x p), C (n x p)
            # C[i,k] = sum_j A[i,j] B[j,k]
            #
            # Tensor form: sum_{i,j,k} C[i,k] A[i,j] B[j,k] = sum_{i,j,k} (sum_j' A[i,j'] B[j',k]) A[i,j] B[j,k]
            # = sum_{i,j,j',k} A[i,j'] B[j',k] A[i,j] B[j,k]
            #
            # For a rank-1 term with factors u (n*m), v (m*p), w (n*p):
            # The contribution is (sum_{i,j} u[i,j] A[i,j]) * (sum_{j,k} v[j,k] B[j,k]) * (sum_{i,k} w[i,k] C[i,k])
            #
            # So the full decomposition satisfies:
            # sum_r (u_r . A) * (v_r . B) * (w_r . C) = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
            # for all A, B, C.
            #
            # This means: sum_r u_r[i1,j1] v_r[j2,k2] w_r[i3,k3] = delta_{i1,i3} delta_{j1,j2} delta_{k2,k3}
            # (where the LHS is summed over r)
            #
            # So verification: for all i1,i3,j1,j2,k2,k3:
            # sum_r U[i1*m+j1, r] V[j2*p+k2, r] W[k3*n+i3, r] = delta_{i1,i3} delta_{j1,j2} delta_{k2,k3}
            #
            # Fast check: use random A, B and compute C
            # C[i,k] should equal sum_j A[i,j] B[j,k]
            # And also equal sum_r (sum_{i',j'} U[i',j',r] A[i',j']) * (sum_{j',k'} V[j',k',r] B[j',k']) * W[k,i,r]
            # Wait, that gives a scalar, not C[i,k].
            #
            # Let me re-read the problem statement:
            # "C[i,k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]"
            # No, that's not standard.
            #
            # Standard tensor decomposition for matmul:
            # C[i,k] = sum_r (sum_{i',j'} U[i',j',r] A[i',j']) * (sum_{j',k'} V[j',k',r] B[j',k']) * W[i,k,r]
            #
            # This expands to:
            # C[i,k] = sum_r W[i,k,r] * (sum_{i',j'} U[i',j',r] A[i',j']) * (sum_{j',k'} V[j',k',r] B[j',k'])
            #
            # For this to equal sum_j A[i,j] B[j,k], we need:
            # sum_r W[i,k,r] U[i',j',r] V[j'',k'',r] = delta_{i',i} delta_{j',j''} delta_{k'',k} * (some normalization)
            #
            # Actually the condition is:
            # sum_r U[i1,j1,r] V[j2,k2,r] W[i3,k3,r] = delta_{i1,i3} delta_{j1,j2} delta_{k2,k3}
            #
            # Let me verify with schoolbook: U[i*m+j, r] = 1 for r corresponding to (i,j,k)
            # V[j*p+k, r] = 1, W[k*n+i, r] = 1
            # Then sum_r gives 1 only when i1=i3, j1=j2, k2=k3. ✓
            #
            # So for random A, B:
            # C[i,k] = sum_r W[i,k,r] * (sum_{i',j'} U[i',j',r] A[i',j']) * (sum_{j',k'} V[j',k',r] B[j',k'])
            #
            # Let me test this.
            
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        uA = np.dot(U[:, r].reshape(n, m), A)
                        vB = np.dot(V[:, r].reshape(m, p), B)
                        val += W[k*n+i, r] * uA * vB
                    if abs(val - C_expected[i, k]) > tol:
                        return False
        return True
    
    # Schoolbook decomposition (rank 63)
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
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    if not verify_fast(U_sb, V_sb, W_sb):
        # Should not happen, but just in case
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find lower rank decompositions
    # Approach: Use the structure of the problem
    # C = A @ B where A is 3x3, B is 3x7
    # We can split B into B1 (3x3) and B2 (3x4)
    # C1 = A @ B1 (3x3 x 3x3) and C2 = A @ B2 (3x3 x 3x4)
    
    # For 3x3 x 3x3, known rank is 19 (Coppersmith-Winograd)
    # But implementing CW is complex. Let's try a different approach.
    
    # Approach: Iterative rank reduction
    # Start with schoolbook, try to merge/prune terms
    
    # Alternative: Use the fact that we can share A-factors
    # For C2 = A @ B2 where B2 is 3x4:
    # C2[i,k] = sum_j A[i,j] B2[j,k] for k = 3,4,5,6
    # This is 4 separate 3x3 x 3x1 multiplications
    # Each 3x3 x 3x1 has rank 9 (schoolbook)
    # But we can share the 9 A-side terms
    
    # Actually, let me try a block approach:
    # Terms for C1 (columns 0-2): rank r1
    # Terms for C2 (columns 3-6): rank r2
    # Total rank = r1 + r2
    
    # For C2, since it's 3x3 x 3x4, we can use:
    # C2[i,k] = sum_j A[i,j] B2[j,k]
    # = sum_j A[i,j] (sum over k of B2[j,k] * something)
    #
    # A simple decomposition for 3x3 x 3x4:
    # For each j (0,1,2) and k (0,1,2,3): one term
    # That's 3*4 = 12 terms for B2 side, but we need A side too
    # Term (j,k): U has A[i,j] pattern, V has B[j,k] pattern, W has C[i,k] pattern
    # U[i*m+j, r] = 1 for all i (so it picks out A[:,j])
    # V[j*p+k, r] = 1 (picks out B[j,k])  
    # W[k*n+i, r] = 1 for all i (so it puts result in C[:,k])
    #
    # Wait, that's not right. Let me think more carefully.
    #
    # For term r corresponding to (j,k) in C2:
    # We want: (sum_i A[i,j]) * B[j,k] * (contribution to C[:,k])
    # But that gives sum_i A[i,j] * B[j,k] for all i, which is wrong.
    #
    # Correct: For each i, j, k:
    # C[i,k] gets contribution A[i,j] * B[j,k]
    # So term (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # That's the schoolbook for that sub-problem.
    #
    # For 3x3 x 3x4: 3*3*4 = 36 terms (schoolbook)
    # For 3x3 x 3x3: 3*3*3 = 27 terms (schoolbook), but rank 19 known
    #
    # Total schoolbook: 27 + 36 = 63 ✓
    
    # Let me try to find a better decomposition for 3x3 x 3x4
    # Using the matrix pencil idea:
    # Fix U and V to generic values, solve for W
    
    def try_pencil_decomposition(target_rank, seed=0):
        """
        Try to find a decomposition of rank `target_rank` by:
        1. Randomly initialize U and V
        2. Solve linear system for W
        3. Verify
        """
        if elapsed() > deadline - 5:
            return None
        
        np.random.seed(seed)
        
        # The condition is:
        # sum_r U[i1*m+j1, r] * V[j2*p+k2, r] * W[k3*n+i3, r] = delta_{i1,i3} delta_{j1,j2} delta_{k2,k3}
        #
        # For each (i1,j1,j2,k2,k3,i3), we have a linear equation in W.
        # But W has p*n*target_rank = 21*target_rank unknowns.
        # Number of equations: n*m*m*p*p*n = 3*3*3*7*7*3 = 1323
        #
        # Actually, the equation is:
        # For fixed (i1,j1,j2,k2,i3): sum_r U[i1*m+j1, r] * V[j2*p+k2, r] * W[k2*n+i3, r] = delta_{i1,i3} delta_{j1,j2} delta_{k2,k2}
        # Wait, k2 appears on both sides. Let me re-index.
        #
        # sum_r U[i1*m+j1, r] * V[j2*p+k2, r] * W[k3*n+i3, r] = delta_{i1,i3} delta_{j1,j2} delta_{k2,k3}
        #
        # For each (i1,j1,j2,k2,i3,k3): this is a linear equation in W[:,r] values.
        # But it's coupled across r.
        #
        # Alternative: Think of it as a matrix equation.
        # Let A_r = U[:,r].reshape(n,m), B_r = V[:,r].reshape(m,p), C_r = W[:,r].reshape(p,n)
        # Condition: sum_r A_r[i1,j1] * B_r[j2,k2] * C_r[k3,i3] = delta_{i1,i3} delta_{j1,j2} delta_{k2,k3}
        #
        # This is equivalent to: the tensor T = sum_r A_r ⊗ B_r ⊗ C_r equals the multiplication tensor.
        #
        # For a given U, V (fixed), we need to find W such that:
        # For all (i1,j1,j2,k2,i3,k3): sum_r U[i1*m+j1, r] V[j2*p+k2, r] W[k3*n+i3, r] = delta_{i1,i3} delta_{j1,j2} delta_{k2,k3}
        #
        # This is a system of linear equations in the entries of W.
        # Let's vectorize: let w = W.flatten() (length 21*target_rank)
        # For each equation (i1,j1,j2,k2,i3,k3):
        # sum_r U[i1*m+j1, r] V[j2*p+k2, r] * w[(k3*n+i3)*target_rank + r] = delta_{i1,i3} delta_{j1,j2} delta_{k2,k3}
        #
        # This is a linear system: M w = b
        # where M has 1323 rows and 21*target_rank columns.
        #
        # For target_rank = 63: 1323 rows, 1323 columns (square system)
        # For target_rank < 63: underdetermined (more unknowns than equations) - no wait
        # 21*target_rank unknowns, 1323 equations.
        # For target_rank = 63: 1323 unknowns, 1323 equations (square)
        # For target_rank = 50: 1050 unknowns, 1323 equations (overdetermined)
        #
        # So for target_rank < 63, we have more equations than unknowns, which is overdetermined.
        # We can use least squares and check if the residual is small.
        
        num_eq = n * m * m * p * p * n  # 1323
        num_unk = p * n * target_rank
        
        # Build the linear system
        # This might be too slow for large systems. Let's be smart.
        
        # Actually, let's use a different approach:
        # Fix U to be random, then the condition becomes:
        # sum_r U[i1*m+j1, r] * V[j2*p+k2, r] * W[k3*n+i3, r] = delta_{i1,i3} delta_{j1,j2} delta_{k2,k3}
        #
        # For fixed U, this is linear in V and W jointly.
        # But it's bilinear in V and W.
        #
        # Let me try: fix U and V, solve for W.
        # For each (i1,j1,j2,k2,i3,k3):
        # sum_r (U[i1*m+j1, r] * V[j2*p+k2, r]) * W[k3*n+i3, r] = delta_{i1,i3} delta_{j1,j2} delta_{k2,k3}
        #
        # Let coeff[i1,j1,j2,k2,i3,k3,r] = U[i1*m+j1, r] * V[j2*p+k2, r]
        # Then: sum_r coeff[*,r] * W[k3*n+i3, r] = rhs
        #
        # This is still coupled across r.
        #
        # Alternative: Use the fact that we can separate the indices.
        # For fixed (i1,j1,i3,k3):
        # sum_{j2,k2,r} U[i1*m+j1, r] V[j2*p+k2, r] W[k3*n+i3, r] * delta_{j1,j2} delta_{k2,k3} = delta_{i1,i3}
        #
        # This doesn't separate nicely.
        
        # Let me just try a numerical approach with random U, V and solve for W via least squares.
        
        U_rand = np.random.randn(n*m, target_rank)
        V_rand = np.random.randn(m*p, target_rank)
        
        # Build linear system for W
        # Unknowns: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..target_rank-1
        # Total: p*n*target_rank
        
        # Equations: for each (i1,j1,j2,k2,i3,k3)
        # sum_r U[i1*m+j1, r] * V[j2*p+k2, r] * W[k3*n+i3, r] = delta_{i1,i3} * delta_{j1,j2} * delta_{k2,k3}
        
        # Let's build the matrix M and vector b
        # M[row, col] where col indexes (k3, i3, r) and row indexes (i1, j1, j2, k2, i3, k3)
        
        # This is a 1323 x (21*target_rank) matrix
        # For target_rank up to ~63, that's 1323 x 1323, which is feasible.
        
        if num_unk > 2000 or num_eq > 2000:
            return None  # Too large
        
        M = np.zeros((num_eq, num_unk))
        b = np.zeros(num_eq)
        
        row = 0
        for i1 in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    for k2 in range(p):
                        for i3 in range(n):
                            for k3 in range(p):
                                rhs = (1.0 if i1==i3 and j1==j2 and k2==k3 else 0.0)
                                b[row] = rhs
                                for r in range(target_rank):
                                    col = (k3*n+i3)*target_rank + r
                                    M[row, col] = U_rand[i1*m+j1, r] * V_rand[j2*p+k2, r]
                                row += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_ls, sv = np.linalg.lstsq(M, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        # Check residual
        if len(residuals) > 0 and residuals[0] > 1e-6:
            return None
        
        W_rand = W_flat.reshape(p*n, target_rank)
        
        # Verify
        if verify_fast(U_rand, V_rand, W_rand, tol=1e-6):
            return U_rand, V_rand, W_rand
        
        return None
    
    # Try various ranks from low to high
    # Known lower bounds: rank(3,3,7) >= rank(3,3,3) = 19 (but this is a lower bound for the sub-problem)
    # Actually, rank(3,3,7) >= max(rank(3,3,3), ...) 
    # A simple lower bound: rank >= n*m = 9 (from A side) and rank >= m*p = 21 (from B side)
    # So rank >= 21
    
    # Let's try ranks from 21 upward
    for target_rank in range(21, best_rank + 1):
        if elapsed() > deadline - 10:
            break
        
        # Try a few random seeds
        for seed in range(5):
            if elapsed() > deadline - 10:
                break
            
            result = try_pencil_decomposition(target_rank, seed)
            if result is not None:
                U_t, V_t, W_t = result
                best_rank = target_rank
                best_U, best_V, best_W = U_t, V_t, W_t
                break  # Found a valid decomposition at this rank, move to next
        
        if elapsed() > deadline - 10:
            break
    
    # Also try a structured approach: use the block decomposition
    # C1 = A @ B1 (3x3 x 3x3) with some rank r1
    # C2 = A @ B2 (3x3 x 3x4) with some rank r2
    # Total rank = r1 + r2
    
    # For C2 (3x3 x 3x4), try to find low rank
    # Schoolbook for 3x3 x 3x4 is 36
    # Can we do better?
    
    # Let me try a specific construction for 3x3 x 3x4:
    # Use 9 terms for the "diagonal" part and 4 terms for the "off-diagonal"
    # Actually, let's just use the pencil method for the sub-problems.
    
    # For now, if we found something better than schoolbook, use it
    # Otherwise, return schoolbook
    
    if elapsed() > deadline - 2:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

