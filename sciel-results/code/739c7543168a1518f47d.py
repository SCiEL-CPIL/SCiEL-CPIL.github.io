import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # n=2, m=4, p=4

    # --- Build rank-28 Strassen baseline ---
    def build_strassen_28():
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        def apply_strassen(X_rows, X_cols, Y_rows, Y_cols, Z_rows, Z_cols, offset):
            # M1: (X00+X11)(Y00+Y11)
            U[X_rows[0]*m + X_cols[0]][offset+0] += 1
            U[X_rows[1]*m + X_cols[1]][offset+0] += 1
            V[Y_rows[0]*p + Y_cols[0]][offset+0] += 1
            V[Y_rows[1]*p + Y_cols[1]][offset+0] += 1
            # M2: (X10+X11)Y00
            U[X_rows[1]*m + X_cols[0]][offset+1] += 1
            U[X_rows[1]*m + X_cols[1]][offset+1] += 1
            V[Y_rows[0]*p + Y_cols[0]][offset+1] += 1
            # M3: X00(Y01-Y11)
            U[X_rows[0]*m + X_cols[0]][offset+2] += 1
            V[Y_rows[0]*p + Y_cols[1]][offset+2] += 1
            V[Y_rows[1]*p + Y_cols[1]][offset+2] -= 1
            # M4: X11(Y10-Y00)
            U[X_rows[1]*m + X_cols[1]][offset+3] += 1
            V[Y_rows[1]*p + Y_cols[0]][offset+3] += 1
            V[Y_rows[0]*p + Y_cols[0]][offset+3] -= 1
            # M5: (X00+X01)Y11
            U[X_rows[0]*m + X_cols[0]][offset+4] += 1
            U[X_rows[0]*m + X_cols[1]][offset+4] += 1
            V[Y_rows[1]*p + Y_cols[1]][offset+4] += 1
            # M6: (X10-X00)(Y00+Y01)
            U[X_rows[1]*m + X_cols[0]][offset+5] += 1
            U[X_rows[0]*m + X_cols[0]][offset+5] -= 1
            V[Y_rows[0]*p + Y_cols[0]][offset+5] += 1
            V[Y_rows[0]*p + Y_cols[1]][offset+5] += 1
            # M7: (X01-X11)(Y10+Y11)
            U[X_rows[0]*m + X_cols[1]][offset+6] += 1
            U[X_rows[1]*m + X_cols[1]][offset+6] -= 1
            V[Y_rows[1]*p + Y_cols[0]][offset+6] += 1
            V[Y_rows[1]*p + Y_cols[1]][offset+6] += 1
            # W updates
            w_idx = Z_cols[0]*n + Z_rows[0]
            W[w_idx][offset+0] += 1
            W[w_idx][offset+3] += 1
            W[w_idx][offset+4] -= 1
            W[w_idx][offset+6] += 1
            w_idx = Z_cols[1]*n + Z_rows[0]
            W[w_idx][offset+2] += 1
            W[w_idx][offset+4] += 1
            w_idx = Z_cols[0]*n + Z_rows[1]
            W[w_idx][offset+1] += 1
            W[w_idx][offset+3] += 1
            w_idx = Z_cols[1]*n + Z_rows[1]
            W[w_idx][offset+0] += 1
            W[w_idx][offset+1] -= 1
            W[w_idx][offset+2] += 1
            W[w_idx][offset+5] += 1

        apply_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
        apply_strassen([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 7)
        apply_strassen([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], 14)
        apply_strassen([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], 21)
        return U, V, W

    U_base, V_base, W_base = build_strassen_28()
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = 28

    # --- Verify a decomposition is valid ---
    def verify(U, V, W, rank):
        # Check: for each output element C[i,j] = sum_k A[i,k]*B[k,j]
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # W[k*n+i, r] corresponds to output C[i, k] (row i, col k)
        # U[i*m+k, r] corresponds to A[i, k]
        # V[k*p+j, r] corresponds to B[k, j]
        # Tensor equation: sum_r U[i*m+k, r] * V[k*p+j, r] = sum_r W[j*n+i, r] * delta
        # Actually the decomposition means: C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] summed over k
        # More precisely: the tensor T_{i,j,k} = delta (for matrix mult) must equal
        # sum_r U[i*m+k, r] * V[k*p+j, r] ... no.
        #
        # The bilinear decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where T[i,j,k] = 1 if the (i,k) element of A times (k,j) element of B contributes to C[i,j]
        # Actually for matrix multiplication: C[i,j] = sum_k A[i,k] B[k,j]
        # The tensor is T_{i,j,k} = 1 (it's the identity in the k-index for each i,j pair)
        # Wait, the standard tensor for matrix mult (n,m,p) has T_{i,j,k} = 1 for all valid i,j,k
        # No: T_{i,j,k} = 1 if this triple contributes, which is always 1 for matrix mult.
        # The decomposition: sum_r U[i*m+k, r] * V[k*p+j, r] = W[j*n+i, r] ... no.
        #
        # Correct formulation: The multiplication tensor has entries T_{i,j,k} = 1 for all i,j,k.
        # A rank-r decomposition: T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
        # where (u_r)_{i,k} = U[i*m+k, r], (v_r)_{k,j} = V[k*p+j, r], (w_r)_{i,j} = W[j*n+i, r]
        # and T_{i,j,k} = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] ... no that's not right either.
        #
        # Actually: sum_r U[i*m+k, r] * V[k*p+j, r] should equal W[j*n+i, r] summed... no.
        # The correct identity: For matrix mult, C[i,j] = sum_k A[i,k] B[k,j]
        # In tensor form: sum_k T_{i,j,k} * A[i,k] * B[k,j] = C[i,j] where T_{i,j,k}=1
        # Rank-r decomposition: T_{i,j,k} = sum_r U[i*m+k, r] * V[k*p+j, r] * ... 
        # No. The standard form: T = sum_r u_r ⊗ v_r ⊗ w_r means
        # T_{i,j,k} = sum_r (u_r)_{i,k} (v_r)_{k,j} (w_r)_{i,j}
        # So: 1 = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] for all i,j,k? No that's 3-way.
        #
        # Actually for bilinear complexity, we have:
        # C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j])
        # = sum_r (u_r^T a_i) * (v_r^T b_j)
        # This equals sum_r W[j*n+i, r] * (u_r^T a_i) * (v_r^T b_j) only if W entries are the coefficients.
        #
        # The correct check: For each r, define linear forms on A and B:
        #   a_r = sum_{i,k} U[i*m+k, r] * A[i,k]  (a scalar)
        #   b_r = sum_{k,j} V[k*p+j, r] * B[k,j]  (a scalar)
        # Then C[i,j] = sum_r W[j*n+i, r] * a_r * b_r
        # This must equal sum_k A[i,k] * B[k,j] for all A, B.
        #
        # So the condition is: for all i, j, k:
        #   sum_r W[j*n+i, r] * U[i*m+k, r] * V[k*p+j, r] = 1  (delta for matrix mult)
        # Wait no, it should be: the coefficient of A[i,k]*B[k,j] in the expression equals 1.
        # Coefficient of A[i',k']*B[k'',j'] in sum_r W[j*n+i,r] * (sum_{a,b} U[a*m+b,r]*A[a,b]) * (sum_{c,d} V[c*p+d,r]*B[c,d])
        # = sum_r W[j*n+i, r] * U[i'*m+k', r] * V[k''*p+j', r]
        # This must equal delta_{i,i'} * delta_{k',k''} * delta_{j,j'} for the (i,j) output... 
        # No, it must equal 1 if (i',k') and (k'',j') correspond to the same k, i.e., k'=k'' and the output is (i,j).
        #
        # Let me just verify numerically with random matrices.
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        # Compute C from decomposition
        # a_r = U_flat @ A_flat where U_flat is (rank, n*m) and A_flat is (n*m,)
        U_flat = U.T  # (rank, n*m)
        V_flat = V.T  # (rank, m*p)
        W_flat = W.T  # (rank, p*n)
        A_flat = A.flatten()  # (n*m,)
        B_flat = B.flatten()  # (m*p,)
        a_vals = U_flat @ A_flat  # (rank,)
        b_vals = V_flat @ B_flat  # (rank,)
        C_flat = W_flat @ (a_vals * b_vals)  # (p*n,)
        C_computed = C_flat.reshape(p, n).T  # (n, p)
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
        return True

    # Verify baseline
    assert verify(U_base, V_base, W_base, 28), "Baseline Strassen should be valid"

    # --- Search for lower rank ---
    # The tensor equation: for each (i, j, k) triple, 
    # sum_r W[j*n+i, r] * U[i*m+k, r] * V[k*p+j, r] = 1
    # This is a system of n*p*m = 2*4*4 = 32 equations.
    # Unknowns: rank * (n*m + m*p + p*n) = rank * (8 + 16 + 8) = 32*rank
    # For rank < 28, we have more unknowns than equations, so solutions should exist.
    # But the system is nonlinear (trilinear in U, V, W).
    #
    # Approach: Fix W to be sparse (each output has few non-zero W entries),
    # then solve for U and V. Or use a different strategy.
    #
    # Alternative: Use the fact that for matrix multiplication, there are known
    # lower bounds. The rank of (2,4,4) matmul is at least... 
    # By the standard lower bound, rank >= (n+m+p-2)^2 / 4 = (2+4+4-2)^2/4 = 16/4 = 4.
    # That's a very weak bound. A better bound: rank >= n*m*p / max(n,m,p) = 32/4 = 8.
    # Actually the trivial lower bound is n*m = 8 (each output needs at least one term per input pair).
    #
    # Let's try a structured search: assume a block structure.
    # Since n=2, we can think of this as two rows of A multiplying B.
    # Row 0 of A: [a00, a01, a02, a03]
    # Row 1 of A: [a10, a11, a12, a13]
    # C[0,j] = sum_k A[0,k] B[k,j]
    # C[1,j] = sum_k A[1,k] B[k,j]
    #
    # For each output C[i,j], we need sum_k A[i,k] B[k,j] = sum_r W[j*n+i,r] * (sum_k U[i*m+k,r]*A[i,k]) * (sum_k V[k*p+j,r]*B[k,j])
    #
    # Let's try rank values from 16 down to see if we can find valid solutions.
    # Use a simple approach: try to construct solutions with a specific structure.
    
    # Try to find a rank-r solution by fixing a sparse W pattern and solving
    # For each candidate rank, try multiple random initializations
    
    def try_find_decomposition(rank, max_attempts=50):
        """Try to find a rank-r decomposition using iterative approach."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Iterative optimization: minimize sum of squared errors in tensor equations
            # Tensor equation residual: for each (i,j,k): 
            #   err(i,j,k) = sum_r W[j*n+i,r]*U[i*m+k,r]*V[k*p+j,r] - 1
            # This is non-convex. Use alternating minimization.
            
            best_loss = np.inf
            for iteration in range(200):
                if time.time() > deadline - 5:
                    return None
                
                # Update U: for fixed V, W, the problem in U is linear
                # For each (i,j,k): sum_r W[j*n+i,r]*U[i*m+k,r]*V[k*p+j,r] = 1
                # Group by (i,k): for each i,k, we have equations over j:
                #   sum_j [sum_r W[j*n+i,r]*V[k*p+j,r]] * U[i*m+k,r] ... no, it's per r.
                # Actually: for fixed i,k, the equations over j are:
                #   sum_r U[i*m+k,r] * W[j*n+i,r] * V[k*p+j,r] = 1 for j=0..p-1
                # This is a linear system in U[i*m+k,:] with p equations and rank unknowns.
                
                # Build the linear system for U
                # Unknowns: U[0,0], U[0,1], ..., U[n*m-1, rank-1]  -> n*m*rank unknowns
                # Equations: for each (i,j,k): sum_r U[i*m+k,r] * (W[j*n+i,r] * V[k*p+j,r]) = 1
                # That's n*p*m = 32 equations, n*m*rank = 8*rank unknowns
                
                # For each (i,k), the p equations in j involve only U[i*m+k, :]:
                # sum_r U[i*m+k,r] * W[j*n+i,r] * V[k*p+j,r] = 1, j=0..p-1
                # This is p equations, rank unknowns for each (i,k) pair.
                # Total: n*m*(p) = 2*4*4 = 32 equations, n*m*rank = 8*rank unknowns.
                
                if rank > p:  # rank > 4, underdetermined for each (i,k) block
                    # Solve least squares for each (i,k)
                    for i in range(n):
                        for k in range(m):
                            # Build A matrix: p x rank
                            A_mat = np.zeros((p, rank))
                            for j in range(p):
                                for r in range(rank):
                                    A_mat[j, r] = W[j*n+i, r] * V[k*p+j, r]
                            b_vec = np.ones(p)
                            # Solve least squares
                            try:
                                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                                U[i*m+k, :] = sol
                            except:
                                U[i*m+k, :] = np.random.randn(rank)
                else:
                    # rank <= p, possibly overdetermined or square
                    for i in range(n):
                        for k in range(m):
                            A_mat = np.zeros((p, rank))
                            for j in range(p):
                                for r in range(rank):
                                    A_mat[j, r] = W[j*n+i, r] * V[k*p+j, r]
                            b_vec = np.ones(p)
                            try:
                                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                                U[i*m+k, :] = sol
                            except:
                                U[i*m+k, :] = np.random.randn(rank)
                
                # Update V similarly
                # For fixed U, W: for each (k,j), equations over i:
                # sum_r V[k*p+j,r] * U[i*m+k,r] * W[j*n+i,r] = 1, i=0..n-1
                if rank > n:  # rank > 2
                    for k in range(m):
                        for j in range(p):
                            A_mat = np.zeros((n, rank))
                            for i in range(n):
                                for r in range(rank):
                                    A_mat[i, r] = U[i*m+k, r] * W[j*n+i, r]
                            b_vec = np.ones(n)
                            try:
                                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                                V[k*p+j, :] = sol
                            except:
                                V[k*p+j, :] = np.random.randn(rank)
                else:
                    for k in range(m):
                        for j in range(p):
                            A_mat = np.zeros((n, rank))
                            for i in range(n):
                                for r in range(rank):
                                    A_mat[i, r] = U[i*m+k, r] * W[j*n+i, r]
                            b_vec = np.ones(n)
                            try:
                                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                                V[k*p+j, :] = sol
                            except:
                                V[k*p+j, :] = np.random.randn(rank)
                
                # Update W similarly
                # For fixed U, V: for each (i,j), equations over k:
                # sum_r W[j*n+i,r] * U[i*m+k,r] * V[k*p+j,r] = 1, k=0..m-1
                if rank > m:  # rank > 4
                    for i in range(n):
                        for j in range(p):
                            A_mat = np.zeros((m, rank))
                            for k in range(m):
                                for r in range(rank):
                                    A_mat[k, r] = U[i*m+k, r] * V[k*p+j, r]
                            b_vec = np.ones(m)
                            try:
                                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                                W[j*n+i, :] = sol
                            except:
                                W[j*n+i, :] = np.random.randn(rank)
                else:
                    for i in range(n):
                        for j in range(p):
                            A_mat = np.zeros((m, rank))
                            for k in range(m):
                                for r in range(rank):
                                    A_mat[k, r] = U[i*m+k, r] * V[k*p+j, r]
                            b_vec = np.ones(m)
                            try:
                                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                                W[j*n+i, :] = sol
                            except:
                                W[j*n+i, :] = np.random.randn(rank)
                
                # Check convergence
                loss = 0.0
                for i in range(n):
                    for j in range(p):
                        for k in range(m):
                            val = 0.0
                            for r in range(rank):
                                val += W[j*n+i, r] * U[i*m+k, r] * V[k*p+j, r]
                            loss += (val - 1.0) ** 2
                if loss < best_loss:
                    best_loss = loss
                if loss < 1e-12:
                    # Found a valid solution
                    if verify(U, V, W, rank):
                        return U, V, W
                if iteration > 50 and loss > best_loss * 1.1:
                    break
            
            # Try with different initialization: sparse W
            if attempt % 5 == 4:
                U = np.random.randn(n * m, rank) * 0.1
                V = np.random.randn(m * p, rank) * 0.1
                W = np.zeros((p * n, rank))
                # Give each output a few non-zero W entries
                for i in range(n):
                    for j in range(p):
                        idx = j*n + i
                        nonzero = np.random.choice(rank, size=min(3, rank), replace=False)
                        W[idx, nonzero] = np.random.choice([-1, 1], size=len(nonzero))
                
                for iteration in range(300):
                    if time.time() > deadline - 5:
                        return None
                    
                    # Update U
                    for i in range(n):
                        for k in range(m):
                            A_mat = np.zeros((p, rank))
                            for j in range(p):
                                for r in range(rank):
                                    A_mat[j, r] = W[j*n+i, r] * V[k*p+j, r]
                            b_vec = np.ones(p)
                            try:
                                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                                U[i*m+k, :] = sol
                            except:
                                pass
                    
                    # Update V
                    for k in range(m):
                        for j in range(p):
                            A_mat = np.zeros((n, rank))
                            for i in range(n):
                                for r in range(rank):
                                    A_mat[i, r] = U[i*m+k, r] * W[j*n+i, r]
                            b_vec = np.ones(n)
                            try:
                                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                                V[k*p+j, :] = sol
                            except:
                                pass
                    
                    # Check
                    loss = 0.0
                    for i in range(n):
                        for j in range(p):
                            for k in range(m):
                                val = 0.0
                                for r in range(rank):
                                    val += W[j*n+i, r] * U[i*m+k, r] * V[k*p+j, r]
                                loss += (val - 1.0) ** 2
                    if loss < 1e-12:
                        if verify(U, V, W, rank):
                            return U, V, W
    
    # Try ranks from 27 down
    for candidate_rank in range(27, 7, -1):
        if time.time() > deadline - 10:
            break
        result = try_find_decomposition(candidate_rank, max_attempts=20)
        if result is not None:
            U_new, V_new, W_new = result
            if verify(U_new, V_new, W_new, candidate_rank):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = candidate_rank
                break  # Found the lowest rank we searched, stop
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
