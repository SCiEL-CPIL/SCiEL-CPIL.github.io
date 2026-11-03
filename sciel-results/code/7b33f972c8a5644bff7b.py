"""Rewritten program for 3x4x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s/150s limit
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # Precompute the list of all (i, j, k) indices and their corresponding
    # positions in U, V, W tensors
    all_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_indices.append((i, j, k))
    
    # Build rank-58 solution using two Strassen blocks + schoolbook for rest
    def build_rank_58():
        def get_strassen_block(i0, j0, k0):
            """Returns 7 terms for a 2x2x2 Strassen block with global offsets."""
            terms = []
            
            # Term 1: M1 = (a00+a11)(b00+b11)
            u1 = [((i0+0)*m + (j0+0), 1.0), ((i0+1)*m + (j0+1), 1.0)]
            v1 = [((j0+0)*p + (k0+0), 1.0), ((j0+1)*p + (k0+1), 1.0)]
            w1 = [((k0+0)*n + (i0+0), 1.0), ((k0+1)*n + (i0+1), 1.0)]
            terms.append((u1, v1, w1))
            
            # Term 2: M2 = (a10+a11)b00
            u2 = [((i0+1)*m + (j0+0), 1.0), ((i0+1)*m + (j0+1), 1.0)]
            v2 = [((j0+0)*p + (k0+0), 1.0)]
            w2 = [((k0+0)*n + (i0+1), 1.0), ((k0+1)*n + (i0+1), -1.0)]
            terms.append((u2, v2, w2))
            
            # Term 3: M3 = a00(b01-b11)
            u3 = [((i0+0)*m + (j0+0), 1.0)]
            v3 = [((j0+0)*p + (k0+1), 1.0), ((j0+1)*p + (k0+1), -1.0)]
            w3 = [((k0+1)*n + (i0+0), 1.0), ((k0+1)*n + (i0+1), 1.0)]
            terms.append((u3, v3, w3))
            
            # Term 4: M4 = a11(b10-b00)
            u4 = [((i0+1)*m + (j0+1), 1.0)]
            v4 = [((j0+1)*p + (k0+0), 1.0), ((j0+0)*p + (k0+0), -1.0)]
            w4 = [((k0+0)*n + (i0+0), 1.0), ((k0+0)*n + (i0+1), 1.0)]
            terms.append((u4, v4, w4))
            
            # Term 5: M5 = (a00+a01)b11
            u5 = [((i0+0)*m + (j0+0), 1.0), ((i0+0)*m + (j0+1), 1.0)]
            v5 = [((j0+1)*p + (k0+1), 1.0)]
            w5 = [((k0+0)*n + (i0+0), -1.0), ((k0+1)*n + (i0+0), 1.0)]
            terms.append((u5, v5, w5))
            
            # Term 6: M6 = (a10-a00)(b00+b01)
            u6 = [((i0+1)*m + (j0+0), 1.0), ((i0+0)*m + (j0+0), -1.0)]
            v6 = [((j0+0)*p + (k0+0), 1.0), ((j0+0)*p + (k0+1), 1.0)]
            w6 = [((k0+1)*n + (i0+1), 1.0)]
            terms.append((u6, v6, w6))
            
            # Term 7: M7 = (a01-a11)(b10+b11)
            u7 = [((i0+0)*m + (j0+1), 1.0), ((i0+1)*m + (j0+1), -1.0)]
            v7 = [((j0+1)*p + (k0+0), 1.0), ((j0+1)*p + (k0+1), 1.0)]
            w7 = [((k0+0)*n + (i0+0), 1.0)]
            terms.append((u7, w7, v7))  # Note: fixed order
            
            # Fix the last term
            terms[6] = (u7, v7, w7)
            
            return terms
        
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Block 1: i0=0, j0=0, k0=0
        terms1 = get_strassen_block(0, 0, 0)
        for u, v, w in terms1:
            U_col = np.zeros(dim_u)
            for idx, val in u:
                U_col[idx] = val
            V_col = np.zeros(dim_v)
            for idx, val in v:
                V_col[idx] = val
            W_col = np.zeros(dim_w)
            for idx, val in w:
                W_col[idx] = val
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
            
        # Block 2: i0=0, j0=2, k0=0
        terms2 = get_strassen_block(0, 2, 0)
        for u, v, w in terms2:
            U_col = np.zeros(dim_u)
            for idx, val in u:
                U_col[idx] = val
            V_col = np.zeros(dim_v)
            for idx, val in v:
                V_col[idx] = val
            W_col = np.zeros(dim_w)
            for idx, val in w:
                W_col[idx] = val
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
            
        # Schoolbook for the rest
        covered = set()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    covered.add((i, j, k))
                    covered.add((i, j+2, k))
                    
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    U_col = np.zeros(dim_u)
                    U_col[i*m + j] = 1.0
                    V_col = np.zeros(dim_v)
                    V_col[j*p + k] = 1.0
                    W_col = np.zeros(dim_w)
                    W_col[k*n + i] = 1.0
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
                    
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-9):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m + j
                    idx_v = j*p + k
                    idx_w = k*n + i
                    s = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Try to find a lower-rank decomposition using ALS
    def try_lower_rank(target_rank, max_tries=3, time_per_try=None):
        if time_per_try is None:
            time_per_try = 25.0
        
        for attempt in range(max_tries):
            try_start = time.time()
            try_deadline = try_start + time_per_try
            
            # Initialize with random factors
            np.random.seed(42 + attempt * 1000)
            U = np.random.randn(dim_u, target_rank)
            V = np.random.randn(dim_v, target_rank)
            W = np.random.randn(dim_w, target_rank)
            
            # ALS iterations
            converged = False
            for it in range(100):
                if time.time() > try_deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # A = V[j,:] * W[:,i,:] (broadcast)
                        A = V[j*p:(j+1)*p, :] * W[0:p, i*n:(i+1)*n, :]
                        # A shape: (p, target_rank)
                        b = np.ones(p)
                        # Solve A^T A u = A^T b
                        ATA = A.T @ A
                        ATb = A.T @ b
                        try:
                            u_new = np.linalg.solve(ATA, ATb)
                        except np.linalg.LinAlgError:
                            u_new = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
                        U[i*m+j, :] = u_new
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = U[0:n*m, :] * W[k*n:(k+1)*n, :]
                        # A shape: (n*m, target_rank) but we need to be careful
                        # Actually, for V[j*p+k], we need:
                        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,j,k}
                        # For fixed j,k: sum_c V[j*p+k,c] * (sum_i U[i*m+j,c] * W[k*n+i,c]) 
                        # Wait, the tensor is T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # For fixed (j,k), we want to minimize sum_{i} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                        # This is not a simple linear system in V[j*p+k,:].
                        # Let me reconsider.
                        
                        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        # For fixed i, j, k, this is a product of three vectors' components at index c, summed over c.
                        # This is NOT bilinear in the usual ALS sense.
                        # The tensor decomposition here is of the form:
                        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        # This is a 3-way tensor decomposition where each factor is indexed by one mode.
                        # Standard ALS for 3-way tensors works when the tensor is 
                        # T = sum_c u_c ⊗ v_c ⊗ w_c, which is exactly this.
                        # So for updating V[j*p+k, :], we fix U and W, and the residual for 
                        # entry (i,j,k) is r[i,j,k] = 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                        # But this couples all (i,j,k) for fixed j,k through the same V[j*p+k,:].
                        # Let's define for fixed j,k:
                        # Let B[i,c] = U[i*m+j, c] * W[k*n+i, c] for i=0..n-1, c=0..r-1
                        # Then T[i,j,k] = sum_c B[i,c] * V[j*p+k, c]
                        # So for fixed j,k, we have a linear system:
                        # B @ V[j*p+k, :]^T = ones(n)
                        # where B is (n, target_rank)
                        
                        B = np.zeros((n, target_rank))
                        for i in range(n):
                            for c in range(target_rank):
                                B[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        b = np.ones(n)
                        ATA = B.T @ B
                        ATb = B.T @ b
                        try:
                            v_new = np.linalg.solve(ATA, ATb)
                        except np.linalg.LinAlgError:
                            v_new = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
                        V[j*p+k, :] = v_new
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # For fixed k, i: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # Let C[j,c] = U[i*m+j, c] * V[j*p+k, c] for j=0..m-1, c=0..r-1
                        # Then T[i,j,k] = sum_c C[j,c] * W[k*n+i, c]
                        C = np.zeros((m, target_rank))
                        for j in range(m):
                            for c in range(target_rank):
                                C[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        b = np.ones(m)
                        ATA = C.T @ C
                        ATb = C.T @ b
                        try:
                            w_new = np.linalg.solve(ATA, ATb)
                        except np.linalg.LinAlgError:
                            w_new = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
                        W[k*n+i, :] = w_new
                
                # Check convergence
                if it > 5:
                    err = verify(U, V, W)
                    if err < 1e-9:
                        converged = True
                        break
                
                # Early abort if not improving
                if it == 10:
                    err10 = verify(U, V, W)
                    if err10 > 1.0:  # If error is still large after 10 iterations, likely stuck
                        break
                
                if time.time() > try_deadline:
                    break
            
            # Final verification
            err = verify(U, V, W)
            if err < 1e-9:
                # Try to snap to integers/half-integers
                U_snap = np.round(2 * U) / 2
                V_snap = np.round(2 * V) / 2
                W_snap = np.round(2 * W) / 2
                err_snap = verify(U_snap, V_snap, W_snap)
                if err_snap < 1e-9:
                    return U_snap, V_snap, W_snap
                else:
                    return U, V, W
        
        return None
    
    # First, build the rank-58 solution
    U58, V58, W58 = build_rank_58()
    err58 = verify(U58, V58, W58)
    if err58 > 1e-9:
        # Fallback to pure schoolbook
        U_sb = np.zeros((dim_u, n*m*p))
        V_sb = np.zeros((dim_v, n*m*p))
        W_sb = np.zeros((dim_w, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1.0
                    V_sb[j*p + k, col] = 1.0
                    W_sb[k*n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find lower rank
    best_result = (U58, V58, W58)
    best_rank = 58
    
    for target_rank in [57, 56, 55]:
        if time.time() > deadline - 5:
            break
        result = try_lower_rank(target_rank)
        if result is not None:
            U, V, W = result
            err = verify(U, V, W)
            if err < 1e-9:
                best_result = (U, V, W)
                best_rank = target_rank
                break  # Found the lowest possible, stop
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
