import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Target tensor: T[i,j,k] = 1 if (i,j) in A, (j,k) in B, (k,i) in C
    # We want U (nm x r), V (mp x r), W (pn x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k)
    # where delta is 1 if the tensor entry is 1, 0 otherwise.
    
    # Build the target tensor in flattened form
    # T_flat[idx] = 1 where idx corresponds to (i,j,k) with T[i,j,k] = 1
    target_indices = []
    target_values = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The multiplication tensor has T[i,j,k] = 1 always for matrix mult
                # Actually, for matrix multiplication C = A*B, (C)_{i,k} = sum_j A_{i,j} B_{j,k}
                # The tensor T is defined as T[i,j,k] = 1 for all i,j,k
                # Wait, no. The tensor for matrix mult is T[i,j,k] = delta_{i,j} delta_{j,k} in some conventions...
                # Let me re-read: "exactly decompose the matrix multiplication tensor"
                # The standard matrix multiplication tensor is T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
                # No wait, that's not right either.
                # 
                # The matrix multiplication tensor T_M(n,m,p) is defined as:
                # T[i,j,k] = 1 if we're computing (A*B)[i,k] using A[i,j] and B[j,k]
                # Actually, the tensor is simply all ones? No.
                # 
                # The tensor T is such that sum_{j} A[i,j] * B[j,k] = (A*B)[i,k]
                # This can be written as: (A*B)[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # For this to equal sum_j A[i,j] B[j,k], we need:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j} (i.e., 1 for each j that contributes)
                # Actually it should be: the coefficient of A[i,j]*B[j,k] in the expansion is 1.
                # So T[i,j,k] = 1 for all i, j, k.
                # 
                # Wait, that can't be right because then the tensor is just the all-ones tensor.
                # Let me think again. The tensor T satisfies:
                # <T, A ⊗ B ⊗ C> = tr(C^T A B) for all A, B, C
                # This means T[i,j,k] = 1 for all i, j, k.
                # 
                # Hmm, but that would make the schoolbook rank n*m*p trivially optimal?
                # No, because the schoolbook uses rank n*m*p but there might be lower rank decompositions.
                
                target_indices.append((i, j, k))
                target_values.append(1.0)
    
    T = np.zeros((n, m, p))
    T[:] = 1.0
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    def to_list(mat):
        return [[float(v) for v in row] for row in mat]
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        r = U.shape[1]
        # Compute the reconstruction
        # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_round_to_half_integers(U, V, W, tol=1e-6):
        """Try to round entries to half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        err = verify(U_r, V_r, W_r)
        if err < 1e-9:
            return U_r, V_r, W_r
        return None
    
    # Try ALS search for lower ranks
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s buffer
    
    def als_search(rank, max_iter=2000, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        prev_err = float('inf')
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U: for each row (i*m+j), we need U[i*m+j,:] to satisfy
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is: U[i*m+j,:] @ (V[j*p+,:] ⊙ W[:, k*n+i])^T ... 
            # Actually, for fixed (i,j), we want:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # Let A_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product)
            # Then sum_c U[i*m+j,c] * A_k[c] = 1 for all k
            # This is: U[i*m+j,:] @ A^T = 1 (vector of ones of length p)
            # where A is p x rank matrix with A[k,:] = V[j*p+k,:] * W[k*n+i,:]
            
            # Build the system for U
            for i in range(n):
                for j in range(m):
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i*m+j, :] @ A.T = ones(p)
                    # i.e., A @ U[i*m+j, :].T = ones(p)
                    # Use least squares
                    try:
                        sol = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        U[i*m+j, :] = sol
                    except:
                        pass
            
            # Similarly update V
            for j in range(m):
                for k in range(p):
                    # For fixed (j,k), we want:
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                    # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        sol = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        V[j*p+k, :] = sol
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # For fixed (k,i), we want:
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        sol = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        W[k*n+i, :] = sol
                    except:
                        pass
            
            # Compute error
            err = verify(U, V, W)
            if abs(prev_err - err) < tol:
                break
            if err < 1e-9:
                break
            prev_err = err
        
        if err < 1e-9:
            return U, V, W
        return None
    
    # Try ranks from 99 down
    best_result = None
    for rank in range(rank_school - 1, max(rank_school - 20, 50), -1):
        if time.time() > deadline:
            break
        result = als_search(rank)
        if result is not None:
            U, V, W = result
            # Try to round to half-integers
            rounded = try_round_to_half_integers(U, V, W)
            if rounded is not None:
                U_r, V_r, W_r = rounded
                # Final verification
                err = verify(U_r, V_r, W_r)
                if err < 1e-9:
                    return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
            # If rounding fails, try the float version (might not be accepted)
            err = verify(U, V, W)
            if err < 1e-9:
                # Return as floats - may or may not be accepted
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fallback: schoolbook
    return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
# EVOLVE-BLOCK-END
