import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    def flatten_tensor(U, V, W, n, m, p):
        """Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,k"""
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        temp = np.einsum('ijr,jkr->ikr', U_3d, V_3d)
        W_trans = W_3d.transpose(1, 0, 2)  # (n, p, R)
        result = np.einsum('ikr,ikr->ik', temp, W_trans)
        return result

    def verify_decomposition(U, V, W, n, m, p):
        """Verify exact decomposition"""
        T = flatten_tensor(U, V, W, n, m, p)
        return np.allclose(T, np.ones((n, p)), atol=1e-9)

    def tensor_residual(U, V, W, n, m, p, r):
        """Compute the tensor contribution of term r (flattened to 1D vector of length n*p*m)"""
        U_col = U[:, r].reshape(n, m)
        V_col = V[:, r].reshape(m, p)
        W_col = W[:, r].reshape(p, n)
        # Term r contributes: sum_j U[i,j] * V[j,k] * W[k,i] for each (i,k)
        # But we need the full tensor in 3D space for linear dependence check
        # The tensor has indices (i, j, k) with value U[i,j]*V[j,k]*W[k,i]
        # Actually the matrix multiplication tensor T[i,k] = sum_j A[i,j]*B[j,k]
        # The decomposition is: T[i,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] summed over j,r
        # Wait, let me re-read the convention.
        # W[k*n+i, r] means W is indexed by (k, i) where k is row of B (col of A's output) and i is row of A
        # So the tensor is T[i, k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For term r alone: T_r[i,k] = sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a matrix of size n x p for each r.
        # But for linear dependence, we need to check if the "slice" for term r can be expressed
        # as a linear combination of other terms.
        
        # Actually, the proper way: each term r defines a rank-1 tensor in the space of 
        # (i, j, k) indices? No. Let me think again.
        # The matrix multiplication tensor is T_{i,j,k} = delta_{jk} (i.e., T[i,j,k]=1 if j==k, else 0)
        # Wait no. For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor T[i,k] = sum_j A[i,j]*B[j,k] is not a 3-index tensor in the standard sense.
        # The bilinear form is: C[i,k] = sum_{j,r} U[i,j,r] * V[j,k,r] * W[k,i,r]
        # So for each r, we have a "slice" that is: sum_j U[i,j,r]*V[j,k,r]*W[k,i,r]
        # This is a matrix of size n x p for each r.
        # Two terms r1 and r2 are linearly dependent if the matrix for r1 is a scalar multiple
        # of the matrix for r2. But that's not quite right for the full problem.
        
        # The correct approach: The set of all rank-r decompositions means we want to find
        # the minimum R such that T = sum_{r=1}^R T_r where each T_r is "rank-1" in some sense.
        # Here, T_r[i,k] = sum_j U_r[i,j] * V_r[j,k] * W_r[k,i]
        # This is a product of three matrices contracted appropriately.
        # For linear dependence: term r is redundant if T_r = sum_{s != r} c_s * T_s
        # i.e., the n x p matrix M_r = sum_j U_r[i,j]*V_r[j,k]*W_r[k,i] 
        # can be written as a linear combination of the other M_s matrices.
        
        # So we can flatten each M_r to a vector of length n*p and check linear dependence.
        U_col = U[:, r].reshape(n, m)
        V_col = V[:, r].reshape(m, p)
        W_col = W[:, r].reshape(p, n)
        M_r = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                M_r[i, k] = np.dot(U_col[i, :], V_col[:, k]) * W_col[k, i]
        return M_r.flatten()

    # Schoolbook decomposition (rank n*m*p = 84)
    rank0 = n * m * p
    U = np.zeros((n * m, rank0), dtype=float)
    V = np.zeros((m * p, rank0), dtype=float)
    W = np.zeros((p * n, rank0), dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1

    # Greedy rank reduction: try to remove each term if it's linearly dependent
    # on the remaining terms
    current_rank = rank0
    keep = [True] * rank0
    
    # We'll iterate multiple passes
    max_passes = 5
    for pass_num in range(max_passes):
        if time.time() > deadline - 10:
            break
        removed_any = False
        for r in range(rank0):
            if not keep[r]:
                continue
            if time.time() > deadline - 10:
                break
            # Try to express term r as a linear combination of other kept terms
            # Build the matrix of other terms' contributions
            other_indices = [s for s in range(rank0) if keep[s] and s != r]
            if not other_indices:
                break
            
            # Get the flattened tensor contribution of term r
            M_r = tensor_residual(U, V, W, n, m, p, r)
            
            # Build the system: M_r = sum_{s in other} c_s * M_s
            A_mat = np.zeros((n * p, len(other_indices)))
            for idx, s in enumerate(other_indices):
                A_mat[:, idx] = tensor_residual(U, V, W, n, m, p, s)
            
            # Solve least squares
            try:
                coeffs, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, M_r, rcond=1e-10)
                # Check if the fit is exact
                M_recon = A_mat @ coeffs
                if np.allclose(M_recon, M_r, atol=1e-9):
                    # Term r is redundant, remove it
                    keep[r] = False
                    removed_any = True
            except np.linalg.LinAlgError:
                pass
        
        if not removed_any:
            break
    
    # Build the reduced decomposition
    kept_indices = [r for r in range(rank0) if keep[r]]
    final_rank = len(kept_indices)
    
    if final_rank < rank0:
        U_final = U[:, kept_indices]
        V_final = V[:, kept_indices]
        W_final = W[:, kept_indices]
    else:
        U_final = U
        V_final = V
        W_final = W

    # Verify
    if verify_decomposition(U_final, V_final, W_final, n, m, p) and time.time() < deadline:
        return {
            "U": U_final.tolist(),
            "V": V_final.tolist(),
            "W": W_final.tolist()
        }

    # Fallback: return the original schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

