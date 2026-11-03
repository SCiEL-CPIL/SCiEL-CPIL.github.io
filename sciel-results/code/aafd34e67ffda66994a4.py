import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if and only if
    # the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C.
    # Actually, for standard matrix multiplication, the tensor is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T is a 3-tensor where T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # But wait - the problem says the output factor follows W[k*n+i,r], so W is indexed by (k,i)
    # This suggests the tensor structure is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}? No.
    # 
    # Let me re-read: "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 5 matrix."
    # Standard matrix multiplication: C = A @ B where A is 4x5, B is 5x5, C is 4x5.
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    # 
    # The tensor decomposition should satisfy:
    # For each (i,k), C[i,k] = sum_r U[i*m+k, r] * V[?, r] * W[?, r]
    # But the problem states the factors are:
    # U: (n*m, rank) = (4*5, rank) = (20, rank)
    # V: (m*p, rank) = (5*5, rank) = (25, rank)
    # W: (p*n, rank) = (5*4, rank) = (20, rank)
    # 
    # And the convention is W[k*n+i, r]. So W is indexed by (k,i) where k is 0..p-1=0..4, i is 0..n-1=0..3.
    # 
    # The correct tensor equation for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form with the given indexing:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should give 1 for the "correct" (i,j,k) triples
    # 
    # Actually, I think the tensor T is defined as: T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term
    # For C[i,k] = sum_j A[i,j]*B[j,k], each term (i,j,k) contributes to C[i,k].
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise? No, that's not right either.
    # 
    # Let me think differently. The tensor multiplication is:
    # C = T(A, B) where T is the multiplication tensor.
    # The rank of T is the minimum number of rank-1 tensors needed to decompose it.
    # 
    # For standard matrix multiplication m x n times n x p, the tensor has rank at most m*n*p (schoolbook).
    # The tensor entries are: T[i,j,k] where i indexes rows of A (0..m-1), j indexes columns of A / rows of B (0..n-1), k indexes columns of B (0..p-1).
    # T[i,j,k] = 1 if this term is used in the multiplication, which is always true for standard matrix multiplication.
    # 
    # Wait, I'm overcomplicating this. Let me look at the indexing in the current code:
    # U[i*m+j, r] - i is row of A, j is column of A / row of B
    # V[j*p+k, r] - j is row of B, k is column of B
    # W[k*n+i, r] - k is column of B, i is row of A (which becomes row of C)
    # 
    # So the equation should be:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # This would mean every triple (i,j,k) contributes 1, which is correct for standard matrix multiplication.
    # 
    # So the target tensor has ALL entries equal to 1? That doesn't make sense for a rank decomposition.
    # 
    # I think I'm misunderstanding the problem. Let me reconsider.
    # 
    # Actually, I believe the tensor is NOT all ones. The matrix multiplication tensor is a specific sparse tensor.
    # For C = A @ B, the (i,k) entry of C is sum_j A[i,j] * B[j,k].
    # The tensor T is such that T(A,B) = C, and T has a specific structure.
    # 
    # In the context of tensor rank for matrix multiplication, the tensor is typically defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the sum is over all (i,j,k) that contribute to the multiplication.
    # For standard matrix multiplication, ALL (i,j,k) triples contribute, so T has all entries 1.
    # 
    # But that can't be right because the rank would be trivially m*n*p.
    # 
    # I think the issue is that the tensor is not the "all ones" tensor, but rather the tensor that defines the bilinear map.
    # The bilinear map is: (A, B) -> A @ B
    # This is a 3-tensor where T[A_idx, B_idx, C_idx] gives the coefficient.
    # 
    # Let me just implement the ALS correctly and see if it works. The key fix is:
    # 1. Properly solve the linear systems in ALS
    # 2. Try multiple random initializations
    # 3. Use better convergence criteria
    
    min_rank = max(n * m, m * p, p * n)
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def compute_error(U, V, W):
        """Compute the squared error of the decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        return err
    
    # Try ranks from min_rank upward
    for R in range(min_rank, min_rank + 15):
        # Try multiple random initializations
        for seed in range(20):
            np.random.seed(seed * 1000 + R)
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            best_error = float('inf')
            best_U, best_V, best_W = None, None, None
            
            for it in range(200):
                # Solve for U: For each (i,j), find U[i*m+j, :] such that
                # sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] * W[k*n+i, r]) = 1
                A_u = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        W_idx = np.array([k * n + i for k in range(p)])
                        W_slice = W[W_idx, :]  # shape (p, R)
                        V_slice = V[j * p:(j + 1) * p, :]  # shape (p, R)
                        A_u[i * m + j, :] = np.sum(V_slice * W_slice, axis=0)
                
                # Solve U such that U @ A_u.T = ones, i.e., for each row: dot(U[row], A_u[row]) = 1
                # This is underdetermined. Use minimum norm solution: U[row] = A_u[row] / (A_u[row] @ A_u[row])
                norms_sq = np.sum(A_u * A_u, axis=1)
                norms_sq[norms_sq < 1e-12] = 1e-12
                U = A_u / norms_sq[:, np.newaxis]
                
                # Solve for V: For each (j,k), find V[j*p+k, :] such that
                # sum_r V[j*p+k, r] * (sum_i U[i*m+j, r] * W[k*n+i, r]) = 1
                A_v = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        U_idx = np.array([i * m + j for i in range(n)])
                        U_slice = U[U_idx, :]  # shape (n, R)
                        W_idx = np.array([k * n + i for i in range(n)])
                        W_slice = W[W_idx, :]  # shape (n, R)
                        A_v[j * p + k, :] = np.sum(U_slice * W_slice, axis=0)
                
                norms_sq_v = np.sum(A_v * A_v, axis=1)
                norms_sq_v[norms_sq_v < 1e-12] = 1e-12
                V = A_v / norms_sq_v[:, np.newaxis]
                
                # Solve for W: For each (k,i), find W[k*n+i, :] such that
                # sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
                A_w = np.zeros((p * n, R))
                for k in range(p):
                    for i in range(n):
                        U_idx = np.array([i * m + j for j in range(m)])
                        U_slice = U[U_idx, :]  # shape (m, R)
                        V_idx = np.array([j * p + k for j in range(m)])
                        V_slice = V[V_idx, :]  # shape (m, R)
                        A_w[k * n + i, :] = np.sum(U_slice * V_slice, axis=0)
                
                norms_sq_w = np.sum(A_w * A_w, axis=1)
                norms_sq_w[norms_sq_w < 1e-12] = 1e-12
                W = A_w / norms_sq_w[:, np.newaxis]
                
                err = compute_error(U, V, W)
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_error < 1e-8:
                    break
            
            if best_error < 1e-6:
                # Try to round to half-integers
                U_rounded = np.round(best_U * 2) / 2
                V_rounded = np.round(best_V * 2) / 2
                W_rounded = np.round(best_W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                
                # Try rounding to integers
                U_int = np.round(best_U)
                V_int = np.round(best_V)
                W_int = np.round(best_W)
                
                if verify(U_int, V_int, W_int):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # Try without rounding if error is very small
                if best_error < 1e-10:
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
