"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Build the matrix multiplication tensor T[i,j,k] = delta(i,k)
    # T has shape (n, m, p) where T[i,j,k] = 1 if i==k else 0
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            T[i, j, i] = 1
    
    # Schoolbook: rank = n*m*p = 75
    # Try to find lower rank via numerical SVD on tensor slices
    
    # Approach: try rank r from 74 down, check if tensor can be decomposed
    # For small reductions, try algebraic tricks
    
    # Strategy: For each output index i (0,1,2), the slice T[i,:,:] is a 5x5 matrix
    # T[i,j,k] = delta(i,k), so T[i,:,:] has 1 at position (j=i) for all j... 
    # Wait: T[i,j,k] = 1 iff i==k. So T[i,:,:] is a 5x5 matrix with 1s in column i.
    # That's a rank-1 matrix! Each slice T[i,:,:] has rank 1.
    
    # The tensor T can be written as sum over i of e_i ⊗ T[i,:,:]
    # where T[i,:,:] = e_i^T ⊗ (all-ones column) ... no
    # T[i,j,k] = delta(i,k), so T[i,:,:] has entries T[i,j,k] = 1 if k=i
    # So T[i,:,:] = (ones vector in j) ⊗ (e_i in k) = u_j * v_k where u_j=1, v_k=delta(k,i)
    # This is rank 1 for each i!
    
    # So T = sum_{i=0}^{2} e_i ⊗ (1_m) ⊗ e_i
    # = sum_{i=0}^{2} (e_i ⊗ 1_m) ⊗ e_i
    # This is a sum of 3 rank-1 tensors!
    # But wait, that's not right for matrix multiplication.
    
    # Let me re-examine. The matrix multiplication tensor for (n×m) * (m×p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is T[i,j,k] which when contracted: C[i,k] = sum_j A[i,j] T[i,j,k] B[j,k]
    # For this to give C[i,k] = sum_j A[i,j] B[j,k], we need T[i,j,k] = 1 for all i,j,k.
    # Wait no - T[i,j,k] should be 1 always, because every triple (i,j,k) contributes.
    
    # Actually the standard definition: the matrix multiplication tensor <n,m,p> has
    # T[i,j,k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p.
    # Then C[i,k] = sum_j A[i,j] B[j,k] = sum_{j} sum_{i',k'} A[i',j] T[i',j,k'] B[j,k'] delta(i',i)delta(k',k)
    # Hmm, let me be more careful.
    
    # The bilinear form is: C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
    # As a tensor: T ∈ R^{n×m×p} with T[i,j,k] = 1 for all i,j,k
    # Then C[i,k] = sum_{j} A[i,j] T[i,j,k] B[j,k] ... but this sums over j only,
    # and T[i,j,k]=1, so C[i,k] = sum_j A[i,j]*1*B[j,k] = sum_j A[i,j]*B[j,k]. Correct!
    
    # So T[i,j,k] = 1 for ALL i,j,k. It's the all-ones tensor!
    
    # Rank of all-ones tensor of size n×m×p is 1! Because it's (1_n) ⊗ (1_m) ⊗ (1_p).
    # But that can't be right for matrix multiplication...
    
    # I'm confusing myself. Let me restart with the correct tensor definition.
    
    # The matrix multiplication tensor <n,m,p> is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k.
    # The bilinear map is: (A,B) -> C where C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # With T[i,j,k]=1, this gives C[i,k] = sum_j A[i,j] B[j,k]. Correct.
    
    # Now, the rank of this tensor: T = sum over what?
    # T[i,j,k] = 1 = 1*1*1, so T = (1_n) ⊗ (1_m) ⊗ (1_p), rank 1??
    # That would mean matrix multiplication has rank 1, which is absurd.
    
    # I think I have the tensor definition wrong. Let me look at it differently.
    
    # In the standard formulation, the tensor for matrix multiplication is:
    # T ∈ R^{(n*p) × (n*m) × (m*p)} or similar, depending on flattening.
    
    # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # W[k*n+i, r] - so W is indexed by (k,i) pairs, flattened as k*n+i.
    # U is indexed by (i,j) pairs: U[i*m+j, r]
    # V is indexed by (j,k) pairs: V[j*p+k, r]
    
    # The decomposition: for each r, the term is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # And the sum over r should give the tensor.
    
    # The tensor T has entries T[(i,j), (j,k), (k,i)] = 1 for valid indices.
    # More precisely, T is a 3-tensor with first index (i,j) from 0..n*m-1,
    # second index (j,k) from 0..m*p-1, third index (k,i) from 0..p*n-1.
    # T[(i,j),(j',k),(k',i')] = 1 if i=j' wait no...
    
    # Let me think about it as: the matrix multiplication is a bilinear map
    # f: R^{n*m} × R^{m*p} -> R^{p*n}
    # f(A_flat, B_flat)[k*n+i] = sum_j A_flat[i*m+j] * B_flat[j*p+k]
    
    # The tensor T has T[a, b, c] where a=i*m+j, b=j*p+k, c=k'*n+i'
    # T[a,b,c] = 1 if the indices are consistent: the j in a equals the j in b,
    # and the i in a equals the i' in c, and the k in b equals the k' in c.
    
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # All other entries are 0.
    
    # This is the correct tensor! It's NOT all ones.
    
    # Now, the schoolbook decomposition has rank n*m*p = 75, with one term per (i,j,k).
    
    # Can we do better? The tensor T[i*m+j, j*p+k, k*n+i] = 1.
    
    # For fixed i: T[i*m+j, j*p+k, k*n+i] = 1 for all j,k.
    # This is like a "permutation" structure.
    
    # Let me try to find a lower rank decomposition numerically.
    
    # Build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try to find minimum rank via iterative SVD / alternating least squares
    # Start with schoolbook and try to compress
    
    # For now, let's try a numerical approach: 
    # Reshape T to a matrix and use SVD to get a low-rank approximation,
    # then verify if it's exact.
    
    # T as matrix: shape (n*m, m*p*p*n) or (n*m*m*p, p*n) etc.
    # Let's try: M1 = T.reshape(n*m, m*p * p*n)  -- not standard
    # Better: use the standard unfolding.
    
    # T_unfold1: shape (n*m, m*p * p*n) -- no, that's not right either.
    # The three modes are sizes n*m, m*p, p*n.
    # Unfold along mode 1: shape (n*m, m*p * p*n)
    # Unfold along mode 2: shape (m*p, n*m * p*n)
    # Unfold along mode 3: shape (p*n, n*m * m*p)
    
    # The rank of the tensor is at least the max of the ranks of all unfoldings.
    
    # Let's compute the rank of each unfolding.
    M1 = T.reshape(n*m, m*p * p*n)
    M2 = T.transpose(1,0,2).reshape(m*p, n*m * p*n)
    M3 = T.transpose(2,0,1).reshape(p*n, n*m * m*p)
    
    r1 = np.linalg.matrix_rank(M1, tol=1e-10)
    r2 = np.linalg.matrix_rank(M2, tol=1e-10)
    r3 = np.linalg.matrix_rank(M3, tol=1e-10)
    
    lower_bound = max(r1, r2, r3)
    
    # The tensor rank is at least lower_bound.
    # For our tensor, let's see what these ranks are.
    
    # If lower_bound < 75, we might be able to find a decomposition of rank = lower_bound
    # or slightly above.
    
    # Let's try to find a decomposition of rank r = lower_bound using ALS.
    
    def try_decompose(T_tensor, target_rank, max_iters=1000, tol=1e-10):
        """Try to find a rank-target_rank decomposition of T_tensor using ALS."""
        n1, n2, n3 = T_tensor.shape
        np.random.seed(42)
        
        # Initialize with random matrices
        U = np.random.randn(n1, target_rank)
        V = np.random.randn(n2, target_rank)
        W = np.random.randn(n3, target_rank)
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U: for each i, U[i,:] should satisfy
            # sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k]
            # Fix V and W, solve for U.
            # T_unfold1[i, j*n3 + k] = sum_r U[i,r] (V[:,r] ⊗ W[:,r])
            # Let's build the matrix for this.
            
            # For fixed j,k: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # This is: T_unfold1[i, j*n3+k] = sum_r U[i,r] * (V[j,r] * W[k,r])
            # Define Z1[j*n3+k, r] = V[j,r] * W[k,r]
            # Then T_unfold1 = U @ Z1.T, so U = T_unfold1 @ pinv(Z1.T)
            
            Z1 = np.zeros((n2 * n3, target_rank))
            for j in range(n2):
                for k in range(n3):
                    for r in range(target_rank):
                        Z1[j * n3 + k, r] = V[j, r] * W[k, r]
            
            T1 = T_tensor.reshape(n1, n2 * n3)
            # Solve U @ Z1.T = T1
            if target_rank <= min(n1, n2 * n3):
                U_new, residuals, rank_u, sv = np.linalg.lstsq(Z1.T, T1.T, rcond=tol)
                U = U_new.T
            else:
                U = U  # keep old
            
            # Update V
            Z2 = np.zeros((n1 * n3, target_rank))
            for i in range(n1):
                for k in range(n3):
                    for r in range(target_rank):
                        Z2[i * n3 + k, r] = U[i, r] * W[k, r]
            T2 = T_tensor.transpose(1, 0, 2).reshape(n2, n1 * n3)
            if target_rank <= min(n2, n1 * n3):
                V_new, residuals, rank_v, sv = np.linalg.lstsq(Z2.T, T2.T, rcond=tol)
                V = V_new.T
            else:
                V = V
            
            # Update W
            Z3 = np.zeros((n1 * n2, target_rank))
            for i in range(n1):
                for j in range(n2):
                    for r in range(target_rank):
                        Z3[i * n2 + j, r] = U[i, r] * V[j, r]
            T3 = T_tensor.transpose(2, 0, 1).reshape(n3, n1 * n2)
            if target_rank <= min(n3, n1 * n2):
                W_new, residuals, rank_w, sv = np.linalg.lstsq(Z3.T, T3.T, rcond=tol)
                W = W_new.T
            else:
                W = W
            
            # Check convergence
            error = np.linalg.norm(T_tensor - np.einsum('ir,jr,kr->ijk', U, V, W))
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < tol:
                break
        
        return best_U, best_V, best_W, best_error
    
    # Try ranks from lower_bound up to 75
    found = False
    for r in range(lower_bound, 76):
        if time.time() > deadline - 5:
            break
            
        U, V, W, error = try_decompose(T, r, max_iters=200, tol=1e-8)
        
        if error < 1e-6:
            # Check if we can make it exact with rational/half-integer entries
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            error_rounded = np.linalg.norm(T_tensor - np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded))
            
            if error_rounded < 1e-8:
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U_rounded]
                V_list = [[float(x) for x in row] for row in V_rounded]
                W_list = [[float(x) for x in row] for row in W_rounded]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Also try without rounding if error is very small
        if error < 1e-10:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If nothing found, fall back to schoolbook
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

