import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 36 # Fallback
    
    # Generate the target tensor T[i, j, k] = 1 if valid indices else 0?
    # No, the tensor is the identity map for matrix multiplication.
    # T_{i, j, k} is not a tensor in the usual sense, it's a 3-tensor M_{i,j,k}
    # such that C_{ik} = sum_j A_{ij} B_{jk} = sum_j M_{i,j,k} A_{ij} B_{jk}
    # Wait, the decomposition is:
    # C_{ik} = sum_alpha U_{i,j, alpha} V_{j,k, alpha} W_{k,i, alpha} ??
    # No, the standard tensor decomposition for matmul is:
    # C_{ik} = sum_{j, alpha} A_{ij} B_{jk} T_{i,j,k, alpha}?
    
    # Let's stick to the interface:
    # U shape (n*m, rank) -> U[i*m + j, alpha]
    # V shape (m*p, rank) -> V[j*p + k, alpha]
    # W shape (p*n, rank) -> W[k*n + i, alpha]
    
    # The condition is:
    # For all i, k:
    # sum_j (sum_alpha U[i*m+j, alpha] * W[k*n+i, alpha]) * B[j*p+k, alpha] ??
    # No, the bilinear form is:
    # C_{ik} = sum_{j, alpha} A_{ij} V_{j*p+k, alpha} W_{k*n+i, alpha} U_{i*m+j, alpha} ?
    
    # Let's derive the tensor identity from the code structure in the prompt.
    # The prompt says: "exactly decompose the matrix multiplication tensor"
    # Standard definition:
    # M_{i,j,k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # Actually, the tensor $\mathcal{T}$ has entries $\mathcal{T}_{i,j,k} = 1$.
    # The decomposition is:
    # $\mathcal{T}_{i,j,k} = \sum_{\alpha=1}^r U_{i,j,\alpha} V_{j,k,\alpha} W_{k,i,\alpha}$
    # Wait, indices don't match.
    # U is indexed by (i, j). V by (j, k). W by (k, i).
    # So the term is $U_{i,j,\alpha} V_{j,k,\alpha} W_{k,i,\alpha}$.
    # We need $\sum_{\alpha} U_{i,j,\alpha} V_{j,k,\alpha} W_{k,i,\alpha} = 1$ for all i, j, k?
    # No, that would imply the tensor is the all-ones tensor.
    
    # Let's look at the trivial solution.
    # If we set for each (i,j,k) a unique alpha:
    # U[i*m+j, alpha_ijk] = 1
    # V[j*p+k, alpha_ijk] = 1
    # W[k*n+i, alpha_ijk] = 1
    # Then sum_alpha ... = 1 for the specific (i,j,k) combination.
    # But the matrix multiplication is:
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This is linear in A and B.
    # The "tensor" being decomposed is the structure of the multiplication.
    # The condition for the factors U, V, W to represent matmul is:
    # For any matrices A, B:
    # C_{ik} = sum_{j, alpha} A_{ij} V_{j*p+k, alpha} W_{k*n+i, alpha} U_{i*m+j, alpha}
    # Wait, U depends on i,j.
    # Let's rearrange:
    # C_{ik} = sum_j A_{ij} ( sum_alpha U_{i*m+j, alpha} W_{k*n+i, alpha} V_{j*p+k, alpha} )
    # We want this to equal sum_j A_{ij} B_{jk}.
    # This requires:
    # sum_alpha U_{i*m+j, alpha} W_{k*n+i, alpha} V_{j*p+k, alpha} = delta_{jk}? No.
    # It requires the bilinear form to match.
    
    # Actually, the standard tensor rank problem for matmul $(n, m, p)$ is:
    # Find $r$ and $U_{i,j,\alpha}, V_{j,k,\alpha}, W_{k,i,\alpha}$ such that:
    # $\sum_{\alpha} U_{i,j,\alpha} V_{j,k,\alpha} W_{k,i,\alpha} = 1$ for all $i,j,k$?
    # No, that's not right either.
    
    # Let's look at the error message: "tensor identity failed at target coordinate (1, 7, 2)".
    # This suggests the evaluator checks a specific identity.
    # Usually, the identity is:
    # $\sum_{\alpha} U_{i, j, \alpha} V_{j, k, \alpha} W_{k, i, \alpha} = 1$
    # If this holds, then for any A, B:
    # $C_{ik} = \sum_{j, \alpha} A_{ij} V_{j,k,\alpha} W_{k,i,\alpha} U_{i,j,\alpha}$?
    # No, the standard form is:
    # $C_{ik} = \sum_{\alpha} (\sum_j A_{ij} U_{i,j,\alpha}) (\sum_j B_{jk} V_{j,k,\alpha}) W_{k,i,\alpha}$?
    # No, that's not bilinear in A and B separately in a way that sums to matmul.
    
    # Correct Tensor Decomposition Definition:
    # The matrix multiplication tensor $T$ is defined by $T_{i,j,k} = 1$.
    # A rank-$r$ decomposition is a set of $r$ rank-1 tensors $(U^{(\alpha)}, V^{(\alpha)}, W^{(\alpha)})$
    # such that $T = \sum_{\alpha=1}^r U^{(\alpha)} \otimes V^{(\alpha)} \otimes W^{(\alpha)}$.
    # In components:
    # $T_{i,j,k} = \sum_{\alpha=1}^r U_{i,j,\alpha} V_{j,k,\alpha} W_{k,i,\alpha}$?
    # Indices:
    # U is on (i, j). V is on (j, k). W is on (k, i).
    # The tensor $T$ has indices $(i, j, k)$.
    # So the equation is:
    # $\sum_{\alpha} U_{i,j,\alpha} V_{j,k,\alpha} W_{k,i,\alpha} = 1$ for all $i, j, k$.
    
    # Let's verify with the trivial solution.
    # For each (i,j,k), let alpha be unique.
    # U_{i,j,alpha} = 1, V_{j,k,alpha} = 1, W_{k,i,alpha} = 1.
    # Sum is 1.
    # Does this allow computing C = AB?
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # Using the decomposition:
    # $C_{ik} = \sum_{j, \alpha} A_{ij} B_{jk} \dots$?
    # The reconstruction formula is:
    # $C_{ik} = \sum_{\alpha} W_{k,i,\alpha} (\sum_j A_{ij} U_{i,j,\alpha}) (\sum_j B_{jk} V_{j,k,\alpha})$?
    # No, that's not right.
    
    # The correct reconstruction from rank-1 tensors in matmul context:
    # If $T_{i,j,k} = \sum U_{i,j,\alpha} V_{j,k,\alpha} W_{k,i,\alpha}$,
    # Then $C_{ik} = \sum_{j} A_{ij} B_{jk} = \sum_{j, \alpha} A_{ij} B_{jk} \dots$?
    # Actually, the identity $\sum_\alpha U_{i,j,\alpha} V_{j,k,\alpha} W_{k,i,\alpha} = 1$ implies:
    # $\sum_j A_{ij} B_{jk} = \sum_j A_{ij} B_{jk} (\sum_\alpha U_{i,j,\alpha} V_{j,k,\alpha} W_{k,i,\alpha})$
    # $= \sum_\alpha W_{k,i,\alpha} (\sum_j A_{ij} U_{i,j,\alpha}) (\sum_j B_{jk} V_{j,k,\alpha})$
    # This is the standard Strassen-like form.
    # So the condition IS:
    # $\sum_{\alpha=1}^r U_{i*m+j, \alpha} V_{j*p+k, \alpha} W_{k*n+i, \alpha} = 1$
    # for all $i \in [0,n), j \in [0,m), k \in [0,p)$.
    
    # The previous code failed because it didn't satisfy this identity.
    
    # Strategy:
    # 1. Iterate rank r from 24 to 36.
    # 2. For each r, try to find U, V, W (half-integers) satisfying the identity.
    # 3. Use random initialization + local search (gradient descent on real values, then snap).
    # 4. If found, return.
    # 5. If not, return trivial rank 36.
    
    def check_decomposition(U, V, W, r):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check sum_alpha U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for alpha in range(r):
                        s += U[i*m+j][alpha] * V[j*p+k][alpha] * W[k*n+i][alpha]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Trivial solution
    trivial_rank = n * m * p
    U_triv = [[0]*trivial_rank for _ in range(n*m)]
    V_triv = [[0]*trivial_rank for _ in range(m*p)]
    W_triv = [[0]*trivial_rank for _ in range(p*n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i*m+j][idx] = 1
                V_triv[j*p+k][idx] = 1
                W_triv[k*n+i][idx] = 1
                idx += 1
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}
    best_rank = trivial_rank
    
    start_time = time.time()
    time_limit = 110.0 # 10s margin
    
    # Search ranks
    for r in range(24, trivial_rank + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try a few random restarts for this rank
        found = False
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Local search: minimize sum_{i,j,k} (sum_alpha U_ij_alpha V_jk_alpha W_ki_alpha - 1)^2
            # This is a non-convex problem. Use simple gradient descent or Nelder-Mead.
            # Given the small size, we can use a simple iterative update.
            
            # We will use a simplified gradient step.
            # L = sum_{i,j,k} (S_{ijk} - 1)^2
            # dL/dU_{i,j,alpha} = 2 * sum_{i,j,k} (S_{ijk}-1) * V_{j,k,alpha} * W_{k,i,alpha}
            
            # To speed up, we can use numpy.
            # Precompute V and W tensors?
            # V is (m*p, r) -> V[j, k, alpha]
            # W is (p*n, r) -> W[k, i, alpha]
            
            V_tensor = V.reshape(m, p, r)
            W_tensor = W.reshape(p, n, r)
            
            # S[i, j, k] = sum_alpha U[i, j, alpha] V[j, k, alpha] W[k, i, alpha]
            # This is a tensor contraction.
            
            # We can update U, V, W cyclically.
            # Fixed V, W, solve for U?
            # U[i, j, alpha] appears in S[i, j, k] for all k.
            # S[i, j, k] = sum_alpha U[i, j, alpha] (V[j, k, alpha] W[k, i, alpha])
            # Let C[i, j, k, alpha] = V[j, k, alpha] W[k, i, alpha]
            # Then S[i, j, k] = sum_alpha U[i, j, alpha] C[i, j, k, alpha]
            # This is a linear system for each (i, j)? No, U is free.
            # We want S[i, j, k] = 1.
            # sum_alpha U[i, j, alpha] C[i, j, k, alpha] = 1 for all k.
            # This is a system of p equations for r unknowns U[i, j, 0..r-1].
            # We can solve this least-squares problem for each (i, j).
            
            # Similarly for V and W.
            
            # Let's implement this cyclic coordinate descent.
            
            U_flat = U.reshape(n, m, r)
            
            for step in range(50):
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Solve for U[i, j, :]
                        # Matrix A is (p, r) where A[k, alpha] = V_tensor[j, k, alpha] * W_tensor[k, i, alpha]
                        A_mat = np.einsum('jka, kba -> kb', V_tensor[j], W_tensor[:, i]) 
                        # Wait, V_tensor is (m, p, r). V_tensor[j] is (p, r).
                        # W_tensor is (p, n, r). W_tensor[:, i] is (p, r).
                        # Elementwise product: (p, r).
                        # We want sum_alpha U[i,j,alpha] * A_mat[k, alpha] = 1 for all k.
                        # So A_mat @ U[i,j,:] = 1
                        try:
                            U_flat[i, j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                        except:
                            pass

                # Update V
                V_flat = V.reshape(m, p, r)
                for j in range(m):
                    for k in range(p):
                        # Solve for V[j, k, :]
                        # S[i, j, k] = sum_alpha U[i, j, alpha] V[j, k, alpha] W[k, i, alpha] = 1
                        # For fixed j, k, variable is V[j, k, alpha].
                        # sum_alpha V[j, k, alpha] (U[i, j, alpha] W[k, i, alpha]) = 1 for all i.
                        # Matrix B is (n, r) where B[i, alpha] = U_flat[i, j, alpha] * W_tensor[k, i, alpha]
                        B_mat = np.einsum('ija, kia -> ia', U_flat[:, j], W_tensor[k])
                        try:
                            V_flat[j, k, :] = np.linalg.lstsq(B_mat, np.ones(n), rcond=None)[0]
                        except:
                            pass

                # Update W
                W_flat = W.reshape(p, n, r)
                for k in range(p):
                    for i in range(n):
                        # Solve for W[k, i, :]
                        # sum_alpha W[k, i, alpha] (U[i, j, alpha] V[j, k, alpha]) = 1 for all j.
                        # Matrix C is (m, r) where C[j, alpha] = U_flat[i, j, alpha] * V_flat[j, k, alpha]
                        C_mat = np.einsum('ija, jka -> ja', U_flat[i], V_flat[:, k])
                        try:
                            W_flat[k, i, :] = np.linalg.lstsq(C_mat, np.ones(m), rcond=None)[0]
                        except:
                            pass

            # After local search, snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Check if snapped version works
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            if check_decomposition(U_list, V_list, W_list, r):
                return {"U": U_list, "V": V_list, "W": W_list}
        
    return best_result
# EVOLVE-BLOCK-END
