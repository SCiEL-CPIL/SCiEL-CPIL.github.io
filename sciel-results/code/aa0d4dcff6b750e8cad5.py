"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit

    # Schoolbook factor as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def check_decomposition(U, V, W):
        """Verify that U * W * V^T equals the matrix multiplication tensor."""
        rank = U.shape[1]
        # T[i*n*m, j*m*p, k*p*n] should be delta_{i,k} for all i,j,k... wait no.
        # The tensor M_{i,j,k} = 1 if the (i,k)-th entry of C = A @ B gets contribution from A_{i,j} * B_{j,k}
        # C_{i,k} = sum_j A_{i,j} * B_{j,k}
        # U row index: i*m + j
        # V row index: j*p + k
        # W row index: k*n + i
        # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j,j} = 1 (for the j term)
        # Actually the tensor we want to decompose has T[i,j,k] = 1 for all valid (i,j,k) representing the sum.
        # No, the bilinear form is: C_{i,k} = sum_j A_{i,j} B_{j,k}
        # As a tensor in (A indices, B indices, C indices): T_{(i,j),(j,k),(i,k)} = 1
        # With flattening: A index = i*m+j, B index = j*p+k, C index = k*n+i
        # We need sum_c U[a,c] V[b,c] W[cc,c] = 1 if the indices are consistent (same j), 0 otherwise.
        # Actually it's simpler: just compute the product and check.
        
        # For each column c, we add rank-1 tensor: U[:,c] outer V[:,c] outer W[:,c]
        # The full tensor is sum of these.
        # Tensor shape: (n*m, m*p, p*n)
        # T[a, b, cc] = sum_c U[a,c]*V[b,c]*W[cc,c]
        # We need T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
        # And T should be 0 for "inconsistent" triples (where the j in A index != j in B index, 
        #  or where the (i,k) from C index doesn't match).
        
        # Actually, the correct check: for all a, b, cc, the value should be:
        # Let i = a // m, j_a = a % m
        # Let j_b = b // p, k = b % p  
        # Let k_c = cc // n, i_c = cc % n
        # T = 1 if j_a == j_b and i == i_c and k == k_c, else 0
        
        # Better approach: just check specific entries
        # T[i*m+j, j*p+k, k*n+i] must equal 1 for all i,j,k
        # T[a, b, cc] = 0 if indices are inconsistent
        
        # Efficient check using matrix products:
        # sum_c U[a,c] * V[b,c] = (U @ V.T)[a,b]
        # Then we need: sum_b (U @ V.T)[a,b] * W[cc,b] ... no that's not right.
        # The tensor is T[a,b,cc] = sum_c U[a,c] V[b,c] W[cc,c]
        # This is (U @ V.T) but then contracted with W... 
        # Actually: T[a,b,cc] = sum_c U[a,c] V[b,c] W[cc,c]
        # = element (a, cc) of (U @ V.T) contracted... no.
        
        # Let's think differently. For fixed a, cc:
        # sum_b, c U[a,c] V[b,c] W[cc,c] ... no.
        # T[a,b,cc] is a 3D tensor. 
        # We can check: for each c, M_c = U[:,c] @ V[:,c].T is an (n*m) x (m*p) matrix.
        # Then T[a,b,cc] = sum_c M_c[a,b] * W[cc,c]
        
        # A simpler check: 
        # The product C = A @ B where A is n x m, B is m x p, C is n x p.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # In the decomposition: C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_k' V[j'*p+k', c] B[j',k']) * ... 
        # Hmm, this is getting complicated. Let me just do a direct numerical check.
        
        # Actually the cleanest check:
        # T[a, b, cc] should equal 1 iff:
        # a = i*m + j, b = j*p + k, cc = k*n + i (with consistent i, j, k)
        
        # Let's build the target tensor sparsely and compare.
        # But that's O(n*m*p) entries which is only 96. Very feasible.
        
        target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i * m + j, j * p + k, k * n + i] = 1.0
        
        # Compute actual tensor
        # T[a,b,cc] = sum_c U[a,c] V[b,c] W[cc,c]
        # = np.einsum('ac,bc,cc->abc', U, V, W)
        actual = np.einsum('ac,bc,cc->abc', U, V, W, optimize=True)
        
        return np.allclose(actual, target, atol=1e-10, rtol=1e-10)

    U0, V0, W0 = schoolbook()
    best_U, best_V, best_W = U0, V0, W0
    best_rank = U0.shape[1]
    
    # Try to find lower rank via random init + iterative refinement
    # Start from various rank levels
    for trial in range(30):
        if time.time() > deadline - 5:
            break
            
        # Try different rank values
        target_rank = min(best_rank, n * m * p - trial * 2)
        if target_rank < n + m + p - 1:  # absolute lower bound is max(n,m,p) but in practice higher
            target_rank = n + m + p - 1
        
        if trial == 0:
            target_rank = best_rank  # start with full schoolbook
            target_rank = 90
        
        # Random initialization with small perturbations around schoolbook structure
        # Use the schoolbook as seed and zero out some columns, then random init others
        if target_rank < best_rank:
            U_rand = np.random.randn(n * m, target_rank) * 0.5
            V_rand = np.random.randn(m * p, target_rank) * 0.5
            W_rand = np.random.randn(p * n, target_rank) * 0.5
            
            # Add structure: keep some schoolbook terms
            n_school = min(target_rank, n * m * p)
            for c in range(n_school):
                i = c // (m * p)
                j = (c // p) % m
                k = c % p
                U_rand[i * m + j, c] += 1.0
                V_rand[j * p + k, c] += 1.0
                W_rand[k * n + i, c] += 1.0
            
            U_cand = U_rand.copy()
            V_cand = V_rand.copy()
            W_cand = W_rand.copy()
        else:
            U_cand = U0[:, :target_rank].copy()
            V_cand = V0[:, :target_rank].copy()
            W_cand = W0[:, :target_rank].copy()
        
        # Iterative refinement: minimize ||U ⊗ V ⊗ W - T||_F
        # Gradient-based approach
        # T is the target tensor
        # loss = sum_{a,b,cc} (sum_c U[a,c]V[b,c]W[cc,c] - T[a,b,cc])^2
        
        lr = 0.1
        for it in range(500):
            if time.time() > deadline - 5:
                    break
            
            # Compute gradient
            # dL/du = 2 * sum_{b,cc} (sum_c U[a,c]V[b,c]W[cc,c] - T[a,b,cc]) * V[b,c] * W[cc,c]
            # = 2 * ((residual @ (V @ W.T)) ...) 
            # residual[a,b,cc] = actual[a,b,cc] - target[a,b,cc]
            
            actual = np.einsum('ac,bc,cc->abc', U_cand, V_cand, W_cand, optimize=True)
            residual = actual - np.einsum('ac,bc,cc->abc', *([np.ones((1,1))]*0) or 0, optimize=True) if False else actual - 0
            # Actually just use the target
            # Build target efficiently
            target = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[i * m + j, j * p + k, k * n + i] = 1.0
            
            residual = actual - target
            
            # dU[a,c] = 2 * sum_{b,cc} residual[a,b,cc] * V[b,c] * W[cc,c]
            # = 2 * (residual * V[b,c] broadcasted) ... 
            # dU = 2 * np.einsum('abc,bc,cc->ac', residual, V_cand, W_cand)
            dU = np.einsum('abc,bc,cc->ac', residual, V_cand, W_cand)
            dV = np.einsum('abc,ac,cc->bc', residual, U_cand, W_cand)
            dW = np.einsum('abc,ac,bc->cc', residual, U_cand, V_cand)
            
            # Adaptive learning rate
            norm_dU = np.linalg.norm(dU)
            norm_dV = np.linalg.norm(dV)
            norm_dW = np.linalg.norm(dW)
            max_norm = max(norm_dU, norm_dV, norm_dW, 1e-10)
            lr_eff = lr / max_norm
            
            U_cand -= lr_eff * dU
            V_cand -= lr_eff * dV
            W_cand -= lr_eff * dW
        
        # Check if decomposition is valid
        if check_decomposition(U_cand, V_cand, W_cand):
            if target_rank < best_rank:
                best_rank = target_rank
                best_U, best_V, best_W = U_cand, V_cand, W_cand
                print(f"Found rank {best_rank} at trial {trial}")
    
    # Round to nearest half-integer and re-verify
    if best_rank < n * m * p:
        best_U_r = np.round(best_U * 2) / 2
        best_V_r = np.round(best_V * 2) / 2
        best_W_r = np.round(best_W * 2) / 2
        if check_decomposition(best_U_r, best_V_r, best_W_r):
            best_U, best_V, best_W = best_U_r, best_V_r, best_W_r
    
    # Return in required format: list of lists
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
