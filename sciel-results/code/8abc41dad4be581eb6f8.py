<<<<<<< SEARCH
def _try_rank_search(target_rank, max_iterations=50, seed=0):
    """Try to find a rank-r decomposition using ALS with rounding."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    U_size = n * m  # 12
    V_size = m * p  # 32
    W_size = p * n  # 24
    
    # Initialize with random Gaussian
    U = rng.randn(U_size, target_rank) * 0.1
    V = rng.randn(V_size, target_rank) * 0.1
    W = rng.randn(W_size, target_rank) * 0.1
    
    # The tensor we want to match: T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # We'll use a weighted least squares approach
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for iteration in range(max_iterations):
        if time.time() > _START_TIME + _DEADLINE - 10:
            break
        
        # Simpler: use gradient descent with projection
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Target is 1
                    val = 0.0
                    for r in range(target_rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += (val - 1.0) ** 2
        
        # Also penalize non-zero entries where they shouldn't be
        # But for matrix multiplication tensor, only (i,j,k) with matching j matter
        # Actually all (i,j,k) are valid since j is shared
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        # Gradient step
        lr = 0.001
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for r in range(target_rank):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    err = val - 1.0
                    for r in range(target_rank):
                        grad_U[idx_u, r] += err * V[idx_v, r] * W[idx_w, r]
                        grad_V[idx_v, r] += err * U[idx_u, r] * W[idx_w, r]
                        grad_W[idx_w, r] += err * U[idx_u, r] * V[idx_v, r]
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Round to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Check if we've found an exact solution
        if error < 1e-6:
            # Verify and return
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify_decomposition(U_list, V_list, W_list):
                return U_list, V_list, W_list
    
    return None
=======
def _try_rank_search_fast(target_rank, max_iterations=100, seed=0):
    """
    Try to find a rank-r decomposition using fast vectorized ALS.
    Returns (U, V, W) if an exact half-integer solution is found, else None.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # Precompute indices for vectorized operations
    # We need to compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a contraction of three matrices.
    # Let's reshape U to (n, m, R), V to (m, p, R), W to (p, n, R).
    # We want C[i, k] = sum_j sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Note: W is indexed by [k*n+i], so W[k, i, r].
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(5):
        if time.time() > _START_TIME + _DEADLINE - 20:
            break
            
        # Initialize
        U = rng.randn(n, m, target_rank) * 0.5
        V = rng.randn(m, p, target_rank) * 0.5
        W = rng.randn(p, n, target_rank) * 0.5
        
        for iteration in range(max_iterations):
            if time.time() > _START_TIME + _DEADLINE - 10:
                break
                
            # Compute current C
            # C[i, k] = sum_j sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
            # We can do this by iterating j or using einsum.
            # einsum('ijr,jkr,ki->ik', U, V, W) might be slow if R is large, but let's try.
            # Alternatively, for each j: C += U[:, j, :] * V[j, :, :] * W.T? No, W is [k, i].
            # Let's use explicit loops over j (4 iterations) and vectorize over i, k, r.
            
            C = np.zeros((n, p))
            for j in range(m):
                # U_j shape (n, R)
                # V_j shape (p, R)
                # W shape (p, n, R) -> we need W[k, i, r]
                # Term for this j: sum_r U[i, j, r] * V[k, j, r] * W[k, i, r]
                # This is sum_r (U[:, j, :] * W[j? No.
                
                # Let's reshape W to (p, n, R).
                # We want sum_r U[i, j, r] * V[k, j, r] * W[k, i, r]
                # This is element-wise product of U[:, j, :], V[:, j, :] (transposed?), and W?
                # Let's compute M_r = U[:, j, r] * V[:, j, r] (shape n x p)
                # Then C += sum_r M_r * W[:, :, r] (where W is p x n x R)
                # Wait, W[k, i, r]. So W[:,:,r] is p x n.
                # M_r is n x p.
                # We need sum_r M_r[i, k] * W[k, i, r].
                # This is trace-like.
                # C[i, k] = sum_r (U[i, j, r] * V[k, j, r]) * W[k, i, r]
                
                # Let's compute T_j = np.einsum('ir,kr,ki->ik', U[:, j, :], V[:, j, :], W)
                # This should be fast.
                T_j = np.einsum('ir,kr,ki->ik', U[:, j, :], V[:, j, :], W)
                C += T_j
            
            # Target C is A @ B. But we are fitting the tensor directly.
            # The tensor T_{ijk} is 1 if it contributes to C.
            # Actually, the condition is:
            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
            # This is equivalent to:
            # For all i, k: sum_j sum_r U[i, j, r] V[j, k, r] W[k, i, r] = (A @ B)[i, k]
            # But we don't have A and B. We are decomposing the *multiplication tensor*.
            # The multiplication tensor T has T_{ijk} = 1 if the term A_{ij} B_{jk} is part of C_{ik}.
            # Since every A_{ij} B_{jk} contributes to C_{ik}, T_{ijk} = 1 for all i, j, k.
            
            # So the target for C[i, k] is sum_j 1 = m = 4?
            # No. The decomposition represents the *operation*.
            # C_{ik} = sum_j A_{ij} B_{jk}.
            # The tensor decomposition gives:
            # C_{ik} = sum_r (sum_j U_{ijr} A_{ij}) (sum_k? No.
            
            # Let's re-read the verification logic.
            # _verify_decomposition checks if C_reconstructed == A @ B.
            # C_reconstructed[i, k] = sum_j sum_r U[i, j, r] V[j, k, r] W[k, i, r] * A_{ij} B_{jk}?
            # No, the standard tensor decomposition for matmul is:
            # C_{ik} = sum_j A_{ij} B_{jk}
            # = sum_r (sum_j U_{ijr} A_{ij}) (sum_k? No.
            
            # The factors U, V, W are such that:
            # U_{ijr} is coefficient for A_{ij} in factor r.
            # V_{jkr} is coefficient for B_{jk} in factor r.
            # W_{kir} is coefficient for C_{ik} in factor r.
            
            # The condition is:
            # sum_r U_{ijr} V_{jkr} W_{kir} = 1 for all i, j, k.
            # This ensures that sum_r (sum_j U_{ijr} A_{ij}) (sum_k? No.
            # Let's check the product:
            # P_r = (sum_j U_{ijr} A_{ij}) * (sum_k? No, V is associated with B.
            # Usually:
            # M_r = sum_j U_{ijr} A_{ij}
            # N_r = sum_k? No.
            
            # Let's stick to the tensor identity:
            # sum_r U_{ijr} V_{jkr} W_{kir} = 1
            
            # So for fixed i, k, sum_j sum_r U_{ijr} V_{jkr} W_{kir} = sum_j 1 = m = 4.
            # Wait.
            # If sum_r U_{ijr} V_{jkr} W_{kir} = 1, then summing over j:
            # sum_j sum_r U_{ijr} V_{jkr} W_{kir} = sum_j 1 = 4.
            
            # So the target for C[i, k] in the "tensor space" is 4?
            # Let's verify with schoolbook.
            # U_{ijr} = 1 if r corresponds to (j, k)? No.
            # Schoolbook: r = (j, k).
            # U_{ij, jk} = 1 if j matches, else 0.
            # V_{jk, jk} = 1 if j,k matches, else 0.
            # W_{ki, jk} = 1 if k,i matches? No, W is indexed by k*n+i.
            # W_{k*n+i, jk} = 1 if (k, i) matches, else 0.
            
            # Sum over r=(j', k'):
            # U_{ij, j'k'} V_{jk, j'k'} W_{ki, j'k'}
            # Non-zero only if j'=j and k'=k.
            # Then term is 1 * 1 * 1 = 1.
            # So sum_r ... = 1.
            
            # So the condition is indeed sum_r U_{ijr} V_{jkr} W_{kir} = 1.
            # And summing over j gives 4.
            
            # So we want to minimize:
            # sum_{i, j, k} (sum_r U_{ijr} V_{jkr} W_{kir} - 1)^2
            
            # Let's compute the error directly using einsum.
            # T_ijk = sum_r U_{ijr} V_{jkr} W_{kir}
            # We can compute this as:
            # T = np.einsum('ijr,jkr,ki->ijk', U, V, W)
            
            T = np.einsum('ijr,jkr,ki->ijk', U, V, W)
            error = np.sum((T - 1.0) ** 2)
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if error < 1e-8:
                # Try to round
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_rounded = np.einsum('ijr,jkr,ki->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_rounded, 1.0, atol=1e-9):
                    return U_rounded.flatten().tolist(), V_rounded.flatten().tolist(), W_rounded.flatten().tolist()
            
            # ALS Step
            # Update U:
            # For fixed V, W, we want to minimize sum_{j,k} (sum_r U_{ijr} V_{jkr} W_{kir} - 1)^2
            # This is a linear least squares problem for U_{i, :, :} for each i?
            # Actually, for each i, j, the vector U_{ij, :} is coupled with V_{jk, :} and W_{ki, :}.
            
            # Let's do a simple gradient descent step with adaptive learning rate.
            # Gradient of error wrt U_{ijr}:
            # 2 * sum_{j,k} (T_{ijk} - 1) * V_{jkr} * W_{kir}
            
            grad_U = np.einsum('ijk,jkr,ki->ijr', T - 1.0, V, W)
            grad_V = np.einsum('ijk,ijr,ki->jkr', T - 1.0, U, W)
            grad_W = np.einsum('ijk,ijr,jkr->ki', T - 1.0, U, V)
            
            # Normalize gradients to step size
            norm_U = np.linalg.norm(grad_U)
            norm_V = np.linalg.norm(grad_V)
            norm_W = np.linalg.norm(grad_W)
            
            if norm_U > 1e-12:
                U -= 0.01 * grad_U / norm_U
            if norm_V > 1e-12:
                V -= 0.01 * grad_V / norm_V
            if norm_W > 1e-12:
                W -= 0.01 * grad_W / norm_W
                
    return None
>>>>>>> REPLACE

This change replaces the slow, non-vectorized search with a faster one using `np.einsum` and gradient descent with normalization. It also checks for exactness after rounding.

We also need to update `construct` to use the new function name and try a few ranks.

