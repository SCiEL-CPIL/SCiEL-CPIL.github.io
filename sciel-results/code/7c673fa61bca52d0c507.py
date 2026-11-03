"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 7]
DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    tensor = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Fast verify using tensor contraction
    def verify_fast(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if U,V,W decompose the tensor using vectorized ops."""
        r = U_mat.shape[1]
        recon = np.einsum('ac,bc,dc->abd', U_mat, V_mat, W_mat)
        if np.max(np.abs(recon - tensor)) > tol:
            return False
        return True
    
    # Check schoolbook
    assert verify_fast(U, V, W), "Schoolbook failed verification"
    
    # Strategy: Search for lower rank decompositions using ALS and rounding
    # We try ranks from best_rank - 1 down to 20
    # For each rank, we try multiple random initializations
    
    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def als_decompose(T, R, n_iters=50, tol=1e-8):
        """
        Alternating Least Squares for CP decomposition.
        T is a 3D tensor.
        Returns U, V, W of shapes (I, R), (J, R), (K, R).
        """
        I, J, K = T.shape
        # Initialize randomly
        U = np.random.randn(I, R)
        V = np.random.randn(J, R)
        W = np.random.randn(K, R)
        
        prev_err = np.inf
        for _ in range(n_iters):
            # Update U
            # For each r, U[:, r] = argmin_U sum_{a,b,c} (T_abc - U_ar V_br W_cr)^2
            # This is equivalent to U = T x2 (W x1 V) * (V^T V * W^T W)^{-1} roughly?
            # Actually, for fixed V, W, the objective is sum_r || T - U_r V_r^T W_r^T ||^2
            # Let's use the standard ALS update:
            # U = T x2 (W * V) / (V^T V * W^T W) -- this is for symmetric? No.
            
            # Standard CP ALS:
            # U = (T x2 (V * W)) @ inv( (V^T V) * (W^T W) )? No.
            # Let's derive:
            # Minimize sum_{a,b,c} (T_abc - sum_r U_ar V_br W_cr)^2 w.r.t U_ar.
            # Derivative w.r.t U_ar:
            # -2 sum_{b,c} T_abc V_br W_cr + 2 sum_{b,c} (sum_r' U_ar' V_br' W_cr') V_br W_cr = 0
            # sum_{b,c} T_abc V_br W_cr = sum_{r'} U_ar' (sum_b V_br' V_br) (sum_c W_cr' W_cr)
            # Let M_r' = (sum_b V_br' V_br) (sum_c W_cr' W_cr) = (V^T V)_{r'r} (W^T W)_{r'r} ? No, indices.
            # (V^T V) is R x R. (W^T W) is R x R.
            # The term is sum_{r'} U_ar' (V^T V)_{r' r} (W^T W)_{r' r} ? No.
            # It is sum_{r'} U_ar' (sum_b V_br' V_br) (sum_c W_cr' W_cr)
            # Let G_r'r = (V^T V)_{r'r} (W^T W)_{r'r}? No, this is element-wise product?
            # No, the sum is over b and c separately.
            # sum_b V_br' V_br = (V^T V)_{r'r}
            # sum_c W_cr' W_cr = (W^T W)_{r'r}
            # So the equation is:
            # (T x2 (V * W))_{ar} = sum_{r'} U_ar' (V^T V)_{r'r} (W^T W)_{r'r} ??
            # Wait, the indices in the sum are r'.
            # LHS: sum_{b,c} T_abc V_br W_cr. This is (T x2 (V * W))_{ar}.
            # RHS: sum_{r'} U_ar' (sum_b V_br' V_br) (sum_c W_cr' W_cr).
            # Let A = V^T V, B = W^T W.
            # RHS = sum_{r'} U_ar' A_{r'r} B_{r'r}.
            # This is not a standard matrix product. It is (U * (A * B)) @ 1?
            # Actually, let's just solve the linear system directly for each column r of U.
            # Or use a simpler approach:
            # U = T x2 (W x1 V) * (V^T V * W^T W)^{-1} is incorrect.
            
            # Let's use a robust numerical solver for the linear system.
            # For fixed V, W, we want to find U.
            # The tensor can be unfolded.
            # Let's just use the fact that we can flatten the problem.
            # But R is small.
            
            # Efficient ALS update:
            # U = T x2 (W * V) @ inv( (V^T V) * (W^T W) ) is wrong.
            # Correct update:
            # U = (T x2 (W * V)) @ inv( (V^T V) * (W^T W) ) is for a specific case?
            
            # Let's stick to a simpler, albeit slower, method if R is small.
            # Or use the formula:
            # U = T x2 (W * V) / (V^T V * W^T W) is not right.
            
            # Let's use the standard formula from literature:
            # U = T x2 (W * V) @ inv( (V^T V) * (W^T W) )
            # Wait, (V^T V) * (W^T W) is Hadamard product?
            # Yes, the Gram matrix of the columns of U is (V^T V) * (W^T W) (Hadamard).
            # So we solve:
            # (V^T V) * (W^T W) U^T = (T x2 (W * V))^T
            # U = ((T x2 (W * V))^T @ inv( (V^T V) * (W^T W) ))^T
            
            VW = V * W # Element-wise product, shape (J, K, R)? No, V is (J, R), W is (K, R).
            # We need T x2 (V * W).
            # T is (I, J, K).
            # We want to contract T with V along J and W along K.
            # This is T x2 (V * W) where V*W is a tensor of shape (J, K, R)?
            # No, V is (J, R), W is (K, R).
            # The term is sum_{b,c} T_abc V_br W_cr.
            # This is (T x2 (V * W))_{ar} where (V*W)_{bcr} = V_{br} W_{cr}.
            
            # Let's compute T x2 (V * W)
            # We can do this by reshaping.
            # T_flat: (I, J*K).
            # (V * W) reshaped: (J*K, R).
            # Then T_flat @ (V*W)_flat gives (I, R).
            
            V_W_flat = (V * W).T.reshape(J*K, R) # (V*W) is (J, K, R)? No.
            # V is (J, R), W is (K, R).
            # We want a matrix M of shape (J*K, R) such that M_{bK+c, r} = V_{br} W_{cr}.
            # This is the Kronecker product? No.
            # M = V.reshape(J, 1, R) * W.reshape(1, K, R) -> (J, K, R) -> (J*K, R).
            
            M = (V[:, :, np.newaxis] * W[np.newaxis, :, :]).reshape(J*K, R)
            # Wait, V is (J, R). V[:, :, np.newaxis] is (J, 1, R).
            # W is (K, R). W[np.newaxis, :, :] is (1, K, R).
            # Product is (J, K, R).
            # Reshape to (J*K, R).
            
            T_flat = T.reshape(I, J*K)
            # We want U such that U @ M^T approx T_flat?
            # No, T_flat_{a, bK+c} = sum_r U_{ar} M_{bK+c, r}.
            # So T_flat = U @ M^T.
            # We want to find U.
            # U = T_flat @ inv(M^T M).
            # M^T M is R x R.
            # M^T M = M.T @ M.
            
            try:
                M_T_M = M.T @ M
                # Regularize if needed
                M_T_M += 1e-8 * np.eye(R)
                U_new = T_flat @ np.linalg.inv(M_T_M)
            except np.linalg.LinAlgError:
                U_new = U
            
            # Update V
            # Similarly, V = T x1 (W * U) ...
            # T x1 (W * U)_{b, r} = sum_{a,c} T_abc W_cr U_ar.
            # M_V = (W * U) reshaped.
            # W is (K, R), U is (I, R).
            # M_V_{cK+a? No, index is c*I + a?}
            # T is (I, J, K).
            # We contract I and K.
            # T reshape to (J, I*K).
            # M_V shape (I*K, R) where M_V_{aI+c, r} = U_{ar} W_{cr}.
            M_V = (U[:, :, np.newaxis] * W[np.newaxis, :, :]).reshape(I*K, R)
            T_flat_V = T.reshape(J, I*K)
            try:
                M_V_T_M_V = M_V.T @ M_V
                M_V_T_M_V += 1e-8 * np.eye(R)
                V_new = T_flat_V @ np.linalg.inv(M_V_T_M_V)
            except np.linalg.LinAlgError:
                V_new = V
                
            # Update W
            # W = T x0 (V * U) ...
            # T x0 (V * U)_{c, r} = sum_{a,b} T_abc V_br U_ar.
            # T reshape to (K, I*J).
            # M_W shape (I*J, R) where M_W_{aJ+b, r} = U_{ar} V_{br}.
            M_W = (U[:, :, np.newaxis] * V[np.newaxis, :, :]).reshape(I*J, R)
            T_flat_W = T.reshape(K, I*J)
            try:
                M_W_T_M_W = M_W.T @ M_W
                M_W_T_M_W += 1e-8 * np.eye(R)
                W_new = T_flat_W @ np.linalg.inv(M_W_T_M_W)
            except np.linalg.LinAlgError:
                W_new = W
                
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            recon = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.linalg.norm(recon - T)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W

    # Search loop
    # Try ranks from 63 down to 20
    for R in range(best_rank - 1, 19, -1):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Try multiple random seeds
        for seed in range(20):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            np.random.seed(seed)
            U_try, V_try, W_try = als_decompose(tensor, R, n_iters=30)
            
            # Round to nearest half-integer
            U_rounded = round_to_half(U_try)
            V_rounded = round_to_half(V_try)
            W_rounded = round_to_half(W_try)
            
            if verify_fast(U_rounded, V_rounded, W_rounded, tol=1e-9):
                best_rank = R
                best_U = U_rounded
                best_V = V_rounded
                best_W = W_rounded
                break # Found a decomposition for this rank, try lower
        
        # If we found a solution for R, we can break if we want the minimal rank found so far?
        # No, we want the MINIMAL rank. So we continue down.
        # But if we found one for R, we can stop trying higher ranks.
        # The loop goes down, so if we find one for R, we continue to R-1.
        
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
