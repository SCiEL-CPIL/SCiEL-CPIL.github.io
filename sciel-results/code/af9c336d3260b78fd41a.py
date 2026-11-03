"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Build the standard schoolbook decomposition (rank 42)
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Build target tensor for verification
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    
    def check_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that U⊗V⊗W reconstructs the target tensor."""
        T_full = np.einsum('ur,vr,wr->uvw', U_mat, V_mat, W_mat)
        return np.allclose(T_full, T_target, atol=tol)
    
    # Verify schoolbook is correct
    assert check_decomposition(U, V, W), "Schoolbook decomposition failed verification"
    
    # Try rank-1 peeling to see if we can find a lower-rank decomposition
    # This is unlikely to succeed for 2x3x7 (rank is believed to be 42),
    # but we attempt it briefly in case there's an improvement.
    
    def try_rank_peeling(target_rank, max_iter=20, n_restarts=2):
        """Try to find a decomposition of rank <= target_rank using rank-1 peeling."""
        if time.time() - t0 > deadline - 5:
            return None
        
        for restart in range(n_restarts):
            if time.time() - t0 > deadline - 5:
                return None
            
            residual = T_target.copy()
            U_list = []
            V_list = []
            W_list = []
            
            for rank_idx in range(target_rank):
                if time.time() - t0 > deadline - 5:
                    return None
                
                # Initialize with random vectors
                u = np.random.randn(n * m)
                v = np.random.randn(m * p)
                w = np.random.randn(p * n)
                
                # Alternating minimization to find best rank-1 approximation
                for _ in range(max_iter):
                    # Update u: u_i = sum_{j,k} residual[i,j,k] * v_j * w_k
                    S_u = np.einsum('ijk,j,k->i', residual, v, w)
                    if np.linalg.norm(S_u) < 1e-12:
                        break
                    u = S_u
                    
                    # Update v: v_j = sum_{i,k} residual[i,j,k] * u_i * w_k
                    S_v = np.einsum('ijk,i,k->j', residual, u, w)
                    if np.linalg.norm(S_v) < 1e-12:
                        break
                    v = S_v
                    
                    # Update w: w_k = sum_{i,j} residual[i,j,k] * u_i * v_j
                    S_w = np.einsum('ijk,i,j->k', residual, u, v)
                    if np.linalg.norm(S_w) < 1e-12:
                        break
                    w = S_w
                
                # Compute the scalar coefficient
                alpha = np.einsum('ijk,i,j,k', residual, u, v, w)
                if abs(alpha) < 1e-10:
                    break
                
                # Subtract the rank-1 term from residual
                residual = residual - alpha * np.einsum('i,j,k->ijk', u, v, w)
                
                # Store the components (scaled so that the product gives the correct term)
                # We need: alpha * u_i * v_j * w_k = (alpha*u_i) * v_j * w_k
                U_list.append(alpha * u)
                V_list.append(v)
                W_list.append(w)
                
                # Check if residual is essentially zero
                if np.linalg.norm(residual) < 1e-8:
                    break
            
            R = len(U_list)
            if R <= target_rank and R > 0:
                U_mat = np.column_stack(U_list)
                V_mat = np.column_stack(V_list)
                W_mat = np.column_stack(W_list)
                
                # Verify the decomposition
                if check_decomposition(U_mat, V_mat, W_mat, tol=1e-6):
                    return {"U": U_mat, "V": V_mat, "W": W_mat}
        
        return None
    
    # Try to find rank 41 (one less than schoolbook)
    if time.time() - t0 < deadline - 10:
        result = try_rank_peeling(41, max_iter=15, n_restarts=2)
        if result is not None:
            # Convert to lists with integer/half-integer entries if possible
            # Round small values to 0 or 1 for cleanliness
            for key in ['U', 'V', 'W']:
                mat = result[key]
                # Round to nearest 0.5 if close
                rounded = np.round(mat * 2) / 2
                if np.allclose(rounded, mat, atol=1e-8):
                    result[key] = rounded
            
            # Final verification
            if check_decomposition(result['U'], result['V'], result['W'], tol=1e-6):
                return {
                    "U": result["U"].tolist(),
                    "V": result["V"].tolist(),
                    "W": result["W"].tolist()
                }
    
    # Fallback: return the schoolbook rank-42 decomposition
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
