"""Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # The multiplication tensor T ∈ R^{n*m × m*p × p*n}
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # where i∈[0,n), j∈[0,m), k∈[0,p)
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor
    T_tensor = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_tensor[a, b, c] = 1.0
    
    start_time = time.time()
    deadline = 140  # Leave 10s margin before 150s kill
    
    # Strategy: Use a structured approach exploiting the small n=2 dimension
    # 
    # Key insight: The output C has shape (2, 7), so pn = 14.
    # The input A has shape (2, 4), so nm = 8.
    # The input B has shape (4, 7), so mp = 28.
    #
    # We can decompose the multiplication as:
    # C = A @ B
    # 
    # Let's try a hybrid approach:
    # 1. Use Strassen-like block decomposition where applicable
    # 2. Fill in with rank-1 terms
    #
    # For 2×4 × 4×7:
    # We can think of this as 2×(4×7) or (2×4)×7
    # 
    # Actually, let me try a direct algebraic approach:
    # Since n=2 is small, we can parameterize the decomposition.
    # 
    # The tensor has 8*28*14 = 3136 entries, only 56 of which are 1.
    # We want T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    #
    # Let me try to find a low-rank decomposition by exploiting structure.
    #
    # Approach: Try rank from a lower bound up, using randomized ALS with 
    # multiple restarts and integer/half-integer rounding.
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        R = U.shape[1]
        T_approx = np.zeros((nm, mp, pn), dtype=np.float64)
        for r_idx in range(R):
            T_approx += np.outer(np.outer(U[:, r_idx], V[:, r_idx]), W[:, r_idx])
        return np.allclose(T_tensor, T_approx, atol=1e-8)
    
    def round_to_half(M):
        """Round matrix entries to nearest 0.5 and check validity."""
        M_r = np.round(M * 2) / 2
        if np.allclose(M, M_r, atol=1e-6):
            return M_r
        return None
    
    def try_als(rank, n_restarts=5, max_iters=80, seed=42):
        """Try ALS to find a rank-r decomposition."""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            rng = np.random.RandomState(seed + restart)
            U = rng.randn(nm, rank) * 0.1
            V = rng.randn(mp, rank) * 0.1
            W = rng.randn(pn, rank) * 0.1
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U: for each a, solve T[a,:,:] = U[a,:] @ (V⊙W)^T
                # Build basis: B_u[b*p_n + c, r] = V[b,r]*W[c,r]
                B_u = np.zeros((mp * pn, rank), dtype=np.float64)
                for r_idx in range(rank):
                    B_u[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx])
                
                T_u = T_tensor.reshape(nm, mp * pn)
                # T_u[a, :] = U[a, :] @ B_u.T
                try:
                    U_new = T_u @ np.linalg.pinv(B_u.T)
                except np.linalg.LinAlgError:
                    break
                
                # Update V
                B_v = np.zeros((nm * pn, rank), dtype=np.float64)
                for r_idx in range(rank):
                    B_v[:, r_idx] = np.kron(U[:, r_idx], W[:, r_idx])
                
                T_v = T_tensor.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V_new = T_v @ np.linalg.pinv(B_v.T)
                except np.linalg.LinAlgError:
                    break
                
                # Update W
                B_w = np.zeros((nm * mp, rank), dtype=np.float64)
                for r_idx in range(rank):
                    B_w[:, r_idx] = np.kron(U[:, r_idx], V[:, r_idx])
                
                T_w = T_tensor.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_new = T_w @ np.linalg.pinv(B_w.T)
                except np.linalg.LinAlgError:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if it % 5 == 0 or it == max_iters - 1:
                    T_approx = np.zeros((nm, mp, pn), dtype=np.float64)
                    for r_idx in range(rank):
                        T_approx += np.outer(np.outer(U[:, r_idx], V[:, r_idx]), W[:, r_idx])
                    error = np.linalg.norm(T_tensor - T_approx) / max(np.linalg.norm(T_tensor), 1e-12)
                    if error < 1e-10:
                        # Try rounding
                        U_r = round_to_half(U)
                        V_r = round_to_half(V)
                        W_r = round_to_half(W)
                        if U_r is not None and V_r is not None and W_r is not None:
                            if verify(U_r, V_r, W_r):
                                return (U_r, V_r, W_r)
                        # Try integer rounding
                        U_i = np.round(U)
                        V_i = np.round(V)
                        W_i = np.round(W)
                        if np.allclose(U, U_i, atol=1e-6) and np.allclose(V, V_i, atol=1e-6) and np.allclose(W, W_i, atol=1e-6):
                            if verify(U_i, V_i, W_i):
                                return (U_i, V_i, W_i)
                        # Return as-is if close enough
                        if error < 1e-12:
                            return (U, V, W)
        
        return None
    
    # Try ranks from a reasonable lower bound
    # Theoretical lower bound: at least max(nm, mp, pn) isn't quite right
    # For matrix mult n×m × m×p, rank is at least... 
    # Actually, a lower bound is max(n*m, m*p, p*n) is not correct.
    # The tensor has rank at least the max of the matricization ranks.
    # Let's just try from rank 20 up.
    
    best_result = None
    
    for rank in range(20, 57):
        if time.time() > deadline:
            break
        
        result = try_als(rank, n_restarts=8, max_iters=60)
        if result is not None:
            U, V, W = result
            # Final verification
            if verify(U, V, W):
                best_result = (rank, U, V, W)
                break
    
    if best_result is not None:
        rank, U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook decomposition with rank = n*m*p = 56
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
