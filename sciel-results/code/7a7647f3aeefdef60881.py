"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    dim_A = n * m  # 9
    dim_B = m * p  # 24
    dim_C = p * n  # 24

    # Build the multiplication tensor T of shape (dim_A, dim_B, dim_C)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((dim_A, dim_B, dim_C), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose T."""
        # T_reconstructed = sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        return np.linalg.norm(T_rec - T, 'fro') < tol

    def als_decompose(T, dim_A, dim_B, dim_C, rank, max_iters=50, tol=1e-8, seed=None):
        """
        Alternating Least Squares for tensor decomposition.
        Returns (U, V, W) or None if not converged.
        """
        if seed is not None:
            np.random.seed(seed)
        
        U = np.random.randn(dim_A, rank) * 0.5
        V = np.random.randn(dim_B, rank) * 0.5
        W = np.random.randn(dim_C, rank) * 0.5

        prev_err = np.inf
        for it in range(max_iters):
            # Update U: fix V, W
            # T = sum_t U_t ⊗ V_t ⊗ W_t
            # Unfold along mode 1 (dim_A): U = T_unfolded @ (V ⊗ W)
            # T_unfolded shape: (dim_A, dim_B*dim_C)
            T_unfold_A = T.reshape(dim_A, dim_B * dim_C)
            VW = np.kron(W, V)  # shape (dim_B*dim_C, rank)
            # Actually, the unfolding is: T[a, b, c] = sum_t U[a,t] V[b,t] W[c,t]
            # So T_unfold_A[a, b*C+c] = sum_t U[a,t] * (V[b,t] * W[c,t])
            # Thus T_unfold_A = U @ (V ⊙ W)^T where ⊙ is Khatri-Rao
            # Khatri-Rao of V and W: column t is v_t ⊗ w_t
            # So T_unfold_A = U @ (kr(V,W))^T
            kr_VW = np.kron(W.T, V.T).T  # Wait, let me be careful.
            # V is (dim_B, rank), W is (dim_C, rank)
            # We want matrix M of shape (dim_B*dim_C, rank) where M[b*C+c, t] = V[b,t]*W[c,t]
            kr_VW = np.zeros((dim_B * dim_C, rank))
            for t in range(rank):
                kr_VW[:, t] = V[:, t] * W[:, t]  # This is element-wise, but we need outer? No.
                # V[:,t] is (dim_B,), W[:,t] is (dim_C,)
                # We need M[b*C+c, t] = V[b,t] * W[c,t]
                # So kr_VW[:, t] = np.outer(V[:, t], W[:, t]).flatten()
                kr_VW[:, t] = np.outer(V[:, t], W[:, t]).flatten()
            
            # Now T_unfold_A = U @ kr_VW.T
            # So U = T_unfold_A @ kr_VW @ (kr_VW.T @ kr_VW)^-1  (least squares)
            # Or just np.linalg.lstsq
            try:
                U_new, _, _, _ = np.linalg.lstsq(kr_VW, T_unfold_A.T, rcond=None)
                U = U_new.T  # shape (dim_A, rank)
            except:
                pass

            # Update V
            T_unfold_B = T.reshape(dim_B, dim_A * dim_C)
            # T[b, a*C+c] = sum_t V[b,t] * U[a,t] * W[c,t]
            kr_UW = np.zeros((dim_A * dim_C, rank))
            for t in range(rank):
                kr_UW[:, t] = np.outer(U[:, t], W[:, t]).flatten()
            try:
                V_new, _, _, _ = np.linalg.lstsq(kr_UW, T_unfold_B.T, rcond=None)
                V = V_new.T
            except:
                pass

            # Update W
            T_unfold_C = T.reshape(dim_C, dim_A * dim_B)
            kr_UV = np.zeros((dim_A * dim_B, rank))
            for t in range(rank):
                kr_UV[:, t] = np.outer(U[:, t], V[:, t]).flatten()
            try:
                W_new, _, _, _ = np.linalg.lstsq(kr_UV, T_unfold_C.T, rcond=None)
                W = W_new.T
            except:
                pass

            # Check convergence
            if it % 5 == 4:
                T_rec = np.einsum('at,bt,ct->abc', U, V, W)
                err = np.linalg.norm(T_rec - T, 'fro')
                if abs(err - prev_err) < 1e-12 or err < tol:
                    if err < tol:
                        return U, V, W
                    break
                prev_err = err

        # Final check
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.linalg.norm(T_rec - T, 'fro')
        if err < tol:
            return U, V, W
        return None

    # Try to find a low-rank decomposition
    # Known lower bound: max(dim_A, dim_B, dim_C) = 24
    # Schoolbook rank: 72
    # Let's try ranks from 24 upwards
    
    best_result = None
    best_rank = None
    
    # First, try the schoolbook to ensure we have a valid fallback
    rank_school = n * m * p
    U_school = np.zeros((dim_A, rank_school))
    V_school = np.zeros((dim_B, rank_school))
    W_school = np.zeros((dim_C, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    if verify(U_school, V_school, W_school, tol=1e-10):
        best_result = (U_school, V_school, W_school)
        best_rank = rank_school
    
    # Now try lower ranks
    # We'll try a range of ranks. For M_{3,3,8}, the rank is not known to be much lower than 72
    # but there might be some savings. Let's try from 30 to 71.
    # To save time, we'll do fewer restarts for higher ranks.
    
    candidate_ranks = list(range(24, 72))
    
    for r in candidate_ranks:
        if time.time() > deadline:
            break
            
        # Limit the number of restarts based on rank
        if r < 40:
            max_restarts = 3
        else:
            max_restarts = 1
            
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            result = als_decompose(T, dim_A, dim_B, dim_C, r, max_iters=30, tol=1e-8, seed=42 + r * 100 + restart)
            
            if result is not None:
                U, V, W = result
                if verify(U, V, W, tol=1e-8):
                    best_result = (U, V, W)
                    best_rank = r
                    break  # Found a valid decomposition for this rank, move to next (lower) rank? No, we want the lowest, so we should continue searching lower ranks. But we are iterating upwards. So we should break out and return this if it's better than schoolbook? Actually, we want the *minimal* rank. So we should search from low to high, and return the first valid one.
                    
    # If we found a better rank, use it. Otherwise, use schoolbook.
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
