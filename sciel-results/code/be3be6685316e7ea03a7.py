"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Schoolbook decomposition as baseline (rank = n*m*p = 56)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    def tensor_from_factors(U, V, W):
        return np.einsum('ad,bd,cd->abc', U, V, W)
        
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    def verify_and_round(U, V, W):
        """Check if factors exactly reconstruct T, try to round to half-integers."""
        T_rec = tensor_from_factors(U, V, W)
        if np.allclose(T_rec, T, atol=1e-9, rtol=1e-9):
            # Try rounding to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            T_rec_r = tensor_from_factors(U_r, V_r, W_r)
            if np.allclose(T_rec_r, T, atol=1e-9, rtol=1e-9):
                return U_r, V_r, W_r, True
            else:
                # Check if they are already effectively integers/half-integers
                # If not, return the float version which is valid if exact
                return U, V, W, True
        return None, None, None, False

    def try_rank_alt(r, n_trials=5, max_iter=200):
        """Alternating minimization with efficient matrix construction."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for trial in range(n_trials):
            if time.time() > deadline:
                return False
            
            # Initialize with random Gaussian
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Precompute T views for efficiency
            # T is (nm, mp, pn)
            # For W update: T_vec_w[c, a*mp+b] = T[a, b, c]
            T_vec_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            # For U update: T_vec_u[a, b*pn+c] = T[a, b, c]
            T_vec_u = T.transpose(0, 2, 1).reshape(nm, mp * pn)
            # For V update: T_vec_v[b, a*pn+c] = T[a, b, c]
            T_vec_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)

            for it in range(max_iter):
                if time.time() > deadline:
                    return False
                
                # 1. Update W
                # We want to minimize || T - U V^T W^T ||_F (in flattened sense)
                # T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                # For fixed U, V: T_vec_w[c, :] = sum_d W[c,d] * (U[:,d] * V[:,d])_flat
                # Let X_w be (nm*mp, r) where X_w[a*mp+b, d] = U[a,d] * V[b,d]
                X_w = np.zeros((nm * mp, r))
                for d in range(r):
                    X_w[:, d] = (U[:, d][:, None] * V[:, d][None, :]).flatten()
                
                # Solve W: T_vec_w = W @ X_w.T  =>  T_vec_w.T = X_w @ W.T
                # W.T = (X_w.T X_w)^-1 X_w.T T_vec_w.T
                # W = (T_vec_w X_w (X_w.T X_w)^-1)
                # Actually: T_vec_w (pn, nm*mp) = W (pn, r) @ X_w.T (r, nm*mp)
                # So W = T_vec_w @ X_w @ pinv(X_w.T @ X_w) ... no
                # Let's use least squares: X_w @ W.T = T_vec_w.T
                # W.T = lstsq(X_w, T_vec_w.T)
                try:
                    W_new = np.linalg.lstsq(X_w, T_vec_w.T, rcond=None)[0].T
                except:
                    break
                W = W_new

                # 2. Update U
                # T[a,b,c] = sum_d U[a,d] * (V[:,d] * W[:,d])_flat[b*pn+c]
                # T_vec_u[a, :] = sum_d U[a,d] * (V[:,d] * W[:,d])_flat
                # Let X_u be (mp*pn, r) where X_u[b*pn+c, d] = V[b,d] * W[c,d]
                X_u = np.zeros((mp * pn, r))
                for d in range(r):
                    X_u[:, d] = (V[:, d][:, None] * W[:, d][None, :]).flatten()
                
                # T_vec_u (nm, mp*pn) = U (nm, r) @ X_u.T (r, mp*pn)
                try:
                    U_new = np.linalg.lstsq(X_u, T_vec_u.T, rcond=None)[0].T
                except:
                    break
                U = U_new

                # 3. Update V
                # T[a,b,c] = sum_d V[b,d] * (U[:,d] * W[:,d])_flat[a*pn+c]
                # T_vec_v[b, :] = sum_d V[b,d] * (U[:,d] * W[:,d])_flat
                # Let X_v be (nm*pn, r) where X_v[a*pn+c, d] = U[a,d] * W[c,d]
                X_v = np.zeros((nm * pn, r))
                for d in range(r):
                    X_v[:, d] = (U[:, d][:, None] * W[:, d][None, :]).flatten()
                
                # T_vec_v (mp, nm*pn) = V (mp, r) @ X_v.T (r, nm*pn)
                try:
                    V_new = np.linalg.lstsq(X_v, T_vec_v.T, rcond=None)[0].T
                except:
                    break
                V = V_new
                
                # Check error periodically to save time
                if it % 20 == 0 or it == max_iter - 1:
                    T_rec = tensor_from_factors(U, V, W)
                    err = np.linalg.norm(T_rec - T)
                    if err < 1e-12:
                        break

            # Final verification
            U_final, V_final, W_final, success = verify_and_round(U, V, W)
            if success:
                best_U, best_V, best_W = U_final, V_final, W_final
                best_rank = r
                return True
                
        return False

    def try_rank_strassen_embed(r, n_trials=3):
        """
        Structural approach: Embed 2x4x4 into 3x3x3 (Strassen rank 7)
        and handle the 4x7 part.
        This is a heuristic search for specific structured solutions.
        """
        nonlocal best_U, best_V, best_W, best_rank
        
        # For 2x4x4:
        # A is 2x4, B is 4x4, C is 2x4.
        # We can split A into A1 (2x2), A2 (2x2).
        # B into B11, B12, B21, B22 (2x2 blocks).
        # C into C1 (2x2), C2 (2x2).
        # C1 = A1 B11 + A2 B21
        # C2 = A1 B12 + A2 B22
        # If we use Strassen for 2x2x2, each 2x2 mult takes rank 7.
        # But they share A1, A2.
        # This is complex. Let's just rely on the alternating minimization
        # but try starting from a "structured" initialization if possible.
        
        # Since the problem is small, alternating minimization is the most robust
        # "structural" search if we allow enough iterations and restarts.
        # The "inspiration" was to use known decompositions.
        # For 2x4x7, the rank is likely close to 56 (schoolbook) or slightly less.
        # Known rank for 2x4x4 is 20? No, 2x4x4 is a subcase of 2x4x4.
        # Let's just use the efficient alternating minimization.
        
        return False

    # Search strategy
    # Start from schoolbook rank and go down.
    # Or start from a lower bound and go up?
    # Lower bound for matrix multiplication rank R(n,m,p) >= max(n*m, m*p, p*n) = 28.
    # But it's usually higher.
    # Let's try ranks in a specific order.
    # Schoolbook is 56.
    # Try 50, 52, 54, 55, 56.
    
    ranks_to_try = [50, 52, 54, 55, 56]
    # Also try some lower ones in case of luck, but 50+ is more likely for exact integer/half-integer.
    # Actually, let's try from 45 to 56.
    
    # To save time, we prioritize higher ranks first as they are easier to solve?
    # No, lower rank is the goal. But lower rank is harder to find.
    # If we find a valid rank 50, we are done.
    # Let's try 50, 51, 52, 53, 54, 55.
    
    # Randomize the order slightly to avoid getting stuck in a local min of the search space?
    # No, deterministic is better for reproducibility in evaluation.
    # But random initialization in the solver helps.
    
    # Let's try ranks from 48 to 56.
    # If we find one, break.
    
    # Time budget allocation:
    # ~140s total.
    # 9 ranks. ~15s per rank.
    # Each rank: 5 trials. ~3s per trial.
    # Each trial: ~200 iterations.
    # Iteration cost: 3 lstsq calls on matrices of size (nm*mp, r) ~ (56, r).
    # nm*mp = 8*28 = 224.
    # lstsq on (224, r) with r~50 is very fast.
    # So we can afford more iterations or trials.
    
    # Let's increase trials for lower ranks.
    
    for r in range(48, 57):
        if time.time() > deadline:
            break
        
        # Adjust trials based on remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # 5 trials per rank
        success = try_rank_alt(r, n_trials=5, max_iter=300)
        if success:
            break

    # If we didn't find anything better than schoolbook, return schoolbook.
    # The schoolbook is always valid.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
