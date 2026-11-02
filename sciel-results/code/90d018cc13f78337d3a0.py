"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix.
    
    Strategy:
    1. Build the matrix multiplication tensor T[a,b,c]
    2. Use SVD on different unfoldings to estimate minimal rank
    3. Try to construct valid decomposition at that rank
    4. Round to integer/half-integer entries and verify
    5. Fall back to schoolbook if rounding fails
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p  # 84

    # Build the tensor T[a, b, c] where:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a, b, c] = 1 if the indices are consistent (same i, j, k), else 0
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]"""
        # Compute the tensor from the decomposition
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Efficient computation: for each r, add outer product
        T_recon = np.zeros((nm, mp, pn), dtype=float)
        for r in range(U.shape[1]):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.max(np.abs(T_recon - T)) < tol

    def round_to_half_integer(x):
        """Round to nearest multiple of 0.5"""
        return np.round(2 * x) / 2.0

    def try_decomposition_at_rank(rank, U_init, V_init, W_init):
        """
        Try to refine a decomposition at given rank to have integer/half-integer entries.
        Returns (U, V, W) if valid, else None.
        """
        if time.time() > deadline:
            return None
        
        # First try direct rounding
        U_r = round_to_half_integer(U_init)
        V_r = round_to_half_integer(V_init)
        W_r = round_to_half_integer(W_init)
        
        if verify_decomposition(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        # Try scaling: if we scale V by a factor, we can adjust
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can scale U by alpha, V by beta, W by gamma where alpha*beta*gamma = 1
        # Try to find scalings that make entries closer to half-integers
        
        best_result = None
        best_err = float('inf')
        
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Random scaling factors
            alpha = np.random.uniform(0.5, 2.0)
            beta = np.random.uniform(0.5, 2.0)
            gamma = 1.0 / (alpha * beta)
            
            U_s = U_init * alpha
            V_s = V_init * beta
            W_s = W_init * gamma
            
            U_r = round_to_half_integer(U_s)
            V_r = round_to_half_integer(V_s)
            W_r = round_to_half_integer(W_s)
            
            if verify_decomposition(U_r, V_r, W_r):
                return U_r, V_r, W_r
        
        return None

    # Try to find minimal rank using SVD
    best_result = None
    
    # The tensor is sparse: only 84 non-zero entries
    # Let's try ranks from a lower bound up to schoolbook
    # Lower bound: max of matrix ranks of unfoldings
    
    # Unfolding 1: (nm, mp*pn)
    T_unfold1 = T.reshape(nm, mp * pn)
    # Unfolding 2: (nm*mp, pn)
    T_unfold2 = T.reshape(nm * mp, pn)
    # Unfolding 3: (mp, nm*pn)
    T_unfold3 = T.reshape(mp, nm * pn)
    
    try:
        rank_lb1 = np.linalg.matrix_rank(T_unfold1, tol=1e-10)
        rank_lb2 = np.linalg.matrix_rank(T_unfold2, tol=1e-10)
        rank_lb3 = np.linalg.matrix_rank(T_unfold3, tol=1e-10)
        rank_lb = max(rank_lb1, rank_lb2, rank_lb3)
    except Exception:
        rank_lb = 1

    rank_lb = max(1, int(rank_lb))
    
    # Try ranks from lower bound up to schoolbook
    # But be time-conscious
    max_rank_to_try = min(schoolbook_rank, rank_lb + 30)
    
    for rank in range(rank_lb, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        
        try:
            # Use SVD on unfolding 2: (nm*mp, pn)
            # T_unfold2[a*mp+b, c] = sum_r (U[a,r]*V[b,r]) * W[c,r]
            # Let M = U ⊙ V (Khatri-Rao), shape (nm*mp, rank)
            # Then T_unfold2 = M @ W^T
            # So we need M and W such that M @ W^T = T_unfold2
            # and M can be factored as U ⊙ V where U is (nm, rank), V is (mp, rank)
            
            # Get SVD of T_unfold2
            U_svd, s_vals, Vt_svd = np.linalg.svd(T_unfold2, full_matrices=False)
            
            # Truncate to desired rank
            if rank >= len(s_vals):
                rank = len(s_vals)
            if rank < 1:
                continue
                
            U_trunc = U_svd[:, :rank] * s_vals[:rank]
            Vt_trunc = Vt_svd[:rank, :]
            
            # Reshape U_trunc to (nm, rank) and Vt_trunc to (mp, rank)
            # U_trunc has shape (nm*mp, rank)
            # We need to factor it as U ⊙ V where U is (nm, rank), V is (mp, rank)
            # U_trunc[a*mp+b, r] = U[a,r] * V[b,r]
            
            # This is a rank-1 factorization problem for each r
            # For each r, U_trunc[:, r] should be outer product of U[:, r] and V[:, r]
            # But U_trunc[:, r] is a vector of length nm*mp
            # We need U[:, r] (length nm) and V[:, r] (length mp) such that
            # U[:, r] ⊗ V[:, r] = U_trunc[:, r]
            # This is possible iff U_trunc[:, r] has rank 1 as a (nm, mp) matrix
            
            # Reshape U_trunc[:, r] to (nm, mp)
            U_reshaped = U_trunc.reshape(nm, mp, rank)
            
            # For each r, check if U_reshaped[:, :, r] is rank-1
            # If yes, factor it
            U_fact = np.zeros((nm, rank))
            V_fact = np.zeros((mp, rank))
            
            feasible = True
            for r in range(rank):
                mat_r = U_reshaped[:, :, r]
                # Check rank
                if np.linalg.matrix_rank(mat_r, tol=1e-10) > 1:
                    feasible = False
                    break
                # Factor: mat_r = U[:, r] * V[:, r]^T
                # Use SVD
                try:
                    u_r, s_r, vt_r = np.linalg.svd(mat_r, full_matrices=False)
                    if len(s_r) > 0 and s_r[0] > 1e-12:
                        U_fact[:, r] = u_r[:, 0] * np.sqrt(s_r[0])
                        V_fact[:, r] = vt_r[0, :] * np.sqrt(s_r[0])
                    else:
                        U_fact[:, r] = 0
                        V_fact[:, r] = 0
                except Exception:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            W_fact = Vt_trunc.T  # shape (pn, rank)
            
            # Now we have U_fact, V_fact, W_fact
            # Check if they reconstruct T
            if verify_decomposition(U_fact, V_fact, W_fact):
                # Try to round to half-integers
                result = try_decomposition_at_rank(rank, U_fact, V_fact, W_fact)
                if result is not None:
                    U_r, V_r, W_r = result
                    best_result = (U_r, V_r, W_r, rank)
                    break  # Found valid decomposition at this rank
        except Exception:
            continue

    # If no better decomposition found, fall back to schoolbook
    if best_result is None:
        rank = schoolbook_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        
        best_result = (U, V, W, rank)

    U, V, W, rank = best_result
    
    # Convert to lists of lists for compatibility
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
