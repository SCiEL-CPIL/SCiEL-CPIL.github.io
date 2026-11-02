import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Dimensions
    dim_u = n * m  # 6
    dim_v = m * p  # 30
    dim_w = p * n  # 20
    total_entries = dim_u * dim_v  # 180
    
    # Precompute indices for each gamma
    # gamma = k * n + i
    # Non-zero indices (alpha, beta) where alpha = i*m + j, beta = j*p + k
    # We store the flat index: alpha * dim_v + beta
    gamma_indices = {}
    for k in range(p):
        for i in range(n):
            gamma = k * n + i
            idxs = []
            for j in range(m):
                alpha = i * m + j
                beta = j * p + k
                flat_idx = alpha * dim_v + beta
                idxs.append(flat_idx)
            gamma_indices[gamma] = idxs
    
    # Create list of all flat indices
    all_indices = list(range(total_entries))
    
    # Function to check if a given U, V yields a solution
    def check_solution(U, V, R):
        # M is 180 x R, M[flat_idx, r] = U[alpha, r] * V[beta, r]
        # We can build M efficiently
        # U is (6, R), V is (30, R)
        # M_row_alpha_beta = U[alpha] * V[beta]
        
        # We need null space of M[Z_gamma, :] for each gamma
        # and solve M[I_gamma, :] @ w = 1
        
        # Optimization:
        # If R is small, we can just build the full M and slice.
        # M = np.einsum('ar,br->abr', U, V).reshape(-1, R)
        # This creates a (180, R) matrix.
        
        M = np.einsum('ar,br->abr', U, V).reshape(-1, R)
        
        W_rows = []
        for gamma in range(dim_w):
            I_gamma = gamma_indices[gamma]
            # Z_gamma is all indices except I_gamma
            # Mask for Z_gamma
            mask = np.ones(total_entries, dtype=bool)
            mask[I_gamma] = False
            Z_gamma_mask = mask
            
            M_Z = M[Z_gamma_mask, :]
            M_I = M[I_gamma, :]
            
            # Find null space of M_Z
            # M_Z is (177, R)
            # We want basis N such that M_Z @ N = 0
            # Use SVD
            try:
                U_svd, s, Vh = np.linalg.svd(M_Z, full_matrices=False)
            except np.linalg.LinAlgError:
                return None
            
            # Null space is Vh.T where s is close to 0
            tol = 1e-12
            rank = np.sum(s > tol)
            if rank < R:
                N = Vh[rank:].T  # (R, R-rank)
            else:
                N = np.zeros((R, 0))
            
            d = N.shape[1]
            if d < len(I_gamma): # Need at least 3 dimensions to satisfy 3 independent equations usually
                # If d < 3, it might still be possible if equations are dependent, but unlikely
                # Let's just try to solve
                pass
            
            if d == 0:
                # w must be 0, but M_I @ w = 1 is impossible
                return None
            
            A = M_I @ N  # (3, d)
            b = np.ones(len(I_gamma))
            
            # Solve A y = b
            try:
                # Least squares
                y, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check residual
                if len(residuals) > 0 and residuals[0] > 1e-9:
                    return None
                # Verify exactly
                if np.linalg.norm(A @ y - b) > 1e-9:
                    return None
            except:
                return None
            
            w = N @ y
            W_rows.append(w)
            
        W = np.array(W_rows).T  # (R, 20) -> transpose to (20, R)
        return W

    # Search for minimal R
    # Start from a lower bound.
    # Lower bound for 2x3x10 is likely > 20.
    # Let's try from 20 up to 60.
    rng = np.random.default_rng(42)
    
    best_result = None
    best_rank = n * m * p  # 60
    
    # Initialize with schoolbook if we fail to find better
    def make_schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    fallback = make_schoolbook()
    
    # Iterate R
    # We want to find the smallest R.
    # If we find one, we can stop?
    # The task says "Find a rank-minimal...".
    # If we find R=40, is it minimal?
    # We should check lower R first.
    # So loop R from low to high.
    # If we find a solution for R, return it.
    # If we reach 60 and haven't found, return schoolbook.
    
    # Heuristic: Start checking from 30?
    # Let's try 20 to 60.
    # To save time, we can limit attempts per R.
    
    attempts_per_rank = 200
    start_R = 20
    end_R = best_rank
    
    for R in range(start_R, end_R + 1):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(attempts_per_rank):
            if time.time() > deadline:
                break
                
            U = rng.standard_normal((dim_u, R))
            V = rng.standard_normal((dim_v, R))
            
            W = check_solution(U, V, R)
            if W is not None:
                # Verify solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            if abs(val - 1) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = R
                    found = True
                    break
        
        if found:
            # Found a solution for this R.
            # Since we iterate upwards, this is the minimal R found by this search.
            # We can break and return.
            break
    
    if best_result is None:
        best_result = fallback
        best_rank = n * m * p
        
    return best_result
