"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses structured block decomposition combining Strassen for 2x2x2 blocks
and direct terms, targeting rank < 68. Also attempts randomized ALS
to find lower-rank exact decompositions."""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = 145  # seconds
    
    def check_time():
        return (time.time() - start_time) < deadline
    
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
    
    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    def try_rank_reduction(U_t, V_t, W_t, current_rank):
        """Try to reduce rank by finding linear dependencies among terms."""
        if len(U_t) <= 1:
            return U_t, V_t, W_t, len(U_t)
        
        # Build the coefficient matrices
        nm = n * m
        mp = m * p
        pn = p * n
        
        U_mat = np.zeros((nm, current_rank))
        V_mat = np.zeros((mp, current_rank))
        W_mat = np.zeros((pn, current_rank))
        
        for idx in range(current_rank):
            for (i, j), coeff in U_t[idx]:
                U_mat[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V_mat[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W_mat[k * n + i, idx] = coeff
        
        # Find the rank of the concatenated constraint system
        # We need U*V*W to preserve the tensor
        # This is complex, so we use a simpler approach:
        # Try to eliminate terms that are linear combinations of others
        
        # For now, just return the current decomposition
        # A more sophisticated approach would be needed
        return U_t, V_t, W_t, current_rank
    
    U_t = []
    V_t = []
    W_t = []
    
    # Use Strassen for the 2x2*2x2 blocks
    # A[0:2, 0:2] * B[0:2, k:k+2] for k in {0, 2, 4, 6}
    for k0 in [0, 2, 4, 6]:
        if not check_time():
            break
        strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)
    
    # Direct terms for A[0:2, 2] * B[2, :]
    for k0 in [0, 2, 4, 6]:
        for i in range(2):
            for k in range(k0, k0 + 2):
                if not check_time():
                    break
                direct_term(i, 2, k, U_t, V_t, W_t)
    
    # Direct terms for A[2, 0:2] * B[0:2, :]
    for k0 in [0, 2, 4, 6]:
        for j in range(2):
            for k in range(k0, k0 + 2):
                if not check_time():
                    break
                direct_term(2, j, k, U_t, V_t, W_t)
    
    # Direct terms for A[2, 2] * B[2, :]
    for k0 in [0, 2, 4, 6]:
        for k in range(k0, k0 + 2):
            if not check_time():
                break
            direct_term(2, 2, k, U_t, V_t, W_t)
    
    rank = len(W_t)
    
    # Try to reduce rank
    U_t, V_t, W_t, rank = try_rank_reduction(U_t, V_t, W_t, rank)
    
    # Build the factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def build_schoolbook():
        """Build the schoolbook (rank 72) decomposition as fallback."""
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result
    
    if not verify_decomp(U_arr, V_arr, W_arr):
        return build_schoolbook()
    
    # Now try randomized ALS to find a lower-rank decomposition
    best_result = {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
    best_rank = rank
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[i*m+j, j*p+k] -> W[k*n+i]
    # We need: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # The tensor can be viewed as a 3-way tensor T_{i,j,k} = 1 for all (i,j,k)
    # But with the specific indexing convention.
    
    # Actually, the multiplication tensor is:
    # T[(i,j), (j,k), (i,k)] = 1
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's define the tensor as a 3D array T[i, j, k] = 1
    # and we need factorizations U (n*m x r), V (m*p x r), W (p*n x r)
    
    # ALS approach:
    # Fix V and W, solve for U: U[:,r] = (V * W) / (V^T * W) type update
    # More precisely, for each r:
    #   U[:,r] should satisfy: for all (i,j,k): U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] sums to 1
    
    # This is a non-convex problem. Let's use a different approach:
    # We know the tensor T has rank at most 72 (schoolbook).
    # We try to find a lower-rank approximation.
    
    # Build the tensor as a matrix for ALS:
    # Reshape T into a matrix of size (n*m) x (m*p*n) where:
    # M[a, b] = T[a, b] where a = i*m+j, and b encodes (j,k) and (i,k)
    
    # Actually, let's use a simpler ALS:
    # The constraint is: sum_r U[:,r] * V[:,r] * W[:,r] (elementwise products summed) = T
    
    # Let's define:
    # For each r, we have vectors u_r (nm), v_r (mp), w_r (pn)
    # The tensor reconstruction is: T_recon[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    
    # We want T_recon = T where T[i,j,k] = 1
    
    # ALS: fix two, update the third
    # Fix V and W, update U:
    # For each r, we want to find u_r such that:
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    # This is a linear system in u_r for each r if we fix others.
    
    # Let's use a practical ALS:
    # 1. Initialize U, V, W randomly
    # 2. Iterate:
    #    a. Fix V, W. For each r, solve for u_r:
    #       The contribution of term r to T[i,j,k] is u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    #       The residual is T[i,j,k] - sum_{r' != r} u_r'[i*m+j] * v_r'[j*p+k] * w_r'[k*n+i]
    #       We want u_r to match this residual as closely as possible.
    #       This is: minimize ||residual - u_r * (v_r * w_r) ||
    #       where v_r * w_r is the elementwise product of v_r and w_r reshaped.
    
    # This is getting complex. Let's use a simpler approach:
    # Use the fact that the tensor is all-ones and try structured low-rank decompositions.
    
    # Alternative: try to find a decomposition of rank r by solving:
    # For r from some lower bound up to current best:
    #   Try random initializations and ALS
    
    # Let's implement a basic ALS:
    def als_decompose(target_tensor, n_dim, m_dim, p_dim, rank, max_iter=100, tol=1e-10, seed=42):
        """
        Try to find a rank-`rank` decomposition of the tensor.
        target_tensor: 3D array of shape (n, m, p)
        Returns (U, V, W) or None if failed.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n_dim * m_dim, rank) * 0.1
        V = rng.randn(m_dim * p_dim, rank) * 0.1
        W = rng.randn(p_dim * n_dim, rank) * 0.1
        
        for it in range(max_iter):
            # Update U
            for r in range(rank):
                # Compute the "effective" target for u_r
                # residual = target - sum_{r' != r} u_r' * v_r' * w_r' (elementwise in tensor space)
                # But we need to map this back to u_r space.
                
                # For each (i,j), the contribution to all k is:
                # u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
                # So for fixed r, u_r[i*m+j] should be:
                # sum_k target[i,j,k] * v_r[j*p+k] * w_r[k*n+i] / sum_k v_r[j*p+k]^2 * w_r[k*n+i]^2
                
                # Let's compute this:
                # For each i, j:
                #   num = sum_k target[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                #   den = sum_k V[j*p+k, r]^2 * W[k*n+i, r]^2
                #   U[i*m+j, r] = num / den (if den != 0)
                
                for i in range(n_dim):
                    for j in range(m_dim):
                        num = 0.0
                        den = 0.0
                        for k in range(p_dim):
                            vk = V[j * p_dim + k, r]
                            wk = W[k * n_dim + i, r]
                            num += target_tensor[i, j, k] * vk * wk
                            den += vk * vk * wk * wk
                        if abs(den) > 1e-15:
                            U[i * m_dim + j, r] = num / den
                        else:
                            U[i * m_dim + j, r] = 0.0
            
            # Update V
            for r in range(rank):
                for j in range(m_dim):
                    for k in range(p_dim):
                        num = 0.0
                        den = 0.0
                        for i in range(n_dim):
                            ui = U[i * m_dim + j, r]
                            wi = W[k * n_dim + i, r]
                            num += target_tensor[i, j, k] * ui * wi
                            den += ui * ui * wi * wi
                        if abs(den) > 1e-15:
                            V[j * p_dim + k, r] = num / den
                        else:
                            V[j * p_dim + k, r] = 0.0
            
            # Update W
            for r in range(rank):
                for k in range(p_dim):
                    for i in range(n_dim):
                        num = 0.0
                        den = 0.0
                        for j in range(m_dim):
                            uj = U[i * m_dim + j, r]
                            vj = V[j * p_dim + k, r]
                            num += target_tensor[i, j, k] * uj * vj
                            den += uj * uj * vj * vj
                        if abs(den) > 1e-15:
                            W[k * n_dim + i, r] = num / den
                        else:
                            W[k * n_dim + i, r] = 0.0
            
            # Check convergence
            recon = np.zeros((n_dim, m_dim, p_dim))
            for r in range(rank):
                for i in range(n_dim):
                    for j in range(m_dim):
                        for k in range(p_dim):
                            recon[i, j, k] += U[i * m_dim + j, r] * V[j * p_dim + k, r] * W[k * n_dim + i, r]
            
            err = np.max(np.abs(recon - target_tensor))
            if err < tol:
                return U, V, W, err
        
        return None, None, None, np.max(np.abs(recon - target_tensor))
    
    # Target tensor is all ones
    target = np.ones((n, m, p))
    
    # Try ranks from a lower bound up to current best - 1
    # Lower bound: at least max(n, m, p) = 8, but more realistically
    # For matrix multiplication, rank >= max(n, m, p) * something
    # Let's try from 40 up to best_rank - 1
    
    if check_time():
        # Try a few ranks with random restarts
        ranks_to_try = list(range(40, min(best_rank, 60)))
        # Sort by rank (ascending) to find the lowest first
        ranks_to_try.sort()
        
        for r in ranks_to_try:
            if not check_time():
                break
            
            # Try a few random restarts for this rank
            found = False
            for restart in range(3):
                if not check_time():
                    break
                
                U_als, V_als, W_als, err = als_decompose(
                    target, n, m, p, r, 
                    max_iter=50, tol=1e-10, 
                    seed=42 + restart * 1000
                )
                
                if U_als is not None and err < 1e-9:
                    # Verify
                    if verify_decomp(U_als, V_als, W_als):
                        # Convert to nice values
                        best_result = {
                            "U": to_nice_list(U_als),
                            "V": to_nice_list(V_als),
                            "W": to_nice_list(W_als)
                        }
                        best_rank = r
                        found = True
                        break
            
            if found:
                # Found a lower rank, continue trying even lower
                continue
            else:
                # If we couldn't find rank r, try lower ranks
                # But if r is close to the lower bound, stop
                if r <= 45:
                    break
    
    return best_result
# EVOLVE-BLOCK-END
