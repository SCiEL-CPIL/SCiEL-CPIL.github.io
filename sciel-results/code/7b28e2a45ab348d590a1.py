import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    Nm, Mp, Pn = n * m, m * p, p * n  # 12, 16, 12
    
    # Build target tensor
    T_target = np.zeros((Nm, Mp, Pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify_decomposition(U, V, W):
        """Verify that U,V,W exactly decompose the tensor."""
        R = U.shape[1]
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        UV = U[:, None, :] * V[None, :, :]  # (Nm, Mp, R)
        T_rec = np.einsum('abr,cr->abc', UV, W)
        return np.allclose(T_rec, T_target, atol=1e-6)
    
    def make_direct_decomposition():
        """Trivial rank-48 decomposition."""
        R = n * m * p
        U = np.zeros((Nm, R))
        V = np.zeros((Mp, R))
        W = np.zeros((Pn, R))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W
    
    def make_block_decomposition():
        """Current rank-44 block decomposition."""
        rank = 44
        U = np.zeros((Nm, rank))
        V = np.zeros((Mp, rank))
        W = np.zeros((Pn, rank))
        
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen part (cols 0-6)
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        # Extra part (cols 7-10) for Row 2 of A
        U_loc[4, 7] = 1
        V_loc[0, 7] = 1
        W_loc[4, 7] = 1
        
        U_loc[5, 8] = 1
        V_loc[2, 8] = 1
        W_loc[4, 8] = 1
        
        U_loc[4, 9] = 1
        V_loc[1, 9] = 1
        W_loc[5, 9] = 1
        
        U_loc[5, 10] = 1
        V_loc[3, 10] = 1
        W_loc[5, 10] = 1
        
        def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
        
        add_block(0, 0, 0, 0, 0, 0, 0)
        add_block(1, 0, 2, 2, 0, 0, 0)
        add_block(2, 0, 0, 0, 2, 0, 2)
        add_block(3, 0, 2, 2, 2, 0, 2)
        
        return U, V, W
    
    # Start with the known rank-44 solution
    best_U, best_V, best_W = make_block_decomposition()
    if not verify_decomposition(best_U, best_V, best_W):
        best_U, best_V, best_W = make_direct_decomposition()
    
    best_rank = best_U.shape[1]
    
    # Try to find a lower-rank decomposition using iterative refinement
    # Strategy: start with a slightly higher rank, then try to zero out components
    # and verify correctness
    
    # Approach: Use the fact that we can try rank R for R < best_rank
    # by solving a linear system or using gradient-based optimization
    
    # Try a structured approach: modify the block decomposition
    # to share more structure between blocks
    
    # Key idea: The 4 blocks each use rank 11. Can we reduce per-block rank?
    # Or can we share some rank-1 terms across blocks?
    
    # Let's try: for each pair of blocks, see if we can merge them
    # to save rank
    
    # Alternative: try rank 42, 41, 40 by solving the tensor decomposition
    # as a system of equations
    
    # For a given rank R, we need U (12×R), V (16×R), W (12×R) such that
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    # This is a system of Nm*Mp*Pn = 12*16*12 = 2304 equations
    # with 12*R + 16*R + 12*R = 40*R unknowns
    # For R=44: 1760 unknowns, 2304 equations (overdetermined)
    # The equations are nonlinear (bilinear in the factors)
    
    # Let's try a different approach: use the block structure but try to
    # find a more efficient 3x2x2 decomposition
    
    # Actually, let's try to directly search for a lower rank by
    # starting from the rank-44 solution and trying to eliminate terms
    
    # For each rank R from 43 down to 30:
    #   Try to find U,V,W of rank R that work
    
    # Use a simple iterative approach: start with random init, use ALS (alternating least squares)
    
    def try_rank_als(R, max_iter=200, tol=1e-8):
        """Try to find rank-R decomposition using ALS."""
        if time.time() > deadline - 5:
            return None
        
        # Random initialization with small values
        np.random.seed(42)
        U = np.random.randn(Nm, R) * 0.1
        V = np.random.randn(Mp, R) * 0.1
        W = np.random.randn(Pn, R) * 0.1
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each a, solve for U[a,:]
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # For fixed a: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
            # This is: T[a].reshape(-1) = sum_r U[a,r] * (V[:,r] * W[:,r]).reshape(-1)
            
            for a in range(Nm):
                # Build system: T[a,b,c] for all b,c
                target_a = T_target[a, :, :].reshape(-1)  # (Mp*Pn,)
                
                # Each rank-r term contributes U[a,r] * (V[:,r] * W[:,r]).reshape(-1)
                # So: target_a = sum_r U[a,r] * v_w_r
                # where v_w_r = (V[:,r] * W[:,r]).reshape(-1)
                
                # This is: target_a = V_W_matrix @ U[a,:]
                # where V_W_matrix[b,c,r] = V[b,r]*W[c,r], reshaped to (Mp*Pn, R)
                
                VW = (V[:, :, None] * W[None, :, :]).reshape(Mp * Pn, R)  # (Mp*Pn, R)
                
                # Solve: VW @ U[a,:] = target_a
                # Use least squares
                try:
                    U[a, :], residuals, rank, sv = np.linalg.lstsq(VW, target_a, rcond=None)
                except:
                    pass
            
            # Update V
            for b in range(Mp):
                target_b = T_target[:, b, :].reshape(-1)  # (Nm*Pn,)
                UW = (U[:, :, None] * W[None, :, :]).reshape(Nm * Pn, R)
                try:
                    V[b, :], residuals, rank, sv = np.linalg.lstsq(UW, target_b, rcond=None)
                except:
                    pass
            
            # Update W
            for c in range(Pn):
                target_c = T_target[:, :, c].reshape(-1)  # (Nm*Mp,)
                UV = (U[:, :, None] * V[None, :, :]).reshape(Nm * Mp, R)
                try:
                    W[c, :], residuals, rank, sv = np.linalg.lstsq(UV, target_c, rcond=None)
                except:
                    pass
            
            # Check convergence
            if iteration % 10 == 0:
                UV = U[:, None, :] * V[None, :, :]
                T_rec = np.einsum('abr,cr->abc', UV, W)
                err = np.max(np.abs(T_rec - T_target))
                if err < tol:
                    return U, V, W
        
        # Final check
        UV = U[:, None, :] * V[None, :, :]
        T_rec = np.einsum('abr,cr->abc', UV, W)
        if np.allclose(T_rec, T_target, atol=1e-6):
            return U, V, W
        return None
    
    # Try ranks below 44
    for R in range(43, 29, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_als(R, max_iter=100)
        if result is not None:
            U_cand, V_cand, W_cand = result
            if verify_decomposition(U_cand, V_cand, W_cand):
                best_U, best_V, best_W = U_cand, V_cand, W_cand
                best_rank = R
                break
    
    # If we found a better solution, round to nearest integers/half-integers if possible
    # and verify again
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W):
        # Fall back to the known good solution
        best_U, best_V, best_W = make_block_decomposition()
        if not verify_decomposition(best_U, best_V, best_W):
            best_U, best_V, best_W = make_direct_decomposition()
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
