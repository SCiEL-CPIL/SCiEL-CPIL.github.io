import time
import sys
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen 2x2 * 2x2 decomposition (rank 7)
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ], dtype=float)
    
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [0, 0, 0, 1, 0, 0, 1],   # b10
        [1, 0, -1, 0, 1, 0, 1]   # b11
    ], dtype=float)
    
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ], dtype=float)
    
    # Global mapping for A1 (cols 0,1 of A): local (i,j) -> global i*4+j
    a1_map = [0, 1, 4, 5]
    # Global mapping for A2 (cols 2,3 of A): local (i,j) -> global i*4+2+j
    a2_map = [2, 3, 6, 7]
    
    def verify(U, V, W, rank):
        # Vectorized verification
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # We need T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        # Non-zero entries are at (i,j), (j,k), (k,i) for all i,j,k.
        # There are n*m*p such entries.
        # We can construct the target tensor and compare.
        
        # Construct target tensor T (sparse)
        # Indices:
        # u_idx = i*m + j
        # v_idx = j*p + k
        # w_idx = k*n + i
        
        u_indices = []
        v_indices = []
        w_indices = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_indices.append(i*m + j)
                    v_indices.append(j*p + k)
                    w_indices.append(k*n + i)
        
        u_indices = np.array(u_indices)
        v_indices = np.array(v_indices)
        w_indices = np.array(w_indices)
        
        # Calculate actual values
        # actual[t] = sum_r U[u_indices[t], r] * V[v_indices[t], r] * W[w_indices[t], r]
        
        # This can be done as:
        # M1 = U[u_indices, :] * V[v_indices, :]  (shape: (N, rank))
        # M2 = M1 * W[w_indices, :] (shape: (N, rank))
        # actual = sum(M2, axis=1)
        
        M1 = U[u_indices, :] * V[v_indices, :]
        M2 = M1 * W[w_indices, :]
        actual = np.sum(M2, axis=1)
        
        if np.max(np.abs(actual - 1.0)) > 1e-8:
            return False
        return True

    def try_rank(rank):
        if time.time() - start_time > 140:
            return None
        
        # Initialize with Strassen if possible, else random
        if rank >= 56:
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Build the 56-rank Strassen solution
            for c_pair in range(4):
                base_r = c_pair * 7
                b1_map = [0*8 + 2*c_pair, 0*8 + 2*c_pair + 1,
                          1*8 + 2*c_pair, 1*8 + 2*c_pair + 1]
                c_map = [(2*c_pair)*2 + 0, (2*c_pair+1)*2 + 0,
                         (2*c_pair)*2 + 1, (2*c_pair+1)*2 + 1]
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        if U_local[la, l] != 0:
                            U[a1_map[la], r] += U_local[la, l]
                    for lb in range(4):
                        if V_local[lb, l] != 0:
                            V[b1_map[lb], r] += V_local[lb, l]
                    for lc in range(4):
                        if W_local[lc, l] != 0:
                            W[c_map[lc], r] += W_local[lc, l]
            
            for c_pair in range(4):
                base_r = 28 + c_pair * 7
                b2_map = [2*8 + 2*c_pair, 2*8 + 2*c_pair + 1,
                          3*8 + 2*c_pair, 3*8 + 2*c_pair + 1]
                c_map = [(2*c_pair)*2 + 0, (2*c_pair+1)*2 + 0,
                         (2*c_pair)*2 + 1, (2*c_pair+1)*2 + 1]
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        if U_local[la, l] != 0:
                            U[a2_map[la], r] += U_local[la, l]
                    for lb in range(4):
                        if V_local[lb, l] != 0:
                            V[b2_map[lb], r] += V_local[lb, l]
                    for lc in range(4):
                        if W_local[lc, l] != 0:
                            W[c_map[lc], r] += W_local[lc, l]
            
            if verify(U, V, W, rank):
                return (U, V, W)
        
        # ALS Search
        # We try to minimize || T - sum_r U_r V_r W_r ||
        
        # Pre-compute indices for fast verification
        u_indices = []
        v_indices = []
        w_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_indices.append(i*m + j)
                    v_indices.append(j*p + k)
                    w_indices.append(k*n + i)
        u_indices = np.array(u_indices)
        v_indices = np.array(v_indices)
        w_indices = np.array(w_indices)
        
        best_result = None
        best_err = float('inf')
        
        for attempt in range(20):
            if time.time() - start_time > 145:
                break
            
            np.random.seed(42 + attempt)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(200):
                if time.time() - start_time > 148:
                    break
                
                # Solve for U
                # Target T_u (shape: nm x mp*pn)? No.
                # We have equations: sum_r U_{u,r} V_{v,r} W_{w,r} = 1
                # Fix V, W.
                # For each u, we want to solve for U_u.
                # The equations involving U_u are for all (v,w) such that (u,v,w) is a non-zero entry.
                # Actually, the tensor is sparse.
                # Let's use the matricization approach.
                
                # Reshape V, W to form a matrix M such that U * M = Target_Mat
                # This is tricky with 3-tensors.
                # Alternative: Least squares on the flattened system.
                
                # Let's just do simple coordinate descent / alternating least squares on the residual.
                # To speed up, we can use the fact that we only care about the non-zero entries.
                
                # Construct A_U matrix for least squares
                # Rows correspond to (v,w) pairs? No, rows correspond to the non-zero entries of the tensor.
                # There are N = n*m*p entries.
                # For each entry t (corresponding to u_t, v_t, w_t):
                # We want U_{u_t, r} * (V_{v_t, r} W_{w_t, r}) = 1
                # This is not linear in U if we sum over r.
                # But it IS linear in U_r for a fixed r? No, it's a sum.
                
                # Standard ALS for tensor decomposition:
                # Fix V, W.
                # For each u, we want to minimize sum_{v,w in supp(u)} (1 - sum_r U_{u,r} V_{v,r} W_{w,r})^2
                # This is a quadratic in U_u.
                
                # Let's construct the linear system for U.
                # We can view the problem as:
                # Find U such that vec(U) * (V ⊗ W) = vec(T) ?
                # Yes, if we matricize.
                
                # Let's use a simpler heuristic:
                # Update U by projecting the residual back.
                
                # Calculate current residual R
                # R_t = 1 - sum_r U_{u_t,r} V_{v_t,r} W_{w_t,r}
                
                # Update U:
                # We want to reduce R.
                # Gradient of ||R||^2 w.r.t U_{u,r} is -2 * sum_{t: u_t=u} R_t * V_{v_t,r} * W_{w_t,r}
                
                # This is getting complex to vectorize efficiently without building huge sparse matrices.
                # Given the small size (64 entries), we can just use numpy.
                
                # 1. Compute V_W product matrix for each r?
                # Let M_r = V * W (element-wise? No, V and W have different shapes).
                # V is (mp, rank), W is (pn, rank).
                # We need V_{v,r} * W_{w,r}.
                
                # Let's create a matrix A of size (N, rank) where A[t, r] = V[v_t, r] * W[w_t, r]
                A = V[v_indices, :] * W[w_indices, :]
                
                # We want to solve for U such that U[u_indices, :] * A.T = 1 (element-wise? No)
                # The equation is: sum_r U[u_t, r] * A[t, r] = 1
                # This is a linear system for each unique u?
                # No, U[u, :] is a row vector.
                # Let U_row = U[u, :].
                # We have multiple equations for the same u (for different v,w).
                # We want to minimize sum_{t: u_t=u} (1 - U_row . A[t, :])^2
                
                # This is a least squares problem for each u.
                # A_u is the submatrix of A where u_indices == u.
                # b_u is a vector of 1s.
                # Solve A_u.T X = b_u? No.
                # (A_u A_u.T) U_row.T = A_u b_u.T ?
                # Let's just use lstsq on the transposed system.
                # We want U_row such that A_u U_row.T = b_u.
                
                # Since n*m is small (8), we can loop over u.
                for u in range(n * m):
                    mask = (u_indices == u)
                    if not np.any(mask): continue
                    A_sub = A[mask, :]
                    b_sub = np.ones(np.sum(mask))
                    
                    # Solve A_sub X = b_sub for X (shape rank)
                    # X is U[u, :]
                    try:
                        U[u, :], residuals, rank_sol, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                    except:
                        pass
                
                # Similarly for V
                # Equation: sum_r V[v, r] * (U[u, r] W[w, r]) = 1
                # B = U[u_indices, :] * W[w_indices, :]
                B = U[u_indices, :] * W[w_indices, :]
                for v in range(m * p):
                    mask = (v_indices == v)
                    if not np.any(mask): continue
                    B_sub = B[mask, :]
                    b_sub = np.ones(np.sum(mask))
                    try:
                        V[v, :], residuals, rank_sol, sv = np.linalg.lstsq(B_sub, b_sub, rcond=None)
                    except:
                        pass
                        
                # Similarly for W
                C = U[u_indices, :] * V[v_indices, :]
                for w in range(p * n):
                    mask = (w_indices == w)
                    if not np.any(mask): continue
                    C_sub = C[mask, :]
                    b_sub = np.ones(np.sum(mask))
                    try:
                        W[w, :], residuals, rank_sol, sv = np.linalg.lstsq(C_sub, b_sub, rcond=None)
                    except:
                        pass
                        
                # Check convergence
                if verify(U, V, W, rank):
                    return (U, V, W)
                    
                # Optional: check error to track best
                # err = np.max(np.abs(np.sum(U[u_indices, :] * V[v_indices, :] * W[w_indices, :], axis=1) - 1.0))
                # if err < best_err:
                #    best_err = err
                #    best_result = (U.copy(), V.copy(), W.copy())
                
        if best_result:
            return best_result
        return None

    # Try ranks from 48 to 63
    for rank in range(48, 64):
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return the 56-rank Strassen solution
    U = np.zeros((n * m, 56), dtype=float)
    V = np.zeros((m * p, 56), dtype=float)
    W = np.zeros((p * n, 56), dtype=float)
    
    for c_pair in range(4):
        base_r = c_pair * 7
        b1_map = [0*8 + 2*c_pair, 0*8 + 2*c_pair + 1,
                  1*8 + 2*c_pair, 1*8 + 2*c_pair + 1]
        c_map = [(2*c_pair)*2 + 0, (2*c_pair+1)*2 + 0,
                 (2*c_pair)*2 + 1, (2*c_pair+1)*2 + 1]
        
        for l in range(7):
            r = base_r + l
            for la in range(4):
                if U_local[la, l] != 0:
                    U[a1_map[la], r] += U_local[la, l]
            for lb in range(4):
                if V_local[lb, l] != 0:
                    V[b1_map[lb], r] += V_local[lb, l]
            for lc in range(4):
                if W_local[lc, l] != 0:
                    W[c_map[lc], r] += W_local[lc, l]
    
    for c_pair in range(4):
        base_r = 28 + c_pair * 7
        b2_map = [2*8 + 2*c_pair, 2*8 + 2*c_pair + 1,
                  3*8 + 2*c_pair, 3*8 + 2*c_pair + 1]
        c_map = [(2*c_pair)*2 + 0, (2*c_pair+1)*2 + 0,
                 (2*c_pair)*2 + 1, (2*c_pair+1)*2 + 1]
        
        for l in range(7):
            r = base_r + l
            for la in range(4):
                if U_local[la, l] != 0:
                    U[a2_map[la], r] += U_local[la, l]
            for lb in range(4):
                if V_local[lb, l] != 0:
                    V[b2_map[lb], r] += V_local[lb, l]
            for lc in range(4):
                if W_local[lc, l] != 0:
                    W[c_map[lc], r] += W_local[lc, l]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
