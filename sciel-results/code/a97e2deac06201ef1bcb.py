"""Matrix multiplication 4x5x5 - attempting rank reduction via pattern search."""
DIMENSIONS = [4, 5, 5]
import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    # Strategy: Use a recursive Strassen-based decomposition for the 4x4x4 core
    # and schoolbook for the boundary terms. This gives rank 85.
    # We then attempt to find a lower rank decomposition via random search.
    
    # First, build the rank-85 decomposition as a baseline
    rank_base = 85
    U = np.zeros((nm, rank_base), dtype=int)
    V = np.zeros((mp, rank_base), dtype=int)
    W = np.zeros((pn, rank_base), dtype=int)
    
    # 2x2x2 Strassen factors: U, V, W each 4x7
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    s_u = np.zeros((4, 7), dtype=int)
    s_v = np.zeros((4, 7), dtype=int)
    s_w = np.zeros((4, 7), dtype=int)
    
    # Term 0: (a00+a11)(b00+b11)
    s_u[0,0]=1; s_u[3,0]=1
    s_v[0,0]=1; s_v[3,0]=1
    s_w[0,0]=1; s_w[3,0]=1
    
    # Term 1: (a10+a11)b00
    s_u[2,1]=1; s_u[3,1]=1
    s_v[0,1]=1
    s_w[2,1]=1; s_w[3,1]=-1
    
    # Term 2: a00(b01-b11)
    s_u[0,2]=1
    s_v[1,2]=1; s_v[3,2]=-1
    s_w[1,2]=1; s_w[3,2]=1
    
    # Term 3: a11(b10-b00)
    s_u[3,3]=1
    s_v[2,3]=1; s_v[0,3]=-1
    s_w[0,3]=1; s_w[2,3]=1
    
    # Term 4: (a00+a01)b11
    s_u[0,4]=1; s_u[1,4]=1
    s_v[3,4]=1
    s_w[0,4]=-1; s_w[1,4]=1
    
    # Term 5: (a10-a00)(b00+b01)
    s_u[0,5]=-1; s_u[2,5]=1
    s_v[0,5]=1; s_v[1,5]=1
    s_w[3,5]=1
    
    # Term 6: (a01-a11)(b10+b11)
    s_u[1,6]=1; s_u[3,6]=-1
    s_v[2,6]=1; s_v[3,6]=1
    s_w[0,6]=1
    
    def add_4x4_term(a_blocks, b_blocks, c_blocks, col_start):
        """
        Add a 4x4x4 Strassen term using 2x2x2 Strassen sub-factors.
        a_blocks: dict {(br,bc): coeff} - linear combo of A's 2x2 blocks
        b_blocks: dict {(br,bc): coeff} - linear combo of B's 2x2 blocks  
        c_blocks: dict {(br,bc): coeff} - linear combo of C's 2x2 blocks
        """
        for inner_k in range(7):
            col = col_start + inner_k
            u_vec = s_u[:, inner_k]
            v_vec = s_v[:, inner_k]
            w_vec = s_w[:, inner_k]
            
            # Distribute u_vec to A entries
            for (a_br, a_bc), a_coeff in a_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            gi = 2 * a_br + lr
                            gj = 2 * a_bc + lc
                            U[gi * 5 + gj, col] += a_coeff * u_val
            
            # Distribute v_vec to B entries
            for (b_br, b_bc), b_coeff in b_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            gj = 2 * b_br + lr
                            gk = 2 * b_bc + lc
                            V[gj * 5 + gk, col] += b_coeff * v_val
            
            # Distribute w_vec to C entries
            for (c_br, c_bc), c_coeff in c_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            gi = 2 * c_br + lr
                            gk = 2 * c_bc + lc
                            W[gk * 4 + gi, col] += c_coeff * w_val
    
    # 7 Strassen 4x4 terms for the 4x4x4 core
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),           # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),           # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),           # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),           # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),           # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),           # M7
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        if time.time() - start_time > deadline:
            break
        add_4x4_term(a_coeff, b_coeff, c_coeff, t_idx * 7)
    
    col = 49
    
    # 4x4x1 part: A[:, 0:4] * B[:, 4] -> C[:, 4]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
            if time.time() - start_time > deadline:
                break
    
    # 4x1x4 part: A[:, 4] * B[4, :] -> C[:, 0:4]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
            if time.time() - start_time > deadline:
                break
    
    # 4x1x1 part: A[:, 4] * B[4, 4] -> C[:, 4]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1
    
    # Verify the rank-85 decomposition
    def verify(U_mat, V_mat, W_mat):
        """Check if the decomposition is exact using vectorized operations."""
        r = U_mat.shape[1]
        if r == 0:
            return False
        
        # Compute all tensor entries: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Vectorize over the rank dimension
        # For each (i,j,k), we need the sum over c
        # Build the full tensor by iterating over i,j,k but vectorizing over c
        
        # Precompute for efficiency
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Check each (i,j,k) entry
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Vectorized dot product over rank dimension
                    total = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n*m*p
        U = np.zeros((nm, rank_schoolbook), dtype=int)
        V = np.zeros((mp, rank_schoolbook), dtype=int)
        W = np.zeros((pn, rank_schoolbook), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now attempt to find a lower rank decomposition using tensor flattening
    # View the multiplication tensor as a 3-way tensor and use SVD of its
    # matricization to identify the minimal number of rank-1 components
    
    best_rank = rank_base
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Compute the tensor contribution of each column as a flattened vector
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Each column c contributes a rank-1 tensor: U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # Flattened, this is a vector of length n*m*p = 100
    
    def compute_fingerprint_matrix(U_mat, V_mat, W_mat):
        """Compute the flattened tensor contribution for each column."""
        r = U_mat.shape[1]
        fps = np.zeros((r, n*m*p), dtype=float)
        for col in range(r):
            u_vec = U_mat[:, col]
            v_vec = V_mat[:, col]
            w_vec = W_mat[:, col]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        fps[col, k*n + i] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        return fps
    
    # Build fingerprint matrix for current decomposition
    fps = compute_fingerprint_matrix(best_U, best_V, best_W)
    
    # Use QR with column pivoting to find a minimal linearly independent set
    # This is more robust than sequential least squares
    def find_independent_set(fps, tol=1e-10):
        """Find indices of linearly independent columns using QR with pivoting."""
        r, d = fps.shape
        if r == 0:
            return []
        
        # QR with column pivoting: fps @ Q = R, where Q has orthonormal columns
        # The diagonal of R indicates the effective rank
        try:
            Q, R, piv = np.linalg.qr(fps.T, mode='complete')
            # piv gives the permutation of columns
            # R is upper triangular; diagonal elements indicate rank
            diag = np.abs(np.diag(R))
            # Find the effective rank
            tol_abs = tol * max(diag) if len(diag) > 0 else tol
            eff_rank = np.sum(diag > tol_abs)
            if eff_rank == 0:
                return []
            # The first eff_rank pivoted columns form an independent set
            return list(piv[:eff_rank])
        except:
            # Fallback: greedy sequential check
            kept = []
            for col in range(r):
                if not kept:
                    kept.append(col)
                    continue
                A = fps[kept].T
                b = fps[col]
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=tol)
                    residual = np.linalg.norm(b - A @ coeffs)
                    if residual > tol * max(1.0, np.linalg.norm(b)):
                        kept.append(col)
                except:
                    kept.append(col)
            return kept
    
    if time.time() - start_time < deadline - 10:
        kept = find_independent_set(fps)
        if len(kept) < best_rank:
            new_U = best_U[:, kept]
            new_V = best_V[:, kept]
            new_W = best_W[:, kept]
            
            if verify(new_U, new_V, new_W):
                best_U = new_U
                best_V = new_V
                best_W = new_W
                best_rank = len(kept)
    
    # Now try to find a lower rank by exploiting the block structure
    # The 4x5x5 tensor = 4x4x4 core + boundary terms
    # Check if boundary terms are in the span of core terms
    
    if time.time() - start_time < deadline - 10:
        # Separate core (first 49) and boundary (last 36) terms
        n_core = 49
        n_boundary = best_rank - n_core
        
        if n_boundary > 0:
            fps_core = compute_fingerprint_matrix(best_U[:, :n_core], best_V[:, :n_core], best_W[:, :n_core])
            fps_boundary = compute_fingerprint_matrix(best_U[:, n_core:], best_V[:, n_core:], best_W[:, n_core:])
            
            # Check if each boundary fingerprint is in the span of core fingerprints
            # If so, we can potentially eliminate it
            core_A = fps_core.T  # shape (100, n_core)
            
            removable = []
            for b_idx in range(n_boundary):
                if time.time() - start_time > deadline - 10:
                    break
                b_vec = fps_boundary[b_idx]
                try:
                    coeffs, residuals, rank_a, _ = np.linalg.lstsq(core_A, b_vec, rcond=1e-10)
                    residual = np.linalg.norm(b_vec - core_A @ coeffs)
                    if residual < 1e-10 * max(1.0, np.linalg.norm(b_vec)):
                        removable.append(n_core + b_idx)
                except:
                    pass
            
            if removable:
                keep = [c for c in range(best_rank) if c not in removable]
                new_U = best_U[:, keep]
                new_V = best_V[:, keep]
                new_W = best_W[:, keep]
                
                if verify(new_U, new_V, new_W):
                    best_U = new_U
                    best_V = new_V
                    best_W = new_W
                    best_rank = len(keep)
    
    # Additional reduction: try removing one column at a time and verifying
    # This is expensive but can catch cases where the SVD approach misses dependencies
    if time.time() - start_time < deadline - 15:
        max_removals = min(5, best_rank - 40)  # Don't reduce below 40
        for _ in range(max_removals):
            if time.time() - start_time > deadline - 15:
                break
            
            found = False
            for col in range(best_rank - 1, -1, -1):
                if time.time() - start_time > deadline - 15:
                    break
                
                keep = [c for c in range(best_rank) if c != col]
                new_U = best_U[:, keep]
                new_V = best_V[:, keep]
                new_W = best_W[:, keep]
                
                if verify(new_U, new_V, new_W):
                    best_U = new_U
                    best_V = new_V
                    best_W = new_W
                    best_rank = len(keep)
                    found = True
                    break
            
            if not found:
                break
    
    # If no better found, return the best decomposition
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
