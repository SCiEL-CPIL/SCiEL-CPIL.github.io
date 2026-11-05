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
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(U_mat.shape[1]):
                        total += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if total != 1:
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
    # View the multiplication tensor as a linear map and use SVD to find
    # minimal rank-1 components
    
    best_rank = rank_base
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Tensor flattening approach:
    # The tensor T[i,j,k] = delta_{i,j,k} for matrix mult (4x5x5)
    # A rank-r decomposition means T = sum_{c=1}^r U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where U[i,j,c], V[j,k,c], W[k,i,c]
    
    # Key insight: if we fix the W factors, the problem becomes:
    # For each (i,j,k), sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta
    
    # Strategy: Use the structure to find linear dependencies
    # Group columns by their W pattern and look for redundancies
    
    # First, let's try a more systematic approach:
    # Check if any column is a linear combination of others in the tensor sense
    
    def can_remove_column(U_mat, V_mat, W_mat, col_idx):
        """Check if column col_idx can be removed (is redundant)."""
        if time.time() - start_time > deadline - 5:
            return False
        
        r = U_mat.shape[1]
        if r <= 1:
            return False
        
        # The column contributes: U[:,col] ⊗ V[:,col] ⊗ W[:,col]
        # We need to check if this is in the span of other columns' contributions
        
        # Simplified check: if the column has all zeros in one factor, it's redundant
        if np.all(U_mat[:, col_idx] == 0):
            return True
        if np.all(V_mat[:, col_idx] == 0):
            return True
        if np.all(W_mat[:, col_idx] == 0):
            return True
        
        # More sophisticated: try to see if we can adjust other columns
        # to compensate. This is NP-hard in general, so use a heuristic.
        
        # Heuristic: if two columns have proportional U, V, or W vectors,
        # one might be eliminable
        for other in range(r):
            if other == col_idx:
                continue
            # Check proportionality of U vectors
            u1 = U_mat[:, col_idx]
            u2 = U_mat[:, other]
            if np.any(u1 != 0) and np.all(u1 == u2):
                v1 = V_mat[:, col_idx]
                v2 = V_mat[:, other]
                if np.any(v1 != 0) and np.all(v1 == v2):
                    w1 = W_mat[:, col_idx]
                    w2 = W_mat[:, other]
                    if np.any(w1 != 0) and np.all(w1 == w2):
                        return True
        
        return False
    
    # Try to remove redundant columns
    removed = []
    for col in range(rank_base - 1, -1, -1):
        if time.time() - start_time > deadline - 10:
            break
        if can_remove_column(best_U, best_V, best_W, col):
            removed.append(col)
    
    # Apply removals
    if removed:
        keep = [c for c in range(rank_base) if c not in removed]
        new_rank = len(keep)
        new_U = best_U[:, keep]
        new_V = best_V[:, keep]
        new_W = best_W[:, keep]
        
        if verify(new_U, new_V, new_W):
            best_U = new_U
            best_V = new_V
            best_W = new_W
            best_rank = new_rank
    
    # Additional strategy: Look for the 4x4x1, 4x1x4, 4x1x1 terms
    # and see if they can be merged with Strassen terms
    # The last 36 columns are schoolbook terms for the 5th row/col
    
    # Try to find if any schoolbook term duplicates a Strassen term
    # by checking tensor equality
    
    if time.time() - start_time < deadline - 15:
        # Check for duplicate tensor contributions
        # Compute the "fingerprint" of each column's tensor contribution
        def column_fingerprint(U_mat, V_mat, W_mat, col):
            """Compute a hash-like fingerprint of the tensor contribution."""
            # The contribution is U[:,col] ⊗ V[:,col] ⊗ W[:,col]
            # Flatten to a vector of length n*m*p = 100
            fp = np.zeros(n*m*p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        fp[k*n + i] += U_mat[i*m+j, col] * V_mat[j*p+k, col] * W_mat[k*n+i, col]
            return fp
        
        # Build fingerprint matrix
        fps = []
        for col in range(best_rank):
            fps.append(column_fingerprint(best_U, best_V, best_W, col))
        fps = np.array(fps)  # shape (rank, 100)
        
        # Find linearly dependent columns using rank analysis
        # If the fingerprint matrix has rank < number of columns,
        # some columns are redundant
        
        # Use SVD to find the effective rank
        try:
            U_svd, s_vals, Vt_svd = np.linalg.svd(fps, full_matrices=False)
            # Effective rank: number of singular values above tolerance
            tol = 1e-10
            eff_rank = np.sum(s_vals > tol)
            
            if eff_rank < best_rank:
                # We can potentially reduce to eff_rank
                # Use the left singular vectors to find a basis
                # This is a heuristic - we keep the first eff_rank columns
                # that form an independent set
                
                # Greedy: keep columns that are linearly independent
                kept = []
                kept_fps = []
                for col in range(best_rank):
                    if time.time() - start_time > deadline - 10:
                        break
                    if not kept:
                        kept.append(col)
                        kept_fps.append(fps[col])
                        continue
                    
                    # Check if fps[col] is in span of kept_fps
                    A = np.array(kept_fps).T  # shape (100, len(kept))
                    b = fps[col]
                    
                    # Least squares
                    try:
                        coeffs, residuals, rank_a, _ = np.linalg.lstsq(A, b, rcond=tol)
                        residual = np.linalg.norm(b - A @ coeffs)
                        if residual > tol * np.linalg.norm(b):
                            kept.append(col)
                            kept_fps.append(fps[col])
                    except:
                        kept.append(col)
                        kept_fps.append(fps[col])
                
                if len(kept) < best_rank:
                    new_U = best_U[:, kept]
                    new_V = best_V[:, kept]
                    new_W = best_W[:, kept]
                    
                    if verify(new_U, new_V, new_W):
                        best_U = new_U
                        best_V = new_V
                        best_W = new_W
                        best_rank = len(kept)
        except:
            pass
    
    # Final optimization: try to find a lower rank via random restarts
    # with small perturbations to see if we can merge terms
    
    if time.time() - start_time < deadline - 5:
        # Try a few random merges
        for attempt in range(5):
            if time.time() - start_time > deadline - 5:
                break
            
            # Pick two random columns and try to merge them
            if best_rank < 2:
                break
            
            col1 = random.randint(0, best_rank - 1)
            col2 = random.randint(0, best_rank - 1)
            if col1 == col2:
                continue
            
            # Try to absorb col2 into col1 by adjusting coefficients
            # This is a local search step
            U_try = best_U.copy()
            V_try = best_V.copy()
            W_try = best_W.copy()
            
            # Simple: just remove col2 and check if still valid
            keep = [c for c in range(best_rank) if c != col2]
            new_U = best_U[:, keep]
            new_V = best_V[:, keep]
            new_W = best_W[:, keep]
            
            if verify(new_U, new_V, new_W):
                best_U = new_U
                best_V = new_V
                best_W = new_W
                best_rank = len(keep)
    
    # If no better found, return the best decomposition
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
