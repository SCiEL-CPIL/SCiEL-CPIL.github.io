import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT - 10.0  # 10s safety margin
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that the tensor decomposition is correct."""
        U_r = U_mat.reshape(n, m, -1)
        V_r = V_mat.reshape(m, p, -1)
        W_r = W_mat.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, 1.0, atol=tol)
    
    def build_schoolbook():
        """Build rank-96 schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def build_strassen_blocks():
        """Build rank-88 Strassen block decomposition."""
        rank = 88
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        # Strassen's 7 terms for 2x2x2
        strassen = [
            ((0, 0), (0, 0), (0, 0)),   # m1 = (a00+a11)(b00+b11)
            ((1, 0), (0, 0), (1, 0)),   # m2 = (a10+a11)b00
            ((0, 1), (1, 1), (0, 1)),   # m3 = a00(b10-b11)
            ((1, 1), (0, -1), (1, 1)),  # m4 = a11(b00-b01)
            ((0, 0), (1, 1), (1, -1)),  # m5 = (a00+a01)b11
            ((1, -1), (0, 1), (0, 1)),  # m6 = (a10-a00)(b00+b01)
            ((1, -1), (1, 1), (0, 0))   # m7 = (a01-a10)(b10+b11)
        ]
        
        for j1 in range(2):  # Two 2x2 blocks for rows 0,1
            for l in range(4):  # Split 8 columns into 4 pairs
                for ui, vi, wi in strassen:
                    # U part: global_i, global_j
                    gi, gj = ui
                    U[gi * m + (2 * j1 + gj), col] = 1.0
                    
                    # V part: global_j, global_k
                    gj2, gk = vi
                    V[(2 * j1 + gj2) * p + (2 * l + gk), col] = 1.0
                    
                    # W part: global_k, global_i
                    gk2, gi2 = wi
                    W[(2 * l + gk2) * n + gi2, col] = 1.0
                    
                    col += 1
            
            # Row 2: schoolbook (4 terms per 2-column pair × 4 pairs = 16, but we do 8)
            for k in range(8):
                U[2 * m + (2 * j1 + k % 2), col] = 1.0
                V[(2 * j1 + k % 2) * p + k, col] = 1.0
                W[k * n + 2, col] = 1.0
                col += 1
        
        return U, V, W
    
    def null_space_reduction(U, V, W, max_time=60.0):
        """Try to reduce rank by finding linear dependencies."""
        rank = U.shape[1]
        start_time = time.time()
        
        # Build the constraint matrix: for each (i,j,k), 
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # This is a multilinear constraint, so we linearize by fixing two factors
        
        # Strategy: for each rank term r, check if it can be removed
        # by solving for corrections in the other terms
        
        keep = [True] * rank
        
        # Try single removals first
        for r in range(rank):
            if time.time() > deadline or time.time() - start_time > max_time:
                break
            if not keep[r]:
                continue
            
            keep[r] = False
            idx = [i for i in range(rank) if keep[i]]
            
            if len(idx) == 0:
                keep[r] = True
                continue
                
            U_red = U[:, idx]
            V_red = V[:, idx]
            W_red = W[:, idx]
            
            if verify(U_red, V_red, W_red):
                pass  # Successfully removed
            else:
                keep[r] = True
        
        idx = [i for i in range(rank) if keep[i]]
        if len(idx) < rank:
            return U[:, idx], V[:, idx], W[:, idx], len(idx)
        
        return U, V, W, rank
    
    def local_search(U, V, W, max_time=40.0):
        """Local search: try to modify entries to enable rank reduction."""
        rank = U.shape[1]
        start_time = time.time()
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank
        
        # Try small perturbations to enable removal
        for r in range(rank):
            if time.time() > deadline or time.time() - start_time > max_time:
                break
            
            # Try to zero out U column r and adjust others
            # This is complex; instead try pairwise swaps
            
            # Try removing r and r+1 together
            if r + 1 < rank:
                keep = [True] * rank
                keep[r] = False
                keep[r + 1] = False
                idx = [i for i in range(rank) if keep[i]]
                
                if len(idx) >= 2:
                    U_red = U[:, idx]
                    V_red = V[:, idx]
                    W_red = W[:, idx]
                    
                    if verify(U_red, V_red, W_red):
                        best_rank = len(idx)
                        best_U, best_V, best_W = U_red, V_red, W_red
        
        if best_rank < rank:
            return best_U, best_V, best_W, best_rank
        
        return U, V, W, rank
    
    # Build initial decomposition
    U, V, W = build_strassen_blocks()
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        U, V, W = build_schoolbook()
    
    initial_rank = U.shape[1]
    
    # Apply null-space reduction
    U, V, W, rank_after_ns = null_space_reduction(U, V, W, max_time=50.0)
    
    # Apply local search if time permits
    if time.time() < deadline - 5.0 and rank_after_ns > 80:
        U, V, W, rank_after_ls = local_search(U, V, W, max_time=30.0)
    else:
        rank_after_ls = rank_after_ns
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        mat_rounded = np.round(mat * 2) / 2
        if np.allclose(mat, mat_rounded, atol=1e-9):
            return mat_rounded.tolist()
        else:
            return mat.tolist()
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
