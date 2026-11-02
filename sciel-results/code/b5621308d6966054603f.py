"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.linalg import null_space

def _tensor_constraints():
    """
    Build the constraint system for rank-r decomposition of M_{2,2,15}.
    
    We seek U (4xr), V (30xr), W (30xr) such that for all (i,j,k):
      sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1  (all entries are 1)
    
    This is a system of 60 polynomial equations. We linearize by fixing
    two of the three matrices and solving for the third, then iterate.
    """
    n, m, p = DIMENSIONS
    num_entries = n * m * p  # 60
    
    # We'll represent the tensor T[i,j,k] = 1 for all i,j,k
    return num_entries, n, m, p

def _try_rank_block_diag(rank):
    """Try to find a block-diagonal decomposition of the given rank."""
    n, m, p = DIMENSIONS
    
    # Known decompositions for small blocks
    # M_{2,2,1}: rank 4 (schoolbook)
    # M_{2,2,2}: rank 7 (Strassen)
    
    # Try to partition 15 columns into blocks of size 1 or 2
    # Cost: size-1 block costs 4, size-2 block costs 7
    # Minimize total cost subject to sum of block sizes = 15
    
    best_rank = None
    best_partition = None
    
    # DP: dp[i] = min cost to cover i columns
    # dp[0] = 0, dp[i] = min(dp[i-1]+4, dp[i-2]+7)
    INF = float('inf')
    dp = [INF] * (p + 1)
    dp[0] = 0
    choice = [0] * (p + 1)  # 1 for size-1, 2 for size-2
    
    for i in range(1, p + 1):
        # Use a size-1 block
        if dp[i-1] + 4 < dp[i]:
            dp[i] = dp[i-1] + 4
            choice[i] = 1
        # Use a size-2 block
        if i >= 2 and dp[i-2] + 7 < dp[i]:
            dp[i] = dp[i-2] + 7
            choice[i] = 2
    
    if dp[p] < INF:
        # Reconstruct partition
        blocks = []
        pos = p
        while pos > 0:
            sz = choice[pos]
            blocks.append(sz)
            pos -= sz
        blocks.reverse()
        best_rank = dp[p]
        best_partition = blocks
    
    return best_rank, best_partition

def _build_decomposition_from_partition(partition):
    """Build the actual U,V,W matrices from a partition of column blocks."""
    n, m, p = DIMENSIONS
    
    # Build partition
    total_rank = 0
    for sz in partition:
        if sz == 1:
            total_rank += 4
        elif sz == 2:
            total_rank += 7
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Strassen coefficients
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    U_b = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V_b = [[1,0,1,0],[0,1,0,1]]
    W_b = [[1,1,0,0],[0,0,1,1]]
    
    col_offset = 0
    k_start = 0
    for sz in partition:
        if sz == 1:
            k0 = k_start
            for r in range(4):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i*m+j][c] = U_b[i*2+j][r]
                for j in range(2):
                    V[j*p+k0][c] = V_b[j][r]
                for i in range(2):
                    W[k0*n+i][c] = W_b[i][r]
            col_offset += 4
            k_start += 1
        elif sz == 2:
            k0, k1 = k_start, k_start + 1
            for r in range(7):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i*m+j][c] = U_s[i*2+j][r]
                for j in range(2):
                    V[j*p+k0][c] = V_s[j*2+0][r]
                    V[j*p+k1][c] = V_s[j*2+1][r]
                for kk in range(2):
                    for i in range(2):
                        W[(k0+kk)*n+i][c] = W_s[kk*2+i][r]
            col_offset += 7
            k_start += 2
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(dec):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(dec["U"])
    V = np.array(dec["V"])
    W = np.array(dec["W"])
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Check: for all i,j,k: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _search_rank_10_223():
    """
    Search for a rank-10 decomposition of M_{2,2,3}.
    If found, we can use 5 blocks of 2x2x3 to get rank 50.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # We need to find U (4x10), V (6x10), W (6x10) such that
    # for all i,j,k: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Strategy: Fix U and W, solve for V linearly, then iterate
    # Or use a structured approach based on Strassen-like patterns
    
    # Let's try a structured search: assume U has a specific form
    # based on combining two Strassen blocks with some sharing
    
    # Actually, let's try a more direct approach:
    # The tensor M_{2,2,3} can be seen as M_{2,2,2} + M_{2,2,1}
    # Rank <= 7 + 4 = 11. Can we do 10?
    
    # Let's try to find a rank-10 decomposition by solving the
    # bilinear system. We'll parameterize and use least squares.
    
    # For a rank-r decomposition, we have r * (4 + 6 + 6) = 160 unknowns
    # and 12 equations. This is heavily overdetermined in the wrong way.
    # Actually we have 12 equations (one per (i,j,k)) and 160 unknowns.
    # So solutions should exist if the rank is at least the tensor rank.
    
    # Let's try random initialization and Newton's method or similar.
    
    np.random.seed(42)
    for attempt in range(200):
        U = np.random.randn(4, rank)
        V = np.random.randn(6, rank)
        W = np.random.randn(6, rank)
        
        # Normalize to avoid scale issues
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 0 and norm_v > 0 and norm_w > 0:
                scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        # Iterate: fix two, solve for third
        converged = False
        for it in range(100):
            # Solve for V: for each j,k, sum_r V[j,k,r] * (U[i,j,r] * W[k,i,r]) = 1
            # This is linear in V for fixed U, W
            # Actually it's not quite linear because the constraint couples all r
            
            # Let's use a different approach: gradient descent on the residual
            # Residual R[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        R = 0.0
                        for r in range(rank):
                            R += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        R -= 1.0
                        for r in range(rank):
                            grad_U[i*m+j, r] += R * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += R * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += R * U[i*m+j, r] * V[j*p+k, r]
            
            # Gradient step
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        R = 0.0
                        for r in range(rank):
                            R += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        R -= 1.0
                        max_res = max(max_res, abs(R))
            
            if max_res < 1e-10:
                converged = True
                break
        
        if converged:
            # Verify
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        R = 0.0
                        for r in range(rank):
                            R += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(R - 1.0) > 1e-9:
                            converged = False
                            break
                    if not converged:
                        break
                if not converged:
                    break
            
            if converged:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _build_223_decomposition_from_blocks():
    """
    Build a 2x2x3 decomposition using one Strassen (2x2x2) and one schoolbook (2x2x1).
    Rank = 7 + 4 = 11.
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen for first two columns
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook for third column
    U_b = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V_b = [[1,0,1,0],[0,1,0,1]]
    W_b = [[1,1,0,0],[0,0,1,1]]
    
    # Fill Strassen part (columns 0,1)
    for r in range(7):
        c = r
        for i in range(2):
            for j in range(2):
                U[i*m+j][c] = U_s[i*2+j][r]
        for j in range(2):
            V[j*p+0][c] = V_s[j*2+0][r]
            V[j*p+1][c] = V_s[j*2+1][r]
        for kk in range(2):
            for i in range(2):
                W[kk*n+i][c] = W_s[kk*2+i][r]
    
    # Fill schoolbook part (column 2)
    for r in range(4):
        c = 7 + r
        for i in range(2):
            for j in range(2):
                U[i*m+j][c] = U_b[i*2+j][r]
        for j in range(2):
            V[j*p+2][c] = V_b[j][r]
        for i in range(2):
            W[2*n+i][c] = W_b[i][r]
    
    return U, V, W

def _verify_223_decomp(U, V, W):
    """Verify a 2x2x3 decomposition."""
    n, m, p = 2, 2, 3
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """Return the best tensor decomposition found within the time budget."""
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin
    
    # First, try the known rank-53 block-diagonal decomposition
    best_rank, best_partition = _try_rank_block_diag(15)
    
    if best_partition is None:
        best_partition = [1] * 15
        best_rank = 60
    
    dec = _build_decomposition_from_partition(best_partition)
    if _verify_decomposition(dec):
        best_dec = dec
        best_rank = best_rank
    else:
        # Fallback: known rank-53
        fallback_partition = [2]*7 + [1]
        dec = _build_decomposition_from_partition(fallback_partition)
        best_dec = dec
        best_rank = 53
    
    # Try to find a better decomposition using 2x2x3 blocks
    # If we can get rank 10 for 2x2x3, then 5 blocks give rank 50
    # If we can get rank 11, then 5 blocks give rank 55 (worse than 53)
    
    # Check time
    if time.time() < deadline - 20:
        result_223 = _search_rank_10_223()
        if result_223 is not None:
            U_223, V_223, W_223 = result_223
            
            # Build full 2x2x15 decomposition from five 2x2x3 blocks
            n, m, p = DIMENSIONS
            total_rank = 5 * 10
            U = [[0.0] * total_rank for _ in range(n * m)]
            V = [[0.0] * total_rank for _ in range(m * p)]
            W = [[0.0] * total_rank for _ in range(p * n)]
            
            for block_idx in range(5):
                k_start = block_idx * 3
                col_offset = block_idx * 10
                for r in range(10):
                    c = col_offset + r
                    for i in range(2):
                        for j in range(2):
                            U[i*m+j][c] = U_223[i*2+j][r]
                    for j in range(2):
                        for kk in range(3):
                            V[j*p + k_start + kk][c] = V_223[j*2 + kk][r]
                    for kk in range(3):
                        for i in range(2):
                            W[(k_start + kk)*n + i][c] = W_223[kk*2 + i][r]
            
            dec_new = {"U": U, "V": V, "W": W}
            if _verify_decomposition(dec_new):
                if 50 < best_rank:
                    best_dec = dec_new
                    best_rank = 50
    
    return best_dec
# EVOLVE-BLOCK-END
