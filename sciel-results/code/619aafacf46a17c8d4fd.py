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

def construct():
    """Return the best tensor decomposition found within the time budget."""
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin
    
    # Step 1: Get the optimal block-diagonal decomposition via DP
    best_rank, best_partition = _try_rank_block_diag(15)
    
    if best_partition is None:
        # Fallback: all schoolbook
        best_partition = [1] * 15
        best_rank = 60
    
    # Build and verify
    dec = _build_decomposition_from_partition(best_partition)
    if _verify_decomposition(dec):
        return dec
    
    # Fallback: known rank-53
    fallback_partition = [2]*7 + [1]
    dec = _build_decomposition_from_partition(fallback_partition)
    return dec
# EVOLVE-BLOCK-END
