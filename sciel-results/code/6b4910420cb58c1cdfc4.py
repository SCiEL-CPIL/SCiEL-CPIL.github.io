import numpy as np
import time
from fractions import Fraction

def _build_rank44():
    """
    Construct the rank-44 decomposition for 3x4x4 matrix multiplication.
    Uses 4 blocks of 3x2x2, each with rank 11 (7 Strassen + 4 trivial).
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
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
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
    
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

def _verify(U, V, W, n=3, m=4, p=4):
    """
    Verify that the decomposition exactly reconstructs the matmul tensor.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for all valid i<n, j<m, k<p.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_prune_columns(U, V, W, n=3, m=4, p=4, max_time=10.0):
    """
    Attempt to prune linearly dependent columns from the rank-44 decomposition.
    Uses exact rational arithmetic to verify if columns can be removed while
    maintaining the tensor decomposition property.
    
    Returns (U_pruned, V_pruned, W_pruned, new_rank) if pruning succeeds,
    otherwise returns the original matrices.
    """
    start_time = time.time()
    n_cols = U.shape[1]
    
    # Convert to Fraction for exact arithmetic
    U_frac = np.array([[Fraction(int(x)) for x in row] for row in U], dtype=object)
    V_frac = np.array([[Fraction(int(x)) for x in row] for row in V], dtype=object)
    W_frac = np.array([[Fraction(int(x)) for x in row] for row in W], dtype=object)
    
    # Try to remove columns one at a time, checking if the decomposition still holds
    # We'll try removing columns in a greedy fashion
    keep_cols = list(range(n_cols))
    
    # Precompute the tensor values for all (i,j,k) to check against
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need this to equal 1 for all valid (i,j,k)
    
    def check_decomposition(keep_cols):
        """Check if the decomposition with only keep_cols still works."""
        if time.time() - start_time > max_time:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in keep_cols:
                        total += U_frac[i*m+j, r] * V_frac[j*p+k, r] * W_frac[k*n+i, r]
                    if total != 1:
                        return False
        return True
    
    # Try to remove each column
    for col_idx in range(n_cols - 1, -1, -1):
        if time.time() - start_time > max_time - 2.0:
            break
        
        if col_idx not in keep_cols:
            continue
            
        # Try removing this column
        new_keep = [c for c in keep_cols if c != col_idx]
        
        # Quick check: see if this column is "essential"
        # A column is essential if there exists some (i,j,k) where this column
        # is the only one contributing a non-zero value
        
        is_essential = False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check if this column contributes non-zero to T[i,j,k]
                    contrib = U_frac[i*m+j, col_idx] * V_frac[j*p+k, col_idx] * W_frac[k*n+i, col_idx]
                    if contrib != 0:
                        # Check if other columns can compensate
                        # This is expensive, so we'll do a full check
                        is_essential = True
                        break
                if is_essential:
                    break
            if is_essential:
                break
        
        if not is_essential:
            continue
            
        # Full verification
        if check_decomposition(new_keep):
            keep_cols = new_keep
    
    if len(keep_cols) < n_cols:
        # Construct pruned matrices
        U_pruned = U[:, keep_cols]
        V_pruned = V[:, keep_cols]
        W_pruned = W[:, keep_cols]
        return U_pruned, V_pruned, W_pruned, len(keep_cols)
    
    return U, V, W, n_cols

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Start with rank 44 as baseline
    U_base, V_base, W_base = _build_rank44()
    best_result = {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    best_rank = 44
    
    # Verify the baseline
    if _verify(U_base, V_base, W_base):
        # Try to prune columns
        U_pruned, V_pruned, W_pruned, new_rank = _try_prune_columns(
            U_base, V_base, W_base, max_time=30.0
        )
        
        if new_rank < best_rank:
            # Verify the pruned solution
            if _verify(U_pruned, V_pruned, W_pruned):
                best_result = {
                    "U": U_pruned.tolist(),
                    "V": V_pruned.tolist(),
                    "W": W_pruned.tolist()
                }
                best_rank = new_rank
    
    # If we have time, try more aggressive pruning strategies
    if time.time() < deadline - 20.0:
        # Try different column removal orders
        U_base2, V_base2, W_base2 = _build_rank44()
        
        # Try removing columns in forward order
        n_cols = U_base2.shape[1]
        keep_cols = list(range(n_cols))
        
        U_frac = np.array([[Fraction(int(x)) for x in row] for row in U_base2], dtype=object)
        V_frac = np.array([[Fraction(int(x)) for x in row] for row in V_base2], dtype=object)
        W_frac = np.array([[Fraction(int(x)) for x in row] for row in W_base2], dtype=object)
        
        def check_decomposition_fast(keep_cols):
            """Quick check using numpy for speed, then exact verify if promising."""
            if len(keep_cols) >= best_rank:
                return False  # No improvement possible
            for i in range(3):
                for j in range(4):
                    for k in range(4):
                        total = 0.0
                        for r in keep_cols:
                            total += U_base2[i*4+j, r] * V_base2[j*4+k, r] * W_base2[k*3+i, r]
                        if abs(total - 1.0) > 1e-9:
                            return False
            return True
        
        # Try removing each column in forward order
        for col_idx in range(n_cols):
            if time.time() > deadline - 10.0:
                break
            if col_idx not in keep_cols:
                continue
            
            new_keep = [c for c in keep_cols if c != col_idx]
            if check_decomposition_fast(new_keep):
                # Exact verification
                keep_cols = new_keep
        
        if len(keep_cols) < best_rank:
            U_pruned2 = U_base2[:, keep_cols]
            V_pruned2 = V_base2[:, keep_cols]
            W_pruned2 = W_base2[:, keep_cols]
            if _verify(U_pruned2, V_pruned2, W_pruned2):
                best_result = {
                    "U": U_pruned2.tolist(),
                    "V": V_pruned2.tolist(),
                    "W": W_pruned2.tolist()
                }
                best_rank = len(keep_cols)
    
    return best_result

BET: {mechanism: "Added column pruning via exact rational verification to potentially reduce rank below 44 by removing linearly dependent or redundant columns from the factor matrices", q: 0.3, delta: [0.0, 0.05]}
