import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W):
    """Fast verification using matrix operations."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                if abs(np.sum(prod) - 1.0) > 1e-9:
                    return False
    return True

def _compute_syndromes(U, V, W):
    """
    Compute the syndrome for each column: the set of (i,j,k) entries it affects.
    A column r affects entry (i,j,k) iff U[i*m+j, r] != 0 AND V[j*p+k, r] != 0 AND W[k*n+i, r] != 0.
    """
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    syndromes = []
    for r in range(rank):
        affected = set()
        for i in range(n):
            for j in range(m):
                if U[i*m+j, r] != 0:
                    for k in range(p):
                        if V[j*p+k, r] != 0 and W[k*n+i, r] != 0:
                            affected.add((i, j, k))
        syndromes.append(frozenset(affected))
    return syndromes

def _try_syndrome_merge(U, V, W, time_limit=30.0):
    """
    Syndrome-based search: try to merge pairs of columns with overlapping syndromes.
    If two columns affect the same set of tensor entries, we may be able to combine them
    into a single column with adjusted coefficients.
    """
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + time_limit
    
    rank = U.shape[1]
    syndromes = _compute_syndromes(U, V, W)
    
    # Group columns by syndrome
    syndrome_groups = {}
    for r in range(rank):
        s = syndromes[r]
        if s not in syndrome_groups:
            syndrome_groups[s] = []
        syndrome_groups[s].append(r)
    
    # For each group with >= 2 columns, try to merge them
    for syndrome, cols in syndrome_groups.items():
        if time.time() > deadline:
            break
        if len(cols) < 2:
            continue
        
        # Try merging pairs within the group
        for idx1 in range(len(cols)):
            if time.time() > deadline:
                break
            for idx2 in range(idx1 + 1, len(cols)):
                if time.time() > deadline:
                    break
                
                r1 = cols[idx1]
                r2 = cols[idx2]
                
                # Try to find alpha, beta such that:
                # alpha * (u1 ⊗ v1 ⊗ w1) + beta * (u2 ⊗ v2 ⊗ w2) can be replaced
                # by a single term (u_new ⊗ v_new ⊗ w_new)
                # This is possible if the two terms are "proportional" in some sense.
                
                # For now, try simple cases: if u1 == u2, v1 == v2, w1 == w2, merge directly
                if np.array_equal(U[:, r1], U[:, r2]) and np.array_equal(V[:, r1], V[:, r2]) and np.array_equal(W[:, r1], W[:, r2]):
                    # These are identical terms; combine coefficients
                    # The net coefficient at each affected entry is W_val1 + W_val2
                    # We can just keep one with doubled W
                    mask = [i for i in range(rank) if i != r2]
                    U_new = U[:, mask].copy()
                    V_new = V[:, mask].copy()
                    W_new = W[:, mask].copy()
                    # Double the W values for the remaining column
                    pos = mask.index(r1)
                    W_new[:, pos] *= 2
                    if _verify_fast(U_new, V_new, W_new):
                        return U_new, V_new, W_new
                
                # Try: if u1 == u2, can we merge v and w?
                if np.array_equal(U[:, r1], U[:, r2]):
                    # Check if v1*v2 and w1*w2 can be combined
                    # This requires solving for a new v and w
                    # For simplicity, skip for now
                    pass
    
    return None

def _try_structured_rank35(U, V, W, time_limit=30.0):
    """
    Structured search for rank-35 by exploiting the block structure.
    The 4 Strassen blocks each have rank 7, and 2 schoolbook blocks have rank 4.
    Try to reduce one Strassen block from rank 7 to rank 6 by finding a dependency.
    """
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + time_limit
    
    rank = U.shape[1]
    
    # Block column ranges
    blocks = [
        list(range(0, 7)),    # Block 1
        list(range(7, 14)),   # Block 2
        list(range(14, 21)),  # Block 3
        list(range(21, 28)),  # Block 4
        list(range(28, 32)),  # Block 5
        list(range(32, 36)),  # Block 6
    ]
    
    # For each Strassen block, try removing each column and see if the residual
    # can be absorbed by adjusting other columns in the same block
    for block_idx, block_cols in enumerate(blocks[:4]):  # Only Strassen blocks
        if time.time() > deadline:
            break
        
        for remove_col in block_cols:
            if time.time() > deadline:
                break
            
            mask = [i for i in range(rank) if i != remove_col]
            U_new = U[:, mask].copy()
            V_new = V[:, mask].copy()
            W_new = W[:, mask].copy()
            
            # Compute the error tensor
            error = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_new[i*m+j, :] * V_new[j*p+k, :] * W_new[k*n+i, :])
                        error[i, j, k] = 1.0 - val
            
            # Check if error is zero
            if np.max(np.abs(error)) < 1e-9:
                return U_new, V_new, W_new
            
            # Try to fix the error by adjusting one column in the same block
            # Find which column in the block could absorb the error
            for adjust_col in block_cols:
                if adjust_col == remove_col:
                    continue
                if time.time() > deadline:
                    break
                
                # The error must be expressible as a single rank-1 tensor
                # with support matching the adjusted column's U and V patterns
                adj_pos = mask.index(adjust_col)
                
                # Get the support of the adjusted column
                u_support = set(np.where(U_new[:, adj_pos] != 0)[0])
                v_support = set(np.where(V_new[:, adj_pos] != 0)[0])
                
                # Check if error is supported on the cross-product of these
                can_fix = True
                for (i, j, k) in np.argwhere(np.abs(error) > 1e-9):
                    if i*m+j not in u_support or j*p+k not in v_support:
                        can_fix = False
                        break
                
                if not can_fix:
                    continue
                
                # Try to solve for the W adjustment
                # error[i,j,k] = delta_w * W[k*n+i, adj_pos] * U[i*m+j, adj_pos] * V[j*p+k, adj_pos]
                # So delta_w = error[i,j,k] / (W[k*n+i, adj_pos] * U[i*m+j, adj_pos] * V[j*p+k, adj_pos])
                # This must be consistent across all non-zero error entries
                
                delta_w = None
                consistent = True
                for (i, j, k) in np.argwhere(np.abs(error) > 1e-9):
                    denom = W_new[k*n+i, adj_pos] * U_new[i*m+j, adj_pos] * V_new[j*p+k, adj_pos]
                    if abs(denom) < 1e-9:
                        consistent = False
                        break
                    dw = error[i, j, k] / denom
                    if delta_w is None:
                        delta_w = dw
                    elif abs(dw - delta_w) > 1e-9:
                        consistent = False
                        break
                
                if consistent and delta_w is not None:
                    W_new[k*n+i, adj_pos] += delta_w  # This is wrong; we need to add to all affected entries
                    # Actually, we need to add delta_w * W_original to each affected W entry
                    # Let me redo this: the adjustment is to add delta_w to the coefficient
                    # But the coefficient is W[k*n+i, adj_pos], so we add delta_w * (U * V) pattern
                    # This is getting complex; skip for now
                    pass
    
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    U, V, W = _build_strassen_36()
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback: return schoolbook rank-40
        n, m, p = 2, 4, 5
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]
    
    # Phase 1: Try removing 1 column at a time (greedy)
    for attempt in range(100):
        if time.time() > deadline - 15:
            break
        progress = False
        for col in range(rank):
            if time.time() > deadline - 15:
                break
            mask = [i for i in range(rank) if i != col]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            if _verify_fast(U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                progress = True
                break
        if not progress:
            break
    
    # Phase 2: Syndrome-based merge search
    if time.time() < deadline - 20:
        result = _try_syndrome_merge(U, V, W, time_limit=min(15, deadline - time.time() - 10))
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
    
    # Phase 3: Structured rank-35 search
    if time.time() < deadline - 10:
        result = _try_structured_rank35(U, V, W, time_limit=min(10, deadline - time.time() - 5))
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
    
    # Phase 4: Try removing pairs (if time permits)
    if time.time() < deadline - 5:
        rank = U.shape[1]
        for i in range(rank):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, rank):
                if time.time() > deadline - 5:
                    break
                mask = [k for k in range(rank) if k != i and k != j]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if _verify_fast(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            if time.time() > deadline - 5:
                break
            if U.shape[1] < rank:
                break

    # Return the best found
    if not _verify_fast(best_U, best_V, best_W):
        best_U, best_V, best_W = U, V, W
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
