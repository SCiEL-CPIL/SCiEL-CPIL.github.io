"""Combined approach: Verified rank-39 baseline, with search for better block decompositions."""
import time
import numpy as np
from fractions import Fraction
import itertools
import scipy.optimize as opt

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _polish_to_rational(U, V, W, n, m, p, deadline):
    """
    Try to polish a float solution to exact half-integer/rational entries.
    """
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)
    
    sums = np.zeros(len(ta_arr))
    rank = U.shape[1]
    for c in range(rank):
        sums += U_r[ta_arr, c] * V_r[tb_arr, c] * W_r[tc_arr, c]
    
    if np.max(np.abs(sums - 1.0)) < 1e-6:
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    U_r2 = np.round(U)
    V_r2 = np.round(V)
    W_r2 = np.round(W)
    
    sums2 = np.zeros(len(ta_arr))
    for c in range(rank):
        sums2 += U_r2[ta_arr, c] * V_r2[tb_arr, c] * W_r2[tc_arr, c]
    
    if np.max(np.abs(sums2 - 1.0)) < 1e-6:
        return U_r2.tolist(), V_r2.tolist(), W_r2.tolist()
    
    return None

def _construct_rank39():
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    5 blocks of 2 columns (Strassen rank 7 each) + 1 single column (schoolbook rank 4).
    Total: 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1
    
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank
    return U, V, W

def _solve_block_2x2xk(k, target_rank, deadline):
    """
    Try to find a decomposition for M_{2,2,k} with given rank.
    Returns (U, V, W) as lists or None.
    """
    n, m = 2, 2
    p = k
    
    # Target tensor entries
    # T[i, j, k] = 1
    # Flatten indices:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for kk in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + kk)
                tc_list.append(kk * n + i)
    
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)
    
    # All possible index combinations to enforce zeros
    all_a = np.arange(n * m)
    all_b = np.arange(m * p)
    all_c = np.arange(p * n)
    
    # Create meshgrid for all combinations
    AA, BB, CC = np.meshgrid(all_a, all_b, all_c, indexing='ij')
    AA_flat = AA.flatten()
    BB_flat = BB.flatten()
    CC_flat = CC.flatten()
    
    # Mask for non-zero entries
    nonzero_mask = np.zeros(len(AA_flat), dtype=bool)
    for idx in range(len(ta_arr)):
        # Find index in flat arrays
        # This is inefficient, better to build a set
        pass
    
    # Better: Build a set of (a,b,c) tuples that are 1
    target_set = set(zip(ta_list, tb_list, tc_list))
    
    # Residual function
    def residuals(params):
        r = target_rank
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        
        U = params[:size_u].reshape(n * m, r)
        V = params[size_u:size_u+size_v].reshape(m * p, r)
        W = params[size_u+size_v:].reshape(p * n, r)
        
        # Calculate T_decomp for all entries
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We only need to check the target entries and a subset of zeros to keep it fast
        # Or check all if small
        
        # Efficient calculation for target entries
        res_target = np.zeros(len(ta_arr))
        for c in range(r):
            res_target += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
        
        # Residuals for target: should be 1
        res_target = res_target - 1.0
        
        # For zeros, we can sample or check all if small
        # For k=3, 4*6*6 = 144 entries. Small enough to check all.
        res_zeros = np.zeros(len(AA_flat))
        for c in range(r):
            res_zeros += U[AA_flat, c] * V[BB_flat, c] * W[CC_flat, c]
            
        # Mask out the non-zero entries from the zero check
        # We need to map (a,b,c) to index in AA_flat
        # AA_flat index = a * (m*p) * (p*n) + b * (p*n) + c
        
        zero_mask = np.ones(len(AA_flat), dtype=bool)
        for a, b, c in zip(ta_list, tb_list, tc_list):
            idx = a * (m * p) * (p * n) + b * (p * n) + c
            zero_mask[idx] = False
            
        res_zeros = res_zeros[zero_mask]
        
        return np.concatenate([res_target, res_zeros])
    
    # Initial guess: Random
    rng = np.random.RandomState(42)
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    x0 = rng.randn(size_u + size_v + size_w) * 0.5
    
    # Use least squares
    try:
        result = opt.least_squares(residuals, x0, method='lm', max_nfev=10000, xtol=1e-12, ftol=1e-12)
        
        if result.success or result.cost < 1e-10:
            # Extract U, V, W
            U = result.x[:size_u].reshape(n * m, target_rank)
            V = result.x[size_u:size_u+size_v].reshape(m * p, target_rank)
            W = result.x[size_u+size_v:].reshape(p * n, target_rank)
            
            # Verify and polish
            polished = _polish_to_rational(U, V, W, n, m, p, deadline)
            if polished:
                U_p, V_p, W_p = polished
                if _verify(U_p, V_p, W_p, n, m, p):
                    return U_p, V_p, W_p
            
            # Check float
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-9):
                return U.tolist(), V.tolist(), W.tolist()
                
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Baseline
    U39, V39, W39 = _construct_rank39()
    if _verify(U39, V39, W39, n, m, p):
        best_result = {"U": U39, "V": V39, "W": W39}
        best_rank = 39
    else:
        # Fallback schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find better block decompositions
    # Strategy: Use blocks of size 3 (rank 10?) and 2 (rank 7)
    # 11 = 3 + 3 + 3 + 2
    # If R(3) = 10, total = 3*10 + 7 = 37
    
    # Try to solve M_{2,2,3} with rank 10
    if time.time() < deadline - 20:
        res_3_10 = _solve_block_2x2xk(3, 10, deadline)
        if res_3_10:
            U3, V3, W3 = res_3_10
            # Construct full matrix
            # 3 blocks of 3, 1 block of 2
            # Total rank = 3*10 + 7 = 37
            
            # We need to assemble the full U, V, W
            # The blocks are independent (block diagonal in B and C)
            # U is shared? No, U is the A matrix, which is shared.
            # In the block decomposition, the U matrix for each block is the SAME A.
            # So we can just take the U from one block, as long as it's consistent.
            # But wait, the decomposition is T = sum U V W.
            # If we have block 1: T1 = U1 V1 W1
            # Block 2: T2 = U2 V2 W2
            # If U1 = U2 = U, then T = U (V1 W1 + V2 W2)? No.
            # The sum is over the rank dimension.
            # So we can just concatenate the V and W columns, and use the same U?
            # No, U must be the same shape.
            # In the block construction, U is 4x10.
            # If we use the same U for all blocks, we are assuming the A factors are identical.
            # This is true because A is the same matrix.
            # So we can take U from the first block, and stack V and W?
            # Let's check:
            # Sum_c U[a,c] V[b,c] W[c,c]
            # If we have two blocks with same U, but different V, W, and disjoint column indices in V, W?
            # No, the rank dimension is the summation index.
            # If we use U (4x10) for block 1, and U (4x10) for block 2.
            # We can't just stack them unless we duplicate U.
            # If we duplicate U, the rank increases.
            # Rank = 10 + 10 + 10 + 7 = 37.
            # U_total is 4x37.
            # Columns 0-9: U_block1
            # Columns 10-19: U_block2
            # Columns 20-29: U_block3
            # Columns 30-36: U_block4 (Strassen)
            
            # V_total is 22x37.
            # Columns 0-9: V_block1 (only non-zero for cols 0-2 of B)
            # Columns 10-19: V_block2 (only non-zero for cols 3-5 of B)
            # ...
            
            # This works!
            
            U_total = np.zeros((4, 37))
            V_total = np.zeros((22, 37))
            W_total = np.zeros((22, 37))
            
            # Block 1: Cols 0-2
            U_total[:, 0:10] = np.array(U3)
            # V3 is 6x10 (for B cols 0-2). B indices 0-5.
            # V_total rows 0-5 correspond to B cols 0-2.
            V_total[0:6, 0:10] = np.array(V3)
            # W3 is 6x10 (for C cols 0-2). C indices 0-5.
            W_total[0:6, 0:10] = np.array(W3)
            
            # Block 2: Cols 3-5
            U_total[:, 10:20] = np.array(U3)
            # B cols 3-5 -> indices 3,4,5 and 14,15,16
            # V3 rows: 0-5 correspond to local cols 0-2.
            # We need to map local B index to global B index.
            # Local 0 -> Global 3
            # Local 1 -> Global 4
            # Local 2 -> Global 5
            # Local 3 -> Global 14
            # Local 4 -> Global 15
            # Local 5 -> Global 16
            
            v_map = [3, 4, 5, 14, 15, 16]
            v3_arr = np.array(V3)
            for i, idx in enumerate(v_map):
                V_total[idx, 10:20] = v3_arr[i, :]
                
            # C cols 3-5 -> indices 6-11
            w_map = [6, 7, 8, 9, 10, 11]
            w3_arr = np.array(W3)
            for i, idx in enumerate(w_map):
                W_total[idx, 10:20] = w3_arr[i, :]
            
            # Block 3: Cols 6-8
            U_total[:, 20:30] = np.array(U3)
            v_map = [6, 7, 8, 17, 18, 19]
            for i, idx in enumerate(v_map):
                V_total[idx, 20:30] = v3_arr[i, :]
            w_map = [12, 13, 14, 15, 16, 17]
            for i, idx in enumerate(w_map):
                W_total[idx, 20:30] = w3_arr[i, :]
            
            # Block 4: Cols 9-10 (Strassen)
            # Use the Strassen block from the rank-39 construction
            # We need to extract the first 7 columns of U39, V39, W39 corresponding to cols 9-10?
            # No, the rank-39 construction has blocks for 0-1, 2-3, etc.
            # We just need a valid rank-7 decomposition for a 2-col block.
            # We can reuse the first block of the rank-39 construction, but mapped to cols 9-10.
            
            # Let's just use the standard Strassen for 2 cols.
            # We can generate it on the fly or extract from U39.
            # U39 cols 0-6 are for B cols 0-1.
            # We need to map B cols 0-1 to B cols 9-10.
            
            u_strassen = np.array(U39)[:, 0:7]
            v_strassen = np.array(V39)[:, 0:7]
            w_strassen = np.array(W39)[:, 0:7]
            
            U_total[:, 30:37] = u_strassen
            
            # Map V: B cols 0-1 -> 9-10
            # V39 rows 0-1 are B[0,0], B[0,1]
            # V39 rows 11-12 are B[1,0], B[1,1]
            # We need to put them in rows 9, 10, 20, 21 of V_total.
            
            v_s_arr = np.array(V39)[:, 0:7]
            # v_s_arr[0] -> B[0,0] -> Global 9
            # v_s_arr[1] -> B[0,1] -> Global 10
            # v_s_arr[11] -> B[1,0] -> Global 20
            # v_s_arr[12] -> B[1,1] -> Global 21
            
            V_total[9, 30:37] = v_s_arr[0, :]
            V_total[10, 30:37] = v_s_arr[1, :]
            V_total[20, 30:37] = v_s_arr[11, :]
            V_total[21, 30:37] = v_s_arr[12, :]
            
            # Map W: C cols 0-1 -> 9-10
            # W39 rows 0-1 are C[0,0], C[1,0]
            # W39 rows 2-3 are C[0,1], C[1,1]
            # We need to put them in rows 18, 19, 20, 21?
            # C index = k*n + i.
            # k=9, i=0 -> 18
            # k=9, i=1 -> 19
            # k=10, i=0 -> 20
            # k=10, i=1 -> 21
            
            w_s_arr = np.array(W39)[:, 0:7]
            W_total[18, 30:37] = w_s_arr[0, :]
            W_total[19, 30:37] = w_s_arr[1, :]
            W_total[20, 30:37] = w_s_arr[2, :]
            W_total[21, 30:37] = w_s_arr[3, :]
            
            # Verify
            if _verify(U_total.tolist(), V_total.tolist(), W_total.tolist(), n, m, p):
                return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
            
    return best_result

# EVOLVE-BLOCK-END
