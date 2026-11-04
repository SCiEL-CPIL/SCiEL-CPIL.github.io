"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form an exact decomposition of the matrix multiplication tensor."""
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_np[i*m + j, c] * V_np[j*p + k, c] * W_np[k*n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        errors.append((i, j, k, val))
                    if errors and len(errors) > 10:
                        return False
        return len(errors) == 0
    except Exception:
        return False

def _winograd_2x3x3():
    """
    Return U, V, W for rank-5 decomposition of 2x3 @ 3x3 matrix multiplication.
    Uses Winograd's algorithm for 2x3 @ 3x2 (or similar small cases).
    
    For C = A(2x3) * B(3x2):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Winograd approach: factor A and B to reduce multiplications.
    """
    # For 2x3 @ 3x2, schoolbook is 12 mults. Winograd can do better.
    # Let's use a known construction.
    
    # Actually, let me construct this properly.
    # A is 2x3, B is 3x2, C is 2x2.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Winograd: compute m = (A0+A1) * (B0+B1) style reductions
    # For 2x3 @ 3x2, we can group:
    # Let's define intermediate products:
    # p0 = A[0,0] * B[0,0]
    # p1 = A[0,1] * B[1,0]  
    # p2 = A[0,2] * B[2,0]
    # p3 = A[0,0] * B[0,1]
    # p4 = A[0,1] * B[1,1]
    # p5 = A[0,2] * B[2,1]
    # p6 = A[1,0] * B[0,0]
    # p7 = A[1,1] * B[1,0]
    # p8 = A[1,2] * B[2,0]
    # p9 = A[1,0] * B[0,1]
    # p10 = A[1,1] * B[1,1]
    # p11 = A[1,2] * B[2,1]
    # That's still 12.
    
    # Better: use the fact that we can share computations.
    # Define:
    # a0 = A[0,0], a1 = A[0,1], a2 = A[0,2]
    # a3 = A[1,0], a4 = A[1,1], a5 = A[1,2]
    # b0 = B[0,0], b1 = B[1,0], b2 = B[2,0]
    # b3 = B[0,1], b4 = B[1,1], b5 = B[2,1]
    
    # C[0,0] = a0*b0 + a1*b1 + a2*b2
    # C[0,1] = a0*b3 + a1*b4 + a2*b5
    # C[1,0] = a3*b0 + a4*b1 + a5*b2
    # C[1,1] = a3*b3 + a4*b4 + a5*b5
    
    # Winograd trick: compute (a0+a3)*(b0+b3), etc.
    # But we need exact decomposition, so let me just use a known rank-5 scheme.
    
    # Actually, for 2x3 @ 3x2, the minimum rank might be 6 or 7, not 5.
    # Let me use a simpler approach: just return schoolbook for the subproblems
    # and combine them.
    
    # For now, let me construct the tensor product decomposition directly.
    # 2x3 @ 3x7: split the 7 columns into blocks of 3 and 4.
    # C[:, 0:3] = A(2x3) @ B1(3x3)  -- rank r1
    # C[:, 3:7] = A(2x3) @ B2(3x4)  -- rank r2
    # Total rank = r1 + r2 if we can share the A factor.
    
    # Actually, the tensor product approach:
    # If we have a rank-r decomposition for 2x3 @ 3xq, we can use it.
    # For q=3: schoolbook rank = 2*3*3 = 18
    # For q=4: schoolbook rank = 2*3*4 = 24
    # Total = 42 (same as direct schoolbook)
    
    # To improve, we need sub-decompositions with rank less than schoolbook.
    # For 2x3 @ 3x3: can we do better than 18?
    # Lower bound: max(6, 9, 6) = 9. Known results suggest ~12-15.
    # For 2x3 @ 3x4: can we do better than 24?
    # Lower bound: max(6, 12, 8) = 12.
    
    # Let me just return schoolbook decompositions for the subproblems
    # and combine them. This gives rank 42, same as before.
    # To actually improve, I need to find sub-decompositions with lower rank.
    
    # For now, let me implement the structure and use schoolbook sub-decomps.
    # The key insight: if I can find rank r1 < 18 for 2x3@3x3 and rank r2 < 24 for 2x3@3x4,
    # then total rank = r1 + r2 < 42.
    
    # Let me try to find rank-15 for 2x3@3x3 and rank-20 for 2x3@3x4 by search.
    
    return None  # Placeholder, will implement below

def _schoolbook_decomp(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _combine_block_decomps(decomp1, decomp2, n, m, p1, p2):
    """
    Combine two decompositions for 2x3@3xp1 and 2x3@3xp2 into one for 2x3@3x(p1+p2).
    decomp1 = (U1, V1, W1) for 2x3 @ 3xp1
    decomp2 = (U2, V2, W2) for 2x3 @ 3xp2
    
    The combined U is just [U1 | U2] (same rows, more columns)
    The combined V: for each column c1 of V1, we extend to full p1+p2 by zero-padding
                    for each column c2 of V2, we extend to full p1+p2 by placing in second block
    The combined W: similarly, W1 for first p1 rows, W2 for last p2 rows
    """
    U1, V1, W1 = decomp1
    U2, V2, W2 = decomp2
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    total_rank = rank1 + rank2
    p = p1 + p2
    
    U = []
    for i in range(n * m):
        row = U1[i] + U2[i]
        U.append(row)
    
    V = []
    for j in range(m * p):
        row = [0] * total_rank
        if j < m * p1:
            # This row corresponds to block 1
            j1 = j
            for c in range(rank1):
                row[c] = V1[j1][c]
        else:
            # This row corresponds to block 2
            j2 = j - m * p1
            for c in range(rank2):
                row[rank1 + c] = V2[j2][c]
        V.append(row)
    
    W = []
    for k in range(p * n):
        row = [0] * total_rank
        if k < p1 * n:
            # This row corresponds to block 1
            k1 = k
            for c in range(rank1):
                row[c] = W1[k1][c]
        else:
            # This row corresponds to block 2
            k2 = k - p1 * n
            for c in range(rank2):
                row[rank1 + c] = W2[k2][c]
        W.append(row)
    
    return U, V, W

def _search_low_rank_decomp(n, m, p, target_rank, time_budget=10.0, seed=42):
    """
    Search for a decomposition of rank target_rank for n x m @ m x p.
    Returns (U, V, W) or None.
    """
    start_time = time.time()
    if target_rank < max(n*m, m*p, p*n):
        return None
    
    for attempt in range(50):
        if time.time() - start_time > time_budget:
            break
        
        rng = np.random.RandomState(seed + attempt)
        
        # Initialize with sparse structure
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        for c in range(target_rank):
            # Pick random indices with some structure
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = rng.randint(0, p)
            
            # Assign values
            u_val = rng.choice([-1.0, 1.0])
            v_val = rng.choice([-1.0, 1.0])
            w_val = rng.choice([-1.0, 1.0])
            
            U[i * m + j, c] = u_val
            V[j * p + k, c] = v_val
            W[k * n + i, c] = w_val
            
            # Sometimes add secondary terms
            if rng.random() < 0.2:
                i2 = rng.randint(0, n)
                j2 = rng.randint(0, m)
                k2 = rng.randint(0, p)
                U[i2 * m + j2, c] += rng.choice([-1.0, 1.0])
                V[j2 * p + k2, c] += rng.choice([-1.0, 1.0])
                W[k2 * n + i2, c] += rng.choice([-1.0, 1.0])
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Start with schoolbook decomposition (guaranteed valid, rank 42)
    U_sb, V_sb, W_sb = _schoolbook_decomp(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p  # 42
    
    start_time = time.time()
    
    # Strategy: split p=7 into p1=3 and p2=4
    # Find low-rank decompositions for 2x3@3x3 and 2x3@3x4
    # Combine them
    
    p1, p2 = 3, 4
    
    # Try to find rank < 18 for 2x3@3x3
    # Lower bound: max(6, 9, 6) = 9
    # Try ranks from 17 down to 9
    decomp1 = None
    for r1 in range(17, 8, -1):
        if time.time() - start_time > deadline - 30:
            break
        result = _search_low_rank_decomp(n, m, p1, r1, time_budget=5.0, seed=100)
        if result is not None:
            decomp1 = result
            break
    
    if decomp1 is None:
        # Use schoolbook for block 1
        decomp1 = _schoolbook_decomp(n, m, p1)
    
    # Try to find rank < 24 for 2x3@3x4
    # Lower bound: max(6, 12, 8) = 12
    decomp2 = None
    for r2 in range(23, 11, -1):
        if time.time() - start_time > deadline - 30:
            break
        result = _search_low_rank_decomp(n, m, p2, r2, time_budget=5.0, seed=200)
        if result is not None:
            decomp2 = result
            break
    
    if decomp2 is None:
        # Use schoolbook for block 2
        decomp2 = _schoolbook_decomp(n, m, p2)
    
    # Combine the two decompositions
    U_comb, V_comb, W_comb = _combine_block_decomps(decomp1, decomp2, n, m, p1, p2)
    
    # Verify the combined decomposition
    if _verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
        comb_rank = len(U_comb[0]) if U_comb else 0
        if comb_rank < best_rank:
            best_U, best_V, best_W = U_comb, V_comb, W_comb
            best_rank = comb_rank
    
    # Also try direct search for the full 2x3@3x7
    for target_rank in range(best_rank - 1, max(n*m, m*p, p*n) - 1, -1):
        if time.time() - start_time > deadline - 15:
            break
        result = _search_low_rank_decomp(n, m, p, target_rank, time_budget=10.0, seed=300)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
