import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110  # 110s limit, leave margin for evaluator overhead

DIMENSIONS = [2, 4, 8]


def _build_rank_48(n, m, p):
    """
    Build a rank-48 decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy: Split the middle dimension into two 2x4 blocks.
    Each 2x4 * 4x8 block uses rank 24 (via Strassen-like decomposition).
    
    For 2x4 * 4x8:
    - Split B into two 2x8 halves: B_top (rows 0-1), B_bot (rows 2-3)
    - A is 2x4, split into A_left (cols 0-1), A_right (cols 2-3)
    - C = A_left @ B_top + A_right @ B_bot
    
    For each 2x2 * 2x8 block, we can use a rank-12 decomposition:
    Split the 2x8 into four 2x2 blocks, each with rank 3 (using a specific decomposition)
    Actually, let's use a more direct approach.
    
    For 2x2 * 2x8:
    A = [[a, b], [c, d]], B = [[e, f, g, h, i, j, k, l], [m, n, o, p, q, r, s, t]]
    C = [[ae+bm, af+bn, ag+bo, ah+bp, ai+bq, aj+br, ak+bs, al+bt],
         [ce+dm, cf+dn, cg+do, ch+dp, ci+dq, cj+dr, ck+ds, cl+dt]]
    
    We can use rank 12 by expressing each of the 8 columns with 3 multiplications:
    For column j (0..7):
    C[0,j] = a*B[0,j] + b*B[1,j]
    C[1,j] = c*B[0,j] + d*B[1,j]
    
    Using 3 multiplications per column group:
    M1 = (a+b) * B[0,j]
    M2 = (c+d) * B[1,j]  
    M3 = (a-b) * (B[0,j]-B[1,j])
    
    Then:
    C[0,j] = (M1 + M3)/2 + (M1 - M3)/2... this doesn't work cleanly.
    
    Let me use a different approach. For 2x2 * 2x2, rank 7 is optimal (Strassen).
    For 2x2 * 2x8, we can view it as 4 independent 2x2 * 2x2 multiplications,
    each with rank 7, giving rank 28 total. But we can do better by sharing.
    
    Actually, the optimal rank for 2x2 * 2x8 is 12, achieved by:
    For each of the 4 pairs of columns, use 3 multiplications.
    
    Let's use a known rank-12 decomposition for 2x2 * 2x4 (two 2x2 blocks):
    This gives rank 12 for 2x2 * 2x4, so rank 24 for 2x2 * 2x8.
    
    For 2x4 * 4x8:
    = (2x2 * 2x8) + (2x2 * 2x8) where the two 2x2s are rows of A and cols 0-1, 2-3
    Each 2x2 * 2x8 has rank 24, so total rank 48.
    """
    total_rank = 48
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 48
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 48
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 48
    
    # A indices: i in 0..1, j in 0..3 -> idx = i*4 + j
    # B indices: j in 0..3, k in 0..7 -> idx = j*8 + k
    # W indices: k*n + i = k*2 + i, k in 0..7, i in 0..1
    
    # Block 1: A_left (cols 0,1) * B_top (rows 0,1)
    # A_left: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,4,5
    # B_top: B[0,0..7], B[1,0..7] -> indices 0..7, 8..15
    # Contributes to C[0,0..7], C[1,0..7]
    
    # Block 2: A_right (cols 2,3) * B_bot (rows 2,3)
    # A_right: A[0,2], A[0,3], A[1,2], A[1,3] -> indices 2,3,6,7
    # B_bot: B[2,0..7], B[3,0..7] -> indices 16..23, 24..31
    # Contributes to C[0,0..7], C[1,0..7]
    
    # For each 2x2 * 2x8 block, use rank 24:
    # Split the 2x8 into four 2x2 blocks (cols 0-1, 2-3, 4-5, 6-7)
    # Each 2x2 * 2x2 uses rank 6 (a specific decomposition)
    
    # Actually, let's use a simpler approach: for 2x2 * 2x2, use rank 7 (Strassen)
    # Four 2x2 blocks -> rank 28 per 2x2*2x8, total 56. That's what we had before.
    
    # Let me try a different split: 2x4 * 4x8 = (2x4 * 2x8) + (2x4 * 2x8)
    # where we split B into top 2 rows and bottom 2 rows.
    # Each 2x4 * 2x8 can be done with rank 24.
    
    # For 2x4 * 2x8:
    # A is 2x4, B is 2x8
    # C[i,k] = sum_j A[i,j] * B[j,k] for j in 0..1
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # This is a 2x2 * 2x8 problem in disguise if we group A columns.
    # Actually, it's just two independent 2x1 * 1x8 multiplications summed.
    # Rank for each term: A[i,0]*B[0,k] is rank 1, A[i,1]*B[1,k] is rank 1.
    # Total rank 2 per (i,k) pair, but we can share.
    
    # For fixed i: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a dot product of [A[i,0], A[i,1]] with [B[0,k], B[1,k]]
    # For all k, we have 8 such dot products.
    # We can use rank 2 for each i, total rank 4 for both i values.
    # But we need to handle all k's.
    
    # Actually, the rank of 2x4 * 2x8 is at most 8 (each of the 4 A entries times each of 2 B rows).
    # But we can do better.
    
    # Let me just use the known rank-56 and try to verify it works, then see if I can find a better one.
    
    # For now, let me implement a clean rank-56 and return it.
    pass


def _build_rank_56(n, m, p):
    """
    Build the rank-56 decomposition using Strassen on 2x2 blocks.
    This is the baseline construction: 8 blocks of 2x2 multiplication, each rank 7.
    """
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            a11_idx = 0 * 4 + 2 * u
            a12_idx = 0 * 4 + 2 * u + 1
            a21_idx = 1 * 4 + 2 * u
            a22_idx = 1 * 4 + 2 * u + 1
            
            b11_idx = 2 * u * 8 + 2 * v
            b12_idx = 2 * u * 8 + 2 * v + 1
            b21_idx = (2 * u + 1) * 8 + 2 * v
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1
            
            c11_idx = 2 * v * 2 + 0
            c12_idx = (2 * v + 1) * 2 + 0
            c21_idx = 2 * v * 2 + 1
            c22_idx = (2 * v + 1) * 2 + 1
            
            # s=0: M1 = (A11+A22)(B11+B22)
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            # s=1: M2 = (A21+A22)B11
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            # s=2: M3 = A11(B12-B22)
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            # s=3: M4 = A22(B21-B11)
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            # s=4: M5 = (A11+A12)B22
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            # s=5: M6 = (A21-A11)(B11+B12)
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            # s=6: M7 = (A12-A22)(B21+B22)
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1
            
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Tensor check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U_arr[i*m+j, s] * V_arr[j*p+k, s] * W_arr[k*n+i, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Random matrix check
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for s in range(rank):
        factor_A = np.zeros(n)
        for i in range(n):
            for j in range(m):
                factor_A[i] += A[i, j] * U_arr[i*m+j, s]
        
        factor_B = np.zeros(p)
        for k in range(p):
            for j in range(m):
                factor_B[k] += B[j, k] * V_arr[j*p+k, s]
        
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += factor_A[i] * factor_B[k] * W_arr[k*n+i, s]
    
    if np.max(np.abs(C_computed - C_expected)) > 1e-9:
        return False
        
    return True


def construct():
    n, m, p = DIMENSIONS
    
    if time.time() > _DEADLINE - 5:
        return {"U": [[0]*1 for _ in range(n*m)], "V": [[0]*1 for _ in range(m*p)], "W": [[0]*1 for _ in range(p*n)]}
    
    U, V, W = _build_rank_56(n, m, p)
    
    return {"U": U, "V": V, "W": W}
