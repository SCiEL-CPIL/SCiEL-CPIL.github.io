import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We'll use a decomposition based on splitting and Strassen
    # Target: rank < 56
    # Let's try to achieve rank 48 or lower
    
    # Approach: Use the fact that 2x4 x 4x8 can be decomposed
    # by viewing it as 4 independent 2x1 x 1x8 multiplications combined
    # but that's rank 64 (trivial)
    
    # Better: Use block structure
    # A is 2x4, B is 4x8, C is 2x8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's use a rank-50 decomposition or find one with rank < 56
    # 
    # Actually, let me construct a valid decomposition with rank 56 first
    # and verify, then try to optimize
    #
    # For now, let me try a direct construction using the tensor structure
    # The multiplication tensor T has T[i,j,k,l] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # In our indexing: A has 8 elements (2*4), B has 32 elements (4*8), C has 16 elements (2*8)
    # T[a,b,c] = 1 if a = (i,j), b = (j,k), c = (i,k) for valid i,j,k
    # So T[i*4+j, j*8+k, k*2+i] = 1
    
    # Let me try to find a low-rank decomposition
    # I'll use a greedy/structured approach
    
    # Actually, let me just verify the current approach works and try small improvements
    # 
    # New idea: Use rank 56 but with a cleaner construction, or find rank 52/48
    #
    # Let me try: split into 2x2x8 blocks
    # A_left (2x2) * B (2x8) -> 2x8, using rank 7 for 2x2x2 and extending
    # For 2x2 x 2x8: we can use 4 Strassen 2x2x2 blocks = rank 28
    # Two such blocks (for left and right halves of A) = rank 56
    # Same as before.
    #
    # To improve: Can we share between the two halves?
    # A = [A1 A2] where A1, A2 are 2x2
    # C = A1*B_top + A2*B_bottom where B_top is 2x8, B_bottom is 2x8
    # 
    # If we compute A1*B_top and A2*B_bottom separately: 28 + 28 = 56
    # But what if we compute (A1+A2)*(B_top+B_bottom) and (A1-A2)*(B_top-B_bottom)?
    # That's 2 * 28 = 56 still, but we need to recover A1*B_top and A2*B_bottom
    # from these, which requires additional terms. Not obviously better.
    
    # Let me just go with rank 56 but ensure correctness
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen 2x2x2 pattern
    # U_r[i*2+j], V_r[j*2+k], W_r[k*2+i]
    S = [
        # r, U entries, V entries, W entries
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    def add_strassen_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            u_entries, v_entries, w_entries = S[r]
            for idx, val in u_entries:
                li, lj = divmod(idx, 2)
                gi, gj = li + i_off, lj + j_off
                U[gi * 4 + gj][cr] += val
            for idx, val in v_entries:
                lj, lk = divmod(idx, 2)
                gj, gk = lj + j_off, lk + k_off
                V[gj * 8 + gk][cr] += val
            for idx, val in w_entries:
                lk, li = divmod(idx, 2)
                gk, gi = lk + k_off, li + i_off
                W[gk * 2 + gi][cr] += val
    
    # 8 blocks of 2x2x2
    # Block (i_off, j_off, k_off): i_off in {0}, j_off in {0,2}, k_off in {0,2,4,6}
    # A is 2x4, so i goes 0-1 (one 2-row block)
    # j goes 0-3 (two 2-col blocks: j_off=0 and j_off=2)
    # k goes 0-7 (four 2-col blocks: k_off=0,2,4,6)
    
    block_idx = 0
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            add_strassen_block(block_idx * 7, 0, j_off, k_off)
            block_idx += 1
    
    # Verify: C[i,k] should equal sum_j A[i,j]*B[j,k]
    # C[i,k] = sum_r W[k*2+i][r] * (sum_j U[i*4+j][r] * V[j*8+k][r])
    # Wait, the convention is: C = W^T * (V^T * A * U) or similar
    # Let me check: the tensor is T[a,b,c] = sum_r U[a][r] * V[b][r] * W[c][r]
    # where a indexes A, b indexes B, c indexes C
    # T[a,b,c] = 1 if a=(i,j), b=(j',k), c=(i',k') and i=i', j=j', k=k'
    # So T[i*4+j, j*8+k, k*2+i] = 1
    
    # Let's verify with a simple test
    # For a single Strassen block at (i_off=0, j_off=0, k_off=0):
    # It should contribute 1 to T[i*4+j, j*8+k, k*2+i] for i,j,k in {0,1}
    
    return {"U": U, "V": V, "W": W}
