"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-24 decomposition for M_{2,2,8}
    # Uses shared multiplications across the 4 column-pairs
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: 00->0, 01->1, 10->2, 11->3
    # We use 6 multiplications per block with sharing
    
    # Define 24 rank-1 tensors explicitly
    # For each block q (0..3), we have 6 terms, but terms are shared between blocks
    
    # Block 0: terms 0-5
    # Block 1: terms 6-11 (term 6 shares with block 0's term 5)
    # Block 2: terms 12-17 (term 12 shares with block 1's term 11)  
    # Block 3: terms 18-23 (term 18 shares with block 2's term 17)
    
    # Actually, let's use a cleaner approach: 6 unique terms per block = 24 total
    # with no sharing, but a rank-6 decomposition per 2x2x2 block
    
    # Wait - rank 6 is not possible for 2x2x2 (Strassen proves min 7)
    # So we need a different strategy
    
    # Let me use the known rank-24 construction for M_{2,2,8}
    # This uses the fact that we can combine blocks
    
    # Explicit rank-24 tensors:
    # r=0..23
    
    # For each output C[i,k], we need sum_r U[i,j,r]*V[j,k,r]*W[k',i',r] to match
    # Actually the tensor is T[i,j,k] = delta for matmul
    
    # Let me construct it differently. 
    # C = A*B, A is 2x2, B is 2x8
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # 16 outputs, each bilinear in (A,B)
    
    # Rank-24 construction:
    # Use the following 24 rank-1 decompositions
    
    # I'll use a verified rank-24 decomposition
    
    # Term definitions: (U_entries, V_entries, W_entries)
    # U has 4 entries (A[0,0], A[0,1], A[1,0], A[1,1])
    # V has 16 entries (B[j,k] for j in 0,1 and k in 0..7)
    # W has 16 entries (C[i,k] for i in 0,1 and k in 0..7)
    
    # V index: j*8 + k
    # W index: k*2 + i (trace convention: W[k*n+i] where n=2)
    
    terms = [
        # Block 0 (columns 0,1)
        # Term 0: (A00+A11)(B00+B11) -> contributes to C
        ({0:1, 3:1}, {0:1, 11:1}, {0:1, 3:1}),  # C[0,0], C[1,1]... 
        # This is getting complex. Let me use a different known construction.
    ]
    
    # Let me use a simpler verified approach:
    # Actually, I'll keep the 28-rank Strassen but check if we can do 24
    
    # For now, let me try a direct rank-24 construction
    # Based on the fact that mu(2,2,8)=24
    
    # I'll define all 24 terms explicitly
    # U: 4x24, V: 16x24, W: 16x24
    
    # Term r: U[:,r] * V[:,r] * W[:,r]
    
    # Let me use a construction where for each pair of columns (2q, 2q+1),
    # we use a modified Strassen with 6 terms, sharing one with adjacent
    
    # This is complex. Let me just return the 28-rank solution for now
    # and mark that 24 is the target
    
    # Revert to 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    W_coeffs = [
        {0: 1, 3: 1},
        {1: 1, 3: -1},
        {2: 1, 3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {3: 1},
        {0: 1}
    ]

    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            for u_idx, val in U_coeffs[s].items():
                U[u_idx][r] = val
            for v_local_idx, val in V_coeffs[s].items():
                j = v_local_idx // 2
                k_local = v_local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
            for w_local_idx, val in W_coeffs[s].items():
                i = w_local_idx % 2
                k_local = w_local_idx // 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
