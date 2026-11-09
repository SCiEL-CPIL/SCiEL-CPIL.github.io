"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    n, m, p = DIMENSIONS
    
    def check_time():
        return time.time() > deadline
    
    def schoolbook():
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
    
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.einsum('r,r,r->', U_np[i*m+j], V_np[j*p+k], W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        if not np.allclose(T, target, atol=1e-9):
            return False
        return True
    
    # Structural change: SHARE Strassen terms across blocks
    # The 7 Strassen terms for the 2x2 part (j=0,1) have U and V components
    # that are independent of the output block. Only W differs per block.
    # We can merge the 5 copies of each Strassen term into 1 term with
    # a W vector that spans all 10 output columns.
    
    # Strassen terms for 2x2x2 (local k in {0,1}):
    strassen_terms = [
        {'U': [(0,0,1),(1,1,1)], 'V': [(0,0,1),(1,1,1)], 'W': [(0,0,1),(1,1,1)]},
        {'U': [(1,0,1),(1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1),(1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1),(1,1,-1)], 'W': [(1,0,1),(1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1),(0,0,-1)], 'W': [(0,0,1),(0,1,1)]},
        {'U': [(0,0,1),(0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1),(1,0,1)]},
        {'U': [(1,0,1),(0,0,-1)], 'V': [(0,0,1),(0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1),(1,1,-1)], 'V': [(1,0,1),(1,1,1)], 'W': [(0,0,1)]}
    ]
    
    # Outer product terms for the 2x1x2 part (j=2):
    # These CANNOT be shared because V depends on the specific k_global.
    # But we can still group them efficiently.
    
    # Build the shared Strassen decomposition:
    # 7 shared Strassen terms + 20 outer product terms (2 per block * 5 blocks * 2 i values)
    # Wait: for each block, we have 4 outer terms (i=0,1 and k_local=0,1)
    # Total outer terms: 5 * 4 = 20
    # Total rank: 7 + 20 = 27
    
    # Let's build this:
    num_blocks = p // 2
    block_size = 2
    total_rank = 7 + num_blocks * 4  # 7 shared + 20 outer = 27
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill in the 7 shared Strassen terms
    for t_idx, term in enumerate(strassen_terms):
        col = t_idx
        for i, j, val in term['U']:
            U[i * m + j][col] += val
        for j, k_local, val in term['V']:
            # V is shared: for each block, the same V entry applies
            # But V[j*p + k_global] needs to be set for each block
            # Actually, V is the same for all blocks, so we set it once
            # but it contributes to all blocks through W
            # Wait, no. V[j*p+k_global] is specific to a global k.
            # The Strassen term for block b uses V entries at k_global = b*2 + k_local
            # So V is NOT shared - it depends on the block!
            
            # This means we CANNOT share the V components across blocks.
            # The sharing only works if U, V, and W are all block-independent.
            # But V depends on which output columns we're computing.
            
            # So the sharing idea doesn't work as simply as I thought.
            # Let me reconsider.
            
            # Actually, in the tensor decomposition, each rank term r has:
            # U_r: a vector in R^{n*m}
            # V_r: a vector in R^{m*p}  
            # W_r: a vector in R^{p*n}
            # And the contribution is U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
            
            # For the Strassen term t, in block b:
            # U_r = same for all b
            # V_r has non-zero entries only at positions (j, b*2+k_local) for the specific k_local
            # W_r has non-zero entries only at positions (b*2+k_local, i) for the specific i
            
            # So V_r and W_r ARE block-dependent. We cannot merge them.
            
            pass
        
        # This approach doesn't work for sharing. Let me try a different structural idea.
    
    # Different structural approach: use a "wide" decomposition
    # Treat the problem as: for each output column k, compute C[:,k] = A @ B[:,k]
    # where A is 2x3 and B[:,k] is 3x1.
    # This is 10 independent 2x3x1 multiplications, each of rank 3.
    # Total rank: 30. But that's worse than 55? No, 30 < 55!
    
    # Wait, that can't be right. Let me reconsider.
    # For a single 2x3x1 multiplication: C[i] = sum_j A[i,j] * B[j]
    # This has 2*3 = 6 on-entries, and rank 3 (one for each j).
    # For 10 output columns: 10 * 3 = 30 rank terms.
    # But each rank term r has:
    # U_r[i*m+j] = A[i,j] (same for all k in this term's group)
    # V_r[j*p+k] = 1 if k is the specific output column, 0 otherwise
    # W_r[k*n+i] = B[j] (depends on j and the specific B value)
    
    # Hmm, but B[j] is a variable, not a constant. In the tensor decomposition,
    # the entries of U, V, W are constants (the structure), not the matrix values.
    
    # Let me re-read the problem. The tensor decomposition is for the MULTIPLICATION TENSOR,
    # which is a fixed tensor T where T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
    # The rank of this tensor is what we're minimizing.
    
    # For the 2x3x1 case: T has 6 on-entries: T[i*3+j, j*1+k, k*2+i] = 1 for i in {0,1}, j in {0,1,2}, k=0.
    # The rank of this tensor is 3 (it's the sum of 3 rank-1 tensors: for each j, e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}).
    # Actually, it's rank 3 because we need 3 terms: one for each j.
    
    # For 10 output columns: the tensor is the "direct sum" of 10 copies of the 2x3x1 tensor.
    # The rank of a direct sum is the sum of the ranks: 10 * 3 = 30.
    
    # But wait, the 2x3x10 tensor is NOT a direct sum in the tensor sense.
    # It's a single tensor with 60 on-entries.
    # The rank could be less than 30 if there's cross-term structure.
    
    # Let me just build the rank-30 "column-by-column" decomposition and verify:
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for j in range(m):
            # Term for output column k, intermediate column j
            for i in range(n):
                U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If rank 30 doesn't work (it should), fall back to rank 55
    # Build rank-55
    num_blocks = p // 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    U55 = [[0] * total_rank for _ in range(n * m)]
    V55 = [[0] * total_rank for _ in range(m * p)]
    W55 = [[0] * total_rank for _ in range(p * n)]
    
    all_terms = strassen_terms + [
        {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
        {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
        {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
        {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
    ]
    
    col_offset = 0
    for b in range(num_blocks):
        for term in all_terms:
            c = col_offset
            for i, j, val in term['U']:
                U55[i * m + j][c] += val
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V55[j * p + k_global][c] += val
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W55[k_global * n + i][c] += val
            col_offset += 1
    
    if verify(U55, V55, W55, n, m, p):
        return {"U": U55, "V": V55, "W": W55}
    
    return schoolbook()
# EVOLVE-BLOCK-END
