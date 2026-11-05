import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_for(n, m, p):
    """Generic schoolbook decomposition for n x m * m x p -> n x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """Strassen decomposition for 2x2x2 matrix multiplication. Rank 7."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00 -> C10
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3: A00(B10-B11) -> C01
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4: A11(B01-B10) -> C00, C10
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)B11 -> C00, C01
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00, C11
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _verify(decomp, n, m, p):
    """Verify that the decomposition exactly decomposes the matmul tensor."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    T = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    i_vals = np.arange(n)
    j_vals = np.arange(m)
    k_vals = np.arange(p)
    II, JJ, KK = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
    u_idx = II * m + JJ
    v_idx = JJ * p + KK
    w_idx = KK * n + II
    expected = np.zeros((n*m, m*p, p*n))
    expected[u_idx.ravel(), v_idx.ravel(), w_idx.ravel()] = 1.0
    
    if not np.allclose(T, expected, atol=1e-9):
        return False
    return True

def _block_decomposition(sizes):
    """Construct a block decomposition for 5x5x5 with given block sizes."""
    n, m, p = DIMENSIONS
    num_blocks = len(sizes)
    
    # Precompute offsets
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    
    # Verify total size is 5
    if sum(sizes) != n:
        raise ValueError(f"Block sizes {sizes} don't sum to {n}")
    
    block_decomps = []
    total_rank = 0
    
    for a in range(num_blocks):
        for k in range(num_blocks):
            for b in range(num_blocks):
                sa = sizes[a]
                sk = sizes[k]
                sb = sizes[b]
                
                # Get decomposition for sa x sk * sk x sb -> sa x sb
                if sa == 2 and sk == 2 and sb == 2:
                    decomp = _strassen_2x2()
                else:
                    decomp = _schoolbook_for(sa, sk, sb)
                
                rank_block = len(decomp["U"][0]) if decomp["U"] else 0
                total_rank += rank_block
                
                block_decomps.append({
                    "U": decomp["U"],
                    "V": decomp["V"],
                    "W": decomp["W"],
                    "u_off_a": offsets[a],
                    "u_off_k": offsets[k],
                    "v_off_k": offsets[k],
                    "v_off_b": offsets[b],
                    "w_off_b": offsets[b],
                    "w_off_a": offsets[a],
                    "sa": sa,
                    "sk": sk,
                    "sb": sb,
                    "rank": rank_block
                })
    
    # Allocate final matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_col = 0
    for bd in block_decomps:
        sa, sk, sb = bd["sa"], bd["sk"], bd["sb"]
        U_block = bd["U"]
        V_block = bd["V"]
        W_block = bd["W"]
        rank_block = bd["rank"]
        
        for local_i in range(sa):
            for local_j in range(sk):
                global_u_row = (bd["u_off_a"] + local_i) * m + (bd["u_off_k"] + local_j)
                for r in range(rank_block):
                    U[global_u_row][current_col + r] = U_block[local_i * sk + local_j][r]
        
        for local_j in range(sk):
            for local_k in range(sb):
                global_v_row = (bd["v_off_k"] + local_j) * p + (bd["v_off_b"] + local_k)
                for r in range(rank_block):
                    V[global_v_row][current_col + r] = V_block[local_j * sb + local_k][r]
        
        for local_k in range(sb):
            for local_i in range(sa):
                global_w_row = (bd["w_off_b"] + local_k) * n + (bd["w_off_a"] + local_i)
                for r in range(rank_block):
                    W[global_w_row][current_col + r] = W_block[local_k * sa + local_i][r]
        
        current_col += rank_block
    
    return {"U": U, "V": V, "W": W}, total_rank

def _try_6x6_embedding():
    """
    Try embedding 5x5 matmul into 6x6 matmul and using a 6x6 decomposition.
    The idea: pad A, B, C with zero row/column to make them 6x6,
    use a 6x6 decomposition, then project back to 5x5.
    
    For 6x6, use block decomposition with sizes (2,2,2):
    - 8 blocks of 2x2x2, each using Strassen (rank 7)
    - Total rank for 6x6: 8 * 7 = 56
    
    But we only need the 5x5 sub-tensor. We can zero out the 6th row/col
    in the projection, which doesn't reduce the rank of the decomposition
    but we can potentially use a smarter approach.
    
    Actually, a better approach: use the 6x6 decomposition but only keep
    the terms that contribute to the 5x5 sub-block.
    """
    n, m, p = DIMENSIONS
    
    # Build 6x6 decomposition using (2,2,2) blocks
    sizes_6 = [2, 2, 2]
    decomp_6, rank_6 = _block_decomposition_6x6(sizes_6)
    
    # Now project to 5x5: keep only rows/cols corresponding to indices 0-4
    # U_6 has shape (36, rank_6), we need U_5 with shape (25, rank_5)
    # The 5x5 sub-block of U_6 corresponds to indices where both i,j < 5
    
    U_6 = np.array(decomp_6["U"], dtype=float)
    V_6 = np.array(decomp_6["V"], dtype=float)
    W_6 = np.array(decomp_6["W"], dtype=float)
    
    # Extract 5x5 sub-blocks
    # U: index = i*6+j, keep i<5, j<5
    U_5 = []
    for i in range(5):
        for j in range(5):
            U_5.append(U_6[i*6+j].tolist())
    
    # V: index = j*6+k, keep j<5, k<5
    V_5 = []
    for j in range(5):
        for k in range(5):
            V_5.append(V_6[j*6+k].tolist())
    
    # W: index = k*6+i, keep k<5, i<5
    W_5 = []
    for k in range(5):
        for i in range(5):
            W_5.append(W_6[k*6+i].tolist())
    
    result = {"U": U_5, "V": V_5, "W": W_5}
    
    # Verify
    if _verify(result, 5, 5, 5):
        return result, len(U_5[0])
    
    return None, rank_6

def _block_decomposition_6x6(sizes):
    """Block decomposition for 6x6x6."""
    n = 6
    num_blocks = len(sizes)
    
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    
    if sum(sizes) != n:
        raise ValueError(f"Block sizes {sizes} don't sum to {n}")
    
    block_decomps = []
    total_rank = 0
    
    for a in range(num_blocks):
        for k in range(num_blocks):
            for b in range(num_blocks):
                sa = sizes[a]
                sk = sizes[k]
                sb = sizes[b]
                
                if sa == 2 and sk == 2 and sb == 2:
                    decomp = _strassen_2x2()
                else:
                    decomp = _schoolbook_for(sa, sk, sb)
                
                rank_block = len(decomp["U"][0]) if decomp["U"] else 0
                total_rank += rank_block
                
                block_decomps.append({
                    "U": decomp["U"],
                    "V": decomp["V"],
                    "W": decomp["W"],
                    "u_off_a": offsets[a],
                    "u_off_k": offsets[k],
                    "v_off_k": offsets[k],
                    "v_off_b": offsets[b],
                    "w_off_b": offsets[b],
                    "w_off_a": offsets[a],
                    "sa": sa,
                    "sk": sk,
                    "sb": sb,
                    "rank": rank_block
                })
    
    U = [[0] * total_rank for _ in range(n * n)]
    V = [[0] * total_rank for _ in range(n * n)]
    W = [[0] * total_rank for _ in range(n * n)]
    
    current_col = 0
    for bd in block_decomps:
        sa, sk, sb = bd["sa"], bd["sk"], bd["sb"]
        U_block = bd["U"]
        V_block = bd["V"]
        W_block = bd["W"]
        rank_block = bd["rank"]
        
        for local_i in range(sa):
            for local_j in range(sk):
                global_u_row = (bd["u_off_a"] + local_i) * n + (bd["u_off_k"] + local_j)
                for r in range(rank_block):
                    U[global_u_row][current_col + r] = U_block[local_i * sk + local_j][r]
        
        for local_j in range(sk):
            for local_k in range(sb):
                global_v_row = (bd["v_off_k"] + local_j) * n + (bd["v_off_b"] + local_k)
                for r in range(rank_block):
                    V[global_v_row][current_col + r] = V_block[local_j * sb + local_k][r]
        
        for local_k in range(sb):
            for local_i in range(sa):
                global_w_row = (bd["w_off_b"] + local_k) * n + (bd["w_off_a"] + local_i)
                for r in range(rank_block):
                    W[global_w_row][current_col + r] = W_block[local_k * sa + local_i][r]
        
        current_col += rank_block
    
    return {"U": U, "V": V, "W": W}, total_rank

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Try different block partitions for 5x5x5
    partitions = [
        [2, 2, 1],  # Rank 117
        [3, 2],      # Try this
        [1, 2, 2],   # Same as [2,2,1] but different order
        [2, 1, 2],   # Another permutation
        [1, 1, 3],   # Try this
        [3, 1, 1],   # Try this
    ]
    
    for sizes in partitions:
        if time.time() > deadline:
            break
        
        try:
            result, rank = _block_decomposition(sizes)
            if _verify(result, n, m, p):
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
        except Exception:
            pass
    
    # Try 6x6 embedding approach
    if time.time() < deadline:
        try:
            result, rank = _try_6x6_embedding()
            if result is not None and rank < best_rank:
                best_rank = rank
                best_result = result
        except Exception:
            pass
    
    # Fallback to schoolbook
    if best_result is None:
        fallback = _schoolbook_for(n, m, p)
        return fallback
    
    return best_result
# EVOLVE-BLOCK-END
