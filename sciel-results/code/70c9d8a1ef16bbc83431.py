import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Vectorized check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster vectorized verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Build target tensor: T[i,j,k] = 1 for all i,j,k
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Reshape approach: for each c, outer product gives contribution
    # Efficient: compute for all (i,j,k) at once using einsum
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # We want S[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    # S[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    S = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(S, 1.0, atol=1e-9)

def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2x2 matrix multiplication tensor.
    U shape (4,7), V shape (4,7), W shape (4,7)
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1; W[3, 3] = 1
    
    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_block_decomp(sa, sb, sp):
    """Build decomposition for (sa x sb) * (sb x sp) -> (sa x sp).
    Returns U (sa*sb, r), V (sb*sp, r), W (sp*sa, r)
    Convention: U[a*sb+b, c], V[b'*sp+k', c], W[k'*sa+a', c]
    Condition: sum_c U[a*sb+b,c]*V[b'*sp+k',c]*W[k'*sa+a',c] = delta_{b,b'} ... 
    Actually the tensor condition is:
    For the sub-problem with dimensions (sa, sb, sp):
    sum_c U[a*sb+b, c] * V[b*sp+k, c] * W[k*sa+a, c] = 1 for all a in [0,sa), b in [0,sb), k in [0,sp)
    """
    if sa == 2 and sb == 2 and sp == 2:
        return _strassen_2x2()
    elif sa == 1 and sb == 1 and sp == 1:
        return np.ones((1,1)), np.ones((1,1)), np.ones((1,1))
    else:
        # Schoolbook
        r = sa * sb * sp
        U = np.zeros((sa * sb, r))
        V = np.zeros((sb * sp, r))
        W = np.zeros((sp * sa, r))
        c = 0
        for a in range(sa):
            for b in range(sb):
                for k in range(sp):
                    U[a * sb + b, c] = 1
                    V[b * sp + k, c] = 1
                    W[k * sa + a, c] = 1
                    c += 1
        return U, V, W

def _build_block_split(n, m, p, splits_n, splits_m, splits_p):
    """Build decomposition using block splitting.
    splits_n: partition of n into block sizes
    splits_m: partition of m into block sizes  
    splits_p: partition of p into block sizes
    """
    # Compute offsets
    def get_offsets(splits):
        offsets = [0]
        for s in splits:
            offsets.append(offsets[-1] + s)
        return offsets
    
    off_n = get_offsets(splits_n)
    off_m = get_offsets(splits_m)
    off_p = get_offsets(splits_p)
    
    # Collect all (block_a, block_b, block_c) triples
    blocks = []
    for bi, sa in enumerate(splits_n):
        for bj, sb in enumerate(splits_m):
            for bk, sp in enumerate(splits_p):
                # Block: A rows [off_n[bi], off_n[bi+1]), A cols = B rows [off_m[bj], off_m[bj+1])
                # B cols [off_p[bk], off_p[bk+1]), C rows [off_n[bi], off_n[bi+1]), C cols [off_p[bk], off_p[bk+1])
                U_b, V_b, W_b = _build_block_decomp(sa, sb, sp)
                blocks.append((bi, bj, bk, sa, sb, sp, U_b, V_b, W_b))
    
    # Total rank
    total_rank = sum(U_b.shape[1] for _, _, _, _, _, _, U_b, _, _ in blocks)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for bi, bj, bk, sa, sb, sp, U_b, V_b, W_b in blocks:
        r_b = U_b.shape[1]
        i0 = off_n[bi]
        j0 = off_m[bj]
        k0 = off_p[bk]
        
        for c in range(r_b):
            # U: rows i0..i0+sa-1, cols j0..j0+sb-1
            for a in range(sa):
                for b in range(sb):
                    U[(i0+a)*m + (j0+b), col] = U_b[a*sb+b, c]
            # V: rows j0..j0+sb-1, cols k0..k0+sp-1
            for b in range(sb):
                for k in range(sp):
                    V[(j0+b)*p + (k0+k), col] = V_b[b*sp+k, c]
            # W: rows k0..k0+sp-1, cols i0..i0+sa-1
            for k in range(sp):
                for a in range(sa):
                    W[(k0+k)*n + (i0+a), col] = W_b[k*sa+a, c]
            col += 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, max_time):
    """Try to reduce rank by removing linearly dependent or zero columns."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Check for zero columns
    nonzero_mask = []
    for c in range(r):
        if np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0):
            nonzero_mask.append(c)
    
    if len(nonzero_mask) < r:
        U = U[:, nonzero_mask]
        V = V[:, nonzero_mask]
        W = W[:, nonzero_mask]
        r = len(nonzero_mask)
    
    # Try to detect if removing any single column still gives a valid decomposition
    # This is expensive, so only try if we have time
    start = time.time()
    reduced_cols = []
    for c in range(r):
        if time.time() - start > max_time * 0.5:
            break
        # Try removing column c
        mask = [x for x in range(r) if x != c]
        if len(mask) < r - 1:
            continue
        U_test = U[:, mask]
        V_test = V[:, mask]
        W_test = W[:, mask]
        if _verify_fast(U_test, V_test, W_test, n, m, p):
            reduced_cols.append(c)
    
    if reduced_cols:
        # Remove all reducible columns (greedy: remove one at a time)
        for c in reduced_cols:
            mask = [x for x in range(U.shape[1]) if x != c]
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Try different block splittings
    # For 3x5x5, we want to partition 3, 5, 5 into blocks
    # Key insight: use Strassen (rank 7) for 2x2x2 blocks instead of schoolbook (rank 8)
    # Saves 1 per 2x2x2 block
    
    # Partition options for dimension 3: [3], [2,1], [1,2], [1,1,1]
    # Partition options for dimension 5: [5], [4,1], [3,2], [2,3], [1,4], [2,2,1], [2,1,2], [1,2,2], [3,1,1], [1,3,1], [1,1,3], [2,1,1,1], etc.
    
    # Best strategy: maximize number of 2x2x2 blocks
    # 3 = 2+1, 5 = 2+2+1
    # So splits: n=[2,1], m=[2,2,1], p=[2,2,1]
    # This gives blocks:
    # (2,2,2): 1 block -> rank 7 (vs 8 schoolbook)
    # (2,2,1): 1 block -> rank 4
    # (2,1,2): 1 block -> rank 4
    # (2,1,1): 1 block -> rank 2
    # (1,2,2): 1 block -> rank 4
    # (1,2,1): 1 block -> rank 2
    # (1,1,2): 1 block -> rank 2
    # (1,1,1): 1 block -> rank 1
    # Total: 7+4+4+2+4+2+2+1 = 26? That seems too low.
    
    # Wait, let me recount. With n=[2,1], m=[2,2,1], p=[2,2,1]:
    # bi=0 (sa=2), bj=0 (sb=2), bk=0 (sp=2): 2x2x2 -> rank 7
    # bi=0 (sa=2), bj=0 (sb=2), bk=1 (sp=2): 2x2x2 -> rank 7
    # bi=0 (sa=2), bj=0 (sb=2), bk=2 (sp=1): 2x2x1 -> rank 4
    # bi=0 (sa=2), bj=1 (sb=2), bk=0 (sp=2): 2x2x2 -> rank 7
    # bi=0 (sa=2), bj=1 (sb=2), bk=1 (sp=2): 2x2x2 -> rank 7
    # bi=0 (sa=2), bj=1 (sb=2), bk=2 (sp=1): 2x2x1 -> rank 4
    # bi=0 (sa=2), bj=2 (sb=1), bk=0 (sp=2): 2x1x2 -> rank 4
    # bi=0 (sa=2), bj=2 (sb=1), bk=1 (sp=2): 2x1x2 -> rank 4
    # bi=0 (sa=2), bj=2 (sb=1), bk=2 (sp=1): 2x1x1 -> rank 2
    # bi=1 (sa=1), bj=0 (sb=2), bk=0 (sp=2): 1x2x2 -> rank 4
    # bi=1 (sa=1), bj=0 (sb=2), bk=1 (sp=2): 1x2x2 -> rank 4
    # bi=1 (sa=1), bj=0 (sb=2), bk=2 (sp=1): 1x2x1 -> rank 2
    # bi=1 (sa=1), bj=1 (sb=2), bk=0 (sp=2): 1x2x2 -> rank 4
    # bi=1 (sa=1), bj=1 (sb=2), bk=1 (sp=2): 1x2x2 -> rank 4
    # bi=1 (sa=1), bj=1 (sb=2), bk=2 (sp=1): 1x2x1 -> rank 2
    # bi=1 (sa=1), bj=2 (sb=1), bk=0 (sp=2): 1x1x2 -> rank 2
    # bi=1 (sa=1), bj=2 (sb=1), bk=1 (sp=2): 1x1x2 -> rank 2
    # bi=1 (sa=1), bj=2 (sb=1), bk=2 (sp=1): 1x1x1 -> rank 1
    
    # Total: 7*4 + 4*4 + 4*3 + 2*3 + 1 = 28 + 16 + 12 + 6 + 1 = 63
    
    # Hmm, that's 63. Let me verify: schoolbook would be 3*5*5 = 75.
    # 75 - 63 = 12 savings from 4 Strassen blocks (4 * 1 = 4) ... that doesn't add up.
    # Wait: each 2x2x2 block saves 1 (8-7=1). We have 4 such blocks, so save 4.
    # But I'm getting 63 vs 75, which is 12 savings. Something is off.
    
    # Let me recount schoolbook for each block type:
    # 2x2x2: 8
    # 2x2x1: 4
    # 2x1x2: 4
    # 2x1x1: 2
    # 1x2x2: 4
    # 1x2x1: 2
    # 1x1x2: 2
    # 1x1x1: 1
    # Schoolbook total: 4*8 + 2*4 + 2*4 + 2 + 4*4 + 2*2 + 2*2 + 1 = 32+8+8+2+16+4+4+1 = 75. Correct.
    # Strassen total: 4*7 + 2*4 + 2*4 + 2 + 4*4 + 2*2 + 2*2 + 1 = 28+8+8+2+16+4+4+1 = 71.
    
    # So the total should be 71, not 63. Let me recount my block list:
    # 4 blocks of 2x2x2 (rank 7 each): bi=0,bj=0,bk=0; bi=0,bj=0,bk=1; bi=0,bj=1,bk=0; bi=0,bj=1,bk=1
    # 2 blocks of 2x2x1 (rank 4 each): bi=0,bj=0,bk=2; bi=0,bj=1,bk=2
    # 2 blocks of 2x1x2 (rank 4 each): bi=0,bj=2,bk=0; bi=0,bj=2,bk=1
    # 1 block of 2x1x1 (rank 2): bi=0,bj=2,bk=2
    # 4 blocks of 1x2x2 (rank 4 each): bi=1,bj=0,bk=0; bi=1,bj=0,bk=1; bi=1,bj=1,bk=0; bi=1,bj=1,bk=1
    # 2 blocks of 1x2x1 (rank 2 each): bi=1,bj=0,bk=2; bi=1,bj=1,bk=2
    # 2 blocks of 1x1x2 (rank 2 each): bi=1,bj=2,bk=0; bi=1,bj=2,bk=1
    # 1 block of 1x1x1 (rank 1): bi=1,bj=2,bk=2
    
    # Total: 4*7 + 2*4 + 2*4 + 1*2 + 4*4 + 2*2 + 2*2 + 1*1 = 28+8+8+2+16+4+4+1 = 71. Correct!
    
    # So the expected rank is 71.
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # schoolbook fallback
    
    # Try the main split
    splits_n = [2, 1]
    splits_m = [2, 2, 1]
    splits_p = [2, 2, 1]
    
    U, V, W = _build_block_split(n, m, p, splits_n, splits_m, splits_p)
    rank = U.shape[1]
    
    if _verify_fast(U, V, W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Try other splits
    other_splits = [
        ([2, 1], [2, 2, 1], [2, 1, 2]),
        ([2, 1], [2, 2, 1], [1, 2, 2]),
        ([1, 2], [2, 2, 1], [2, 2, 1]),
        ([2, 1], [2, 1, 2], [2, 2, 1]),
        ([2, 1], [1, 2, 2], [2, 2, 1]),
    ]
    
    for sn, sm, sp in other_splits:
        if time.time() - start_time > DEADLINE - 20:
            break
        try:
            U2, V2, W2 = _build_block_split(n, m, p, sn, sm, sp)
            if _verify_fast(U2, V2, W2, n, m, p):
                if U2.shape[1] < best_rank:
                    best_U, best_V, best_W = U2, V2, W2
                    best_rank = U2.shape[1]
        except Exception:
            continue
    
    # Fallback to schoolbook if nothing worked
    if best_U is None:
        rank = n * m * p
        best_U = np.zeros((n * m, rank), dtype=float)
        best_V = np.zeros((m * p, rank), dtype=float)
        best_W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j, col] = 1
                    best_V[j * p + k, col] = 1
                    best_W[k * n + i, col] = 1
                    col += 1
        best_rank = rank
    
    # Try rank reduction
    if time.time() - start_time < DEADLINE - 10:
        best_U, best_V, best_W = _try_reduce_rank(
            best_U, best_V, best_W, n, m, p, 
            max_time=min(30, DEADLINE - (time.time() - start_time) - 5)
        )
        best_rank = best_U.shape[1]
    
    # Convert to lists
    def to_list(M):
        result = []
        for i in range(M.shape[0]):
            row = []
            for c in range(M.shape[1]):
                val = M[i, c]
                if val == int(val):
                    row.append(int(val))
                else:
                    row.append(float(val))
            result.append(row)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

# EVOLVE-BLOCK-END
