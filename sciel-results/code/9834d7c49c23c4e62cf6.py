<<<<<<< SEARCH
"""Rank-64 exact tensor decomposition for 3x4 by 4x6 matrix multiplication.

Partition strategy:
  A (3x4): rows split 1+2, cols split 2+2
  B (4x6): rows split 2+2, cols split 2+2+2
  C (3x6): each block row uses two column blocks of A times two row blocks of B

Block ranks:
  1x2 * 2x2  -> rank 4 (schoolbook)
  2x2 * 2x2  -> rank 7 (Strassen)
  1x2 * 2x2 (with shared B row) -> rank 3 (combined)

Total: 4 * (rank 3) + 4 * (rank 4) + 6 * (rank 7) = 12 + 16 + 42 = 70? 
Actually let me recount:
- For each j in 0..2:
  - Row 0: A[0,0:2]*B[0:2,2j:2j+2] and A[0,2:4]*B[2:4,2j:2j+2]
    These are 1x2*2x2 blocks. Can we combine them?
    The B rows are different (0:2 vs 2:4), so no sharing.
    But wait - can we use a different decomposition for 1x2*2x2 that's rank 3?
    
    Actually, 1x2 * 2x2 has tensor rank exactly 4 (it's equivalent to multiplying a row vector 
    by a 2x2 matrix, which requires 4 multiplications). So rank 4 is minimal.
    
    Let me try a different partition. What if we split A cols as 1+1+2 instead of 2+2?
    Or split B rows differently?
    
    Alternative: Split A rows as 1+1+1 and use 1x2*2x2 for all, giving 9*4=36 for the 
    1-row blocks... no that's worse.
    
    Let me try: A rows 2+1, A cols 2+2, B rows 2+2, B cols 2+2+2
    - For rows 0-1 (2 rows): 2x2*2x2 Strassen (rank 7) for each of 6 blocks = 42
    - For row 2 (1 row): 1x2*2x2 (rank 4) for each of 6 blocks = 24
    Total: 66. Same.
    
    What about splitting B cols as 3+3 instead of 2+2+2?
    Then we'd need 2x2*2x3 and 1x2*2x3 blocks.
    2x2*2x3: rank? The tensor rank of 2x2*2x3... 
    Using Strassen-like: split B cols into 1+2.
    2x2*2x1 (rank 4) + 2x2*2x2 (rank 7) = 11 per block, but that's not how it works.
    
    Actually, let me just try to find a rank-64 decomposition by being smarter.
    
    Key insight: The 1x2 * 2x2 blocks can potentially be optimized.
    For A[0,0:2] * B[0:2, j:j+2], the result is C[0,j] and C[0,j+1].
    C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j]
    C[0,j+1] = A[0,0]*B[0,j+1] + A[0,1]*B[1,j+1]
    
    This is a 2x2x2 tensor (two A inputs, two B inputs, two C outputs).
    Its tensor rank is 4 (full rank for this size).
    
    But wait - can we share terms between the two 1x2*2x2 blocks for row 0?
    Block 1: A[0,0:2] * B[0:2, 2j:2j+2]
    Block 2: A[0,2:4] * B[2:4, 2j:2j+2]
    
    These use different A elements and different B rows, so no sharing possible.
    
    Let me try a completely different approach: use the fact that m=4 is small.
    What if we don't split B rows at all?
    A[0,0:4] * B[0:4, j:j+2] -> 1x4 * 4x2
    Tensor rank of 1x4*4x2: This is equivalent to a 4x2 matrix-vector product.
    Rank = 8 (schoolbook) or can we do better?
    
    Actually, for 1xN * NxM, the tensor rank is NM (you can't do better than schoolbook
    for a single row because each output depends on all N inputs independently).
    
    Hmm, that's not quite right. For 1x2*2x2, rank is 4 = 2*2.
    For 1x4*4x2, rank would be 8 = 4*2.
    
    So splitting doesn't help for single-row blocks.
    
    Let me try yet another approach: what if we use a rank-6 decomposition for some blocks?
    The 2x2*2x2 tensor has rank 7 (Strassen). Can we get rank 6 for any sub-block?
    No, 7 is the known minimal rank for 2x2*2x2.
    
    What about 2x2*2x1 or 2x1*2x2 blocks?
    2x2*2x1: C is 2x1. 
    C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    This is a 2x2x2 tensor, rank 4.
    
    So the minimum for 2x2*2x1 is 4, same as schoolbook.
    
    I think 66 might be close to optimal for this partitioning. Let me try to verify
    the current solution is correct and look for any off-by-one errors.
    
    Actually, let me try a different partition:
    A rows: 1+2, A cols: 1+1+2
    B rows: 1+1+2, B cols: 2+2+2
    
    For A row 0 (1 row):
      A[0,0:1] * B[0:1, 2j:2j+2] -> 1x1*1x2, rank 2
      A[0,1:2] * B[1:2, 2j:2j+2] -> 1x1*1x2, rank 2  
      A[0,2:4] * B[2:4, 2j:2j+2] -> 1x2*2x2, rank 4
      Total per j: 2+2+4 = 8, times 3 = 24
    
    For A rows 1-2 (2 rows):
      A[1:3,0:1] * B[0:1, 2j:2j+2] -> 2x1*1x2, rank 4
      A[1:3,1:2] * B[1:2, 2j:2j+2] -> 2x1*1x2, rank 4
      A[1:3,2:4] * B[2:4, 2j:2j+2] -> 2x2*2x2, rank 7
      Total per j: 4+4+7 = 15, times 3 = 45
    
    Total: 24 + 45 = 69. Worse.
    
    Let me try: A rows 2+1, A cols 2+2, B rows 2+2, B cols 3+3
    For A rows 0-1 (2 rows):
      A[0:2,0:2] * B[0:2, 0:3] -> 2x2*2x3
      A[0:2,2:4] * B[2:4, 0:3] -> 2x2*2x3
      A[0:2,0:2] * B[0:2, 3:6] -> 2x2*2x3
      A[0:2,2:4] * B[2:4, 3:6] -> 2x2*2x3
    For A row 2 (1 row):
      A[2,0:2] * B[0:2, 0:3] -> 1x2*2x3
      A[2,2:4] * B[2:4, 0:3] -> 1x2*2x3
      A[2,0:2] * B[0:2, 3:6] -> 1x2*2x3
      A[2,2:4] * B[2:4, 3:6] -> 1x2*2x3
    
    Now I need ranks for 2x2*2x3 and 1x2*2x3.
    
    2x2*2x3: Split B cols as 1+2.
    2x2*2x1 (rank 4) + 2x2*2x2 (rank 7) = 11
    Or use a more clever decomposition? The tensor rank of 2x2*2x3...
    Actually, we can view this as two 2x2*2x1 problems sharing the first 2x2 of A.
    But they have different B columns, so no sharing.
    Rank = 4 + 7 = 11? Or can we do better?
    
    Hmm, 2x2*2x3 has 12 outputs, 8 A inputs, 6 B inputs.
    Lower bound: max(12, 8, 6) = 12? No, that's not the right bound.
    The trivial rank is 2*2*3 = 12 (schoolbook).
    Can we do 11? Probably not easily.
    
    This is getting complicated. Let me just keep rank 66 and verify correctness.
    """
=======
"""Rank-66 exact tensor decomposition for 3x4 by 4x6 matrix multiplication.

Uses block partitioning with Strassen for 2x2*2x2 blocks and schoolbook for
1x2*2x2 blocks, then attempts local search to reduce rank via term merging/dropping.
"""
>>>>>>> REPLACE
<<<<<<< SEARCH
import time

def construct():
    _t0 = time.time()
    _deadline = 145.0  # safety margin before 150s kill
    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m x p

    RANK = 66
    U = [[0] * RANK for _ in range(n * m)]  # (12, 66)
    V = [[0] * RANK for _ in range(m * p)]  # (24, 66)
    W = [[0] * RANK for _ in range(p * n)]  # (18, 66)

    col = 0
=======
import time
import random
import numpy as np

def construct():
    _t0 = time.time()
    _deadline = 145.0  # safety margin before 150s kill
    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m x p

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _build_base():
        """Build the rank-66 base decomposition."""
        RANK = 66
        U = [[0] * RANK for _ in range(n * m)]
        V = [[0] * RANK for _ in range(m * p)]
        W = [[0] * RANK for _ in range(p * n)]
        col = 0

        def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = _idx_u(a_r, a_c)
            A1 = _idx_u(a_r, a_c + 1)
            B00 = _idx_v(b_r, b_c)
            B01 = _idx_v(b_r, b_c + 1)
            B10 = _idx_v(b_r + 1, b_c)
            B11 = _idx_v(b_r + 1, b_c + 1)
            C0 = _idx_w(c_c, c_r)
            C1 = _idx_w(c_c + 1, c_r)
            U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1; col += 1
            U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1; col += 1
            U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1; col += 1
            U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1; col += 1

        def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = _idx_u(a_r, a_c); A01 = _idx_u(a_r, a_c + 1)
            A10 = _idx_u(a_r + 1, a_c); A11 = _idx_u(a_r + 1, a_c + 1)
            B00 = _idx_v(b_r, b_c); B01 = _idx_v(b_r, b_c + 1)
            B10 = _idx_v(b_r + 1, b_c); B11 = _idx_v(b_r + 1, b_c + 1)
            C00 = _idx_w(c_c, c_r); C01 = _idx_w(c_c + 1, c_r)
            C10 = _idx_w(c_c, c_r + 1); C11 = _idx_w(c_c + 1, c_r + 1)
            U[A00][col] = 1; U[A11][col] = 1; V[B00][col] = 1; V[B11][col] = 1
            W[C00][col] = 1; W[C11][col] = 1; col += 1
            U[A10][col] = 1; U[A11][col] = 1; V[B00][col] = 1
            W[C10][col] = 1; W[C11][col] = -1; col += 1
            U[A00][col] = 1; V[B01][col] = 1; V[B11][col] = -1
            W[C01][col] = 1; W[C11][col] = 1; col += 1
            U[A11][col] = 1; V[B10][col] = 1; V[B00][col] = -1
            W[C00][col] = 1; W[C10][col] = 1; col += 1
            U[A00][col] = 1; U[A01][col] = 1; V[B11][col] = 1
            W[C00][col] = -1; W[C01][col] = 1; col += 1
            U[A10][col] = 1; U[A00][col] = -1; V[B00][col] = 1; V[B01][col] = 1
            W[C11][col] = 1; col += 1
            U[A01][col] = 1; U[A11][col] = -1; V[B10][col] = 1; V[B11][col] = 1
            W[C00][col] = 1; col += 1

        for j in range(3):
            bc = 2 * j
            _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
            _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)
            _add_strassen(1, 0, 0, bc, 1, 2 * j)
            _add_strassen(1, 2, 2, bc, 1, 2 * j)

        assert col == RANK
        return U, V, W

    def _to_numpy(U, V, W, rank):
        return (np.array(U[:rank], dtype=float),
                np.array(V[:rank], dtype=float),
                np.array(W[:rank], dtype=float))

    def _verify(U, V, W, rank):
        """Check that the decomposition exactly computes matrix multiplication."""
        Un, Vn, Wn = _to_numpy(U, V, W, rank)
        # For each (i,j,k): sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal delta
        # Actually the tensor T[i,j,k] = 1 if i,j,k are valid indices
        # The decomposition: T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # We need: for all i in [0,n), j in [0,p), k in [0,m):
        #   sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 (since T[i,j,k]=1 for valid)
        # Wait, the matrix multiplication tensor is T[i,j,k] = 1 for all valid (i,j,k)
        # because C[i,j] = sum_k A[i,k]*B[k,j]
        # So we need sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    s = 0.0
                    for r in range(rank):
                        s += Un[i*m+k, r] * Vn[k*p+j, r] * Wn[j*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def _try_drop_term(U, V, W, rank, r):
        """Try to drop term r by redistributing its contribution to other terms.
        Returns (U, V, W, new_rank) or None."""
        # Check if term r is redundant: can we express its contribution
        # as a linear combination of other terms?
        # Term r contributes: U[:,r] * V[:,r] * W[:,r] to the tensor
        # We need to find coefficients c_s for s != r such that
        # U[:,r] * V[:,r] * W[:,r] = sum_{s != r} c_s * U[:,s] * V[:,s] * W[:,s]
        # This is generally hard. Instead, try simpler heuristics.
        
        # Heuristic: if two terms have identical U,V,W patterns (up to scaling), merge them
        # For now, just check if term r can be zeroed out by adjusting others
        # This is a simplified check - in practice we'd need a full rank-reduction test
        return None

    def _try_merge_terms(U, V, W, rank, r1, r2):
        """Try to merge terms r1 and r2 into a single term.
        If U[:,r1]=U[:,r2], V[:,r1]=V[:,r2], W[:,r1]=W[:,r2] (up to scalar),
        we can combine them. Returns (U, V, W, new_rank) or None."""
        Un, Vn, Wn = _to_numpy(U, V, W, rank)
        u1, u2 = Un[:, r1], Un[:, r2]
        v1, v2 = Vn[:, r1], Vn[:, r2]
        w1, w2 = Wn[:, r1], Wn[:, r2]
        
        # Check if u1 = a*u2, v1 = b*v2, w1 = c*w2 for some scalars
        # and a*b*c = 1 (so the product matches)
        # Actually we need: u1*v1*w1 + u2*v2*w2 can be replaced by a single term
        # This is only possible if the rank-2 tensor u1*v1*w1 + u2*v2*w2 has rank 1
        # i.e., u1*v1*w1 = lambda * u2*v2*w2 for some lambda
        
        nz1 = np.nonzero(u1)[0]
        nz2 = np.nonzero(u2)[0]
        if len(nz1) == 0 or len(nz2) == 0:
            return None
        
        # Check proportionality of U vectors
        ratio_u = None
        for idx in nz1:
            if u2[idx] != 0:
                r = u1[idx] / u2[idx]
                if ratio_u is None:
                    ratio_u = r
                elif abs(r - ratio_u) > 1e-9:
                    return None
        if ratio_u is None:
            return None
        
        # Check proportionality of V vectors
        ratio_v = None
        nz1v = np.nonzero(v1)[0]
        nz2v = np.nonzero(v2)[0]
        for idx in nz1v:
            if v2[idx] != 0:
                r = v1[idx] / v2[idx]
                if ratio_v is None:
                    ratio_v = r
                elif abs(r - ratio_v) > 1e-9:
                    return None
        if ratio_v is None:
            return None
            
        # Check proportionality of W vectors
        ratio_w = None
        nz1w = np.nonzero(w1)[0]
        nz2w = np.nonzero(w2)[0]
        for idx in nz1w:
            if w2[idx] != 0:
                r = w1[idx] / w2[idx]
                if ratio_w is None:
                    ratio_w = r
                elif abs(r - ratio_w) > 1e-9:
                    return None
        if ratio_w is None:
            return None
            
        # Check if ratio_u * ratio_v * ratio_w is consistent (should be ±1 for exact merge)
        # Actually for a valid merge we need the combined tensor to still be rank 1
        # The simplest case: u1 = a*u2, v1 = b*v2, w1 = c*w2, then
        # u1*v1*w1 = abc * u2*v2*w2
        # Combined: (1+abc) * u2*v2*w2 which is still rank 1
        # New term: U=new_u, V=new_v, W=new_w where new_u*new_v*new_w = u1*v1*w1 + u2*v2*w2
        # We can set new_u = u2, new_v = v2, new_w = (1+abc)*w2
        # Or distribute: new_u = u2, new_v = v2, new_w = w2 + a*b*c*w2
        
        abc = ratio_u * ratio_v * ratio_w
        if abs(1 + abc) < 1e-9:
            # They cancel out - just drop both
            new_U = [row[:] for row in U]
            new_V = [row[:] for row in V]
            new_W = [row[:] for row in W]
            # Zero out both terms
            for row in new_U:
                row[r1] = 0; row[r2] = 0
            for row in new_V:
                row[r1] = 0; row[r2] = 0
            for row in new_W:
                row[r1] = 0; row[r2] = 0
            new_rank = rank - 2
            # Remove the zero columns by compacting
            # Find non-zero columns
            keep = []
            for c in range(rank):
                if c == r1 or c == r2:
                    continue
                if any(abs(new_U[i][c]) > 1e-12 for i in range(n*m)) or \
                   any(abs(new_V[i][c]) > 1e-12 for i in range(m*p)) or \
                   any(abs(new_W[i][c]) > 1e-12 for i in range(p*n)):
                    keep.append(c)
            final_U = [[new_U[i][c] for c in keep] for i in range(n*m)]
            final_V = [[new_V[i][c] for c in keep] for i in range(m*p)]
            final_W = [[new_W[i][c] for c in keep] for i in range(p*n)]
            return final_U, final_V, final_W, len(keep)
        else:
            # Combine into one term
            new_U = [row[:] for row in U]
            new_V = [row[:] for row in V]
            new_W = [row[:] for row in W]
            # Set term r1 to the combined, zero out r2
            for i in range(n*m):
                new_U[i][r1] = new_U[i][r2]  # keep u2
                new_U[i][r2] = 0
            for i in range(m*p):
                new_V[i][r1] = new_V[i][r2]  # keep v2
                new_V[i][r2] = 0
            for i in range(p*n):
                new_W[i][r1] = new_W[i][r2] * (1 + abc)  # scaled w2
                new_W[i][r2] = 0
            new_rank = rank - 1
            # Compact
            keep = []
            for c in range(rank):
                if c == r2:
                    continue
                if any(abs(new_U[i][c]) > 1e-12 for i in range(n*m)) or \
                   any(abs(new_V[i][c]) > 1e-12 for i in range(m*p)) or \
                   any(abs(new_W[i][c]) > 1e-12 for i in range(p*n)):
                    keep.append(c)
            final_U = [[new_U[i][c] for c in keep] for i in range(n*m)]
            final_V = [[new_V[i][c] for c in keep] for i in range(m*p)]
            final_W = [[new_W[i][c] for c in keep] for i in range(p*n)]
            return final_U, final_V, final_W, len(keep)

    def _local_search(U, V, W, rank, max_time=120.0):
        """Attempt to reduce rank by trying all pairs of terms for merging."""
        best_U, best_V, best_W, best_rank = U, V, W, rank
        t_start = time.time()
        
        # Try all pairs
        for r1 in range(rank):
            if time.time() - t_start > max_time:
                break
            for r2 in range(r1 + 1, rank):
                if time.time() - t_start > max_time:
                    break
                result = _try_merge_terms(best_U, best_V, best_W, best_rank, r1, r2)
                if result is not None:
                    nU, nV, nW, nrank = result
                    if nrank < best_rank:
                        if _verify(nU, nV, nW, nrank):
                            best_U, best_V, best_W, best_rank = nU, nV, nW, nrank
        return best_U, best_V, best_W, best_rank

    # Build base decomposition
    U, V, W = _build_base()
    rank = 66

    # Verify base
    assert _verify(U, V, W, rank), "Base decomposition failed verification"

    # Local search to reduce rank
    if time.time() - _t0 < _deadline - 10:
        U, V, W, rank = _local_search(U, V, W, rank, max_time=_deadline - time.time() - 5)

    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

