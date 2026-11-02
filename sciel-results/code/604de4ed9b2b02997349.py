import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def strassen_2x2x2_cols(i0, j0, k0, n_dim, m_dim, p_dim):
        """Returns list of 7 (U_dict, V_dict, W_dict) for Strassen 2x2x2 block."""
        cols = []
        cols.append((
            {i0*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        cols.append((
            {(i0+1)*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1},
            {k0*n_dim+(i0+1): 1, (k0+1)*n_dim+(i0+1): -1}
        ))
        cols.append((
            {i0*m_dim+j0: 1},
            {j0*p_dim+(k0+1): 1, (j0+1)*p_dim+(k0+1): -1},
            {(k0+1)*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        cols.append((
            {(i0+1)*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+k0: 1, j0*p_dim+k0: -1},
            {k0*n_dim+i0: 1, k0*n_dim+(i0+1): 1}
        ))
        cols.append((
            {i0*m_dim+j0: 1, i0*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: -1, (k0+1)*n_dim+i0: 1}
        ))
        cols.append((
            {(i0+1)*m_dim+j0: 1, i0*m_dim+j0: -1},
            {j0*p_dim+k0: 1, j0*p_dim+(k0+1): 1},
            {(k0+1)*n_dim+(i0+1): 1}
        ))
        cols.append((
            {i0*m_dim+(j0+1): 1, (i0+1)*m_dim+(j0+1): -1},
            {(j0+1)*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1}
        ))
        return cols

    def direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        for di in range(nc):
            for dj in range(mc):
                for dk in range(pc):
                    u_idx = (ii + di) * m_dim + (jj + dj)
                    v_idx = (jj + dj) * p_dim + (kk + dk)
                    w_idx = (kk + dk) * n_dim + (ii + di)
                    cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))

    def decompose_recursive(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        if time.time() > deadline:
            return
        if nc == 1 and mc == 1 and pc == 1:
            u_idx = ii * m_dim + jj
            v_idx = jj * p_dim + kk
            w_idx = kk * n_dim + ii
            cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
            return
        if nc == 1 or mc == 1 or pc == 1:
            direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            return
        if nc >= 2 and mc >= 2 and pc >= 2:
            cols.extend(strassen_2x2x2_cols(ii, jj, kk, n_dim, m_dim, p_dim))
            if nc > 2:
                decompose_recursive(nc - 2, mc, pc, ii + 2, jj, kk, n_dim, m_dim, p_dim, cols)
            if mc > 2:
                decompose_recursive(2, mc - 2, pc, ii, jj + 2, kk, n_dim, m_dim, p_dim, cols)
            if pc > 2:
                decompose_recursive(2, 2, pc - 2, ii, jj, kk + 2, n_dim, m_dim, p_dim, cols)
            return
        if nc > 1:
            decompose_recursive(1, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc - 1, mc, pc, ii + 1, jj, kk, n_dim, m_dim, p_dim, cols)
        elif mc > 1:
            decompose_recursive(nc, 1, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc, mc - 1, pc, ii, jj + 1, kk, n_dim, m_dim, p_dim, cols)
        elif pc > 1:
            decompose_recursive(nc, mc, 1, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc, mc, pc - 1, ii, jj, kk + 1, n_dim, m_dim, p_dim, cols)

    def cols_to_matrices(cols):
        rank = len(cols)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for r, (u_dict, v_dict, w_dict) in enumerate(cols):
            for idx, val in u_dict.items():
                U[idx, r] = val
            for idx, val in v_dict.items():
                V[idx, r] = val
            for idx, val in w_dict.items():
                W[idx, r] = val
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    T[i, j, k] = s
        return np.allclose(T, 1.0, atol=tol)

    def solve_for_W_float(U, V):
        """Given U and V, solve for W using float least squares."""
        rank = U.shape[1]
        W = np.zeros((p * n, rank), dtype=float)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                b = np.ones(m)
                try:
                    w, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                    if residuals is not None and len(residuals) > 0:
                        if residuals[0] > 1e-6:
                            return None
                    W[k*n+i, :] = w
                except np.linalg.LinAlgError:
                    return None
        return W

    def peel_columns(U, V, W, max_iters=5):
        """
        Try to remove columns from the decomposition by solving for W
        after removing each column. Uses float arithmetic for speed.
        Returns reduced (U, V, W) or None if no reduction found.
        """
        if time.time() > deadline:
            return None
        
        rank = U.shape[1]
        for _iter in range(max_iters):
            if time.time() > deadline:
                break
            removed = False
            for col in range(rank):
                if time.time() > deadline:
                    return None
                
                # Remove column
                mask = [r for r in range(rank) if r != col]
                U_new = U[:, mask]
                V_new = V[:, mask]
                
                # Solve for W
                W_new = solve_for_W_float(U_new, V_new)
                if W_new is None:
                    continue
                
                # Verify
                if verify(U_new, V_new, W_new, tol=1e-8):
                    U, V, W = U_new, V_new, W_new
                    rank = U.shape[1]
                    removed = True
                    break
            
            if not removed:
                break
        
        return (U, V, W) if rank < U.shape[1] else None

    def generate_strassen_decomposition():
        cols = []
        decompose_recursive(n, m, p, 0, 0, 0, n, m, p, cols)
        return cols

    def try_multiple_block_placements():
        """
        Try different placements of Strassen blocks to find lower rank.
        For 3x4x6, we can place 2x2x2 blocks at different positions.
        """
        best_rank = n * m * p
        best_cols = None
        
        # Generate all possible 2x2x2 block placements
        # i0 in 0..1, j0 in 0..2, k0 in 0..4
        placements = []
        for i0 in range(n - 1):
            for j0 in range(m - 1):
                for k0 in range(p - 1):
                    placements.append((i0, j0, k0))
        
        # Try single Strassen block placements
        for (i0, j0, k0) in placements:
            if time.time() > deadline:
                break
            cols = []
            # Apply Strassen to block (i0, j0, k0)
            cols.extend(strassen_2x2x2_cols(i0, j0, k0, n, m, p))
            # Apply direct to remaining
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i - i0 in [0, 1]) and (j - j0 in [0, 1]) and (k - k0 in [0, 1]):
                            continue  # Already handled by Strassen
                        cols.append(({i*m+j: 1.0}, {j*p+k: 1.0}, {k*n+i: 1.0}))
            
            if len(cols) < best_rank:
                U, V, W = cols_to_matrices(cols)
                if verify(U, V, W):
                    best_rank = len(cols)
                    best_cols = cols
        
        # Try double Strassen block placements (non-overlapping)
        for idx1 in range(len(placements)):
            for idx2 in range(idx1 + 1, len(placements)):
                if time.time() > deadline:
                    break
                (i0a, j0a, k0a) = placements[idx1]
                (i0b, j0b, k0b) = placements[idx2]
                
                # Check non-overlapping
                overlap_i = not (i0a + 1 < i0b or i0b + 1 < i0a)
                overlap_j = not (j0a + 1 < j0b or j0b + 1 < j0a)
                overlap_k = not (k0a + 1 < k0b or k0b + 1 < k0a)
                if overlap_i or overlap_j or overlap_k:
                    continue
                
                cols = []
                cols.extend(strassen_2x2x2_cols(i0a, j0a, k0a, n, m, p))
                cols.extend(strassen_2x2x2_cols(i0b, j0b, k0b, n, m, p))
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            in_block_a = (i - i0a in [0, 1]) and (j - j0a in [0, 1]) and (k - k0a in [0, 1])
                            in_block_b = (i - i0b in [0, 1]) and (j - j0b in [0, 1]) and (k - k0b in [0, 1])
                            if in_block_a or in_block_b:
                                continue
                            cols.append(({i*m+j: 1.0}, {j*p+k: 1.0}, {k*n+i: 1.0}))
                
                if len(cols) < best_rank:
                    U, V, W = cols_to_matrices(cols)
                    if verify(U, V, W):
                        best_rank = len(cols)
                        best_cols = cols
        
        return best_cols, best_rank

    def kronecker_decomposition():
        """
        Kronecker-product embedding:
        3x4x6 = (1x4x6 schoolbook) + (2x2x2 Strassen) ⊗ (1x2x3 identity)
        Rank = 12 + 7*6 = 54
        """
        cols = []
        
        # Part 1: Row i=0, schoolbook for j in 0..3, k in 0..5
        # This handles all (i=0, j, k) entries directly
        for j in range(4):
            for k in range(6):
                u_idx = 0 * 4 + j       # i=0, j
                v_idx = j * 6 + k       # j, k
                w_idx = k * 3 + 0       # k, i=0
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
        
        # Part 2: Rows i=1,2 use Strassen 2x2x2 ⊗ identity(1x2x3)
        # The 2x2x2 Strassen block covers i in {1,2}, j in {0,1}, k in {0,1}
        # The identity tensor (1x2x3) covers the "repetition" over j in {0,1,2,3} 
        # and k in {0,1,2,3,4,5}
        # 
        # Actually, let me think about this more carefully.
        # The Kronecker product T ⊗ S where T is 2x2x2 and S is 1x2x3:
        # (T ⊗ S)[(i1,i2), (j1,j2), (k1,k2)] = T[i1,j1,k1] * S[i2,j2,k2]
        # 
        # For our case:
        # - i dimension: 3 = 1 + 2 (row 0 separate, rows 1-2 use Strassen)
        # - j dimension: 4 = 2 * 2 (Strassen uses j in {0,1}, identity uses j in {0,1})
        # - k dimension: 6 = 2 * 3 (Strassen uses k in {0,1}, identity uses k in {0,1,2})
        #
        # So the decomposition is:
        # Part A: i=0, all j, all k → schoolbook (rank 4*6 = 24)
        # Part B: i in {1,2}, j in {0,1}, k in {0,1} → Strassen (rank 7)
        # Part C: i in {1,2}, j in {2,3}, k in {0,1} → Strassen (rank 7)
        # Part D: i in {1,2}, j in {0,1}, k in {2,3} → Strassen (rank 7)
        # Part E: i in {1,2}, j in {2,3}, k in {2,3} → Strassen (rank 7)
        # Part F: i in {1,2}, j in {0,1}, k in {4,5} → Strassen (rank 7)
        # Part G: i in {1,2}, j in {2,3}, k in {4,5} → Strassen (rank 7)
        #
        # Wait, that's 24 + 7*6 = 66, not 54.
        #
        # Let me reconsider. The Kronecker product approach:
        # 3x4x6 can be seen as:
        # - i: 3 = 1 + 2
        # - j: 4 = 2 * 2
        # - k: 6 = 2 * 3
        #
        # The tensor T_{3,4,6} = T_{1,4,6} ⊕ (T_{2,2,2} ⊗ T_{1,2,3})
        # where ⊕ is direct sum and ⊗ is Kronecker product.
        #
        # T_{1,4,6}: rank 4*6 = 24 (schoolbook for i=0)
        # T_{2,2,2} ⊗ T_{1,2,3}: rank 7 * (2*3) = 42
        # Total: 24 + 42 = 66
        #
        # Hmm, that's 66, not 54. Let me re-read the directive.
        #
        # "decompose the 3×4×6 multiplication tensor as the sum of a rank-12 term (1×4×6 schoolbook)
        # and a tensor formed by the Kronecker product of a 2×2×2 Strassen tensor (rank 7) 
        # with a 1×2×3 identity tensor (rank 6), exploiting the algebraic property that the rank 
        # of a Kronecker product is the product of the ranks to achieve a total rank of 12 + 42 = 54"
        #
        # So rank-12 term for 1x4x6? That means the 1x4x6 part has rank 12, not 24.
        # How can 1x4x6 have rank 12? 
        #
        # For a 1xm x p tensor, the rank is at most m*p, but can be less.
        # Actually, for matrix multiplication with n=1, the tensor is just the identity-like
        # structure. Let me think...
        #
        # For n=1, the multiplication is: C[0,k] = sum_j A[0,j] * B[j,k]
        # The tensor T[0,j,k] = 1 if we're computing the product.
        # This is a rank m*p tensor in general (schoolbook).
        #
        # But wait - maybe the "rank-12 term" refers to something else.
        # 1x4x6: if we use Strassen-like tricks on the 4x6 part...
        # 4 = 2+2, 6 = 2+2+2
        # We could decompose 4x6 as (2x2 + 2x2) x (2x2 + 2x2 + 2x2)
        # Using Strassen on each 2x2x2 block: 2*3 = 6 blocks, each rank 7 = 42
        # That's worse.
        #
        # Or maybe: 4x6 = 4x(2*3), and we use the fact that for n=1,
        # we can share computations across the k dimension?
        #
        # Actually, I think the directive might be using a different decomposition.
        # Let me just implement the straightforward approach and see what rank we get.
        #
        # Simple approach:
        # - i=0: schoolbook for all (j,k) → rank 24
        # - i in {1,2}: For each (j,k), use Strassen where possible
        #   j in {0,1,2,3}, k in {0,1,2,3,4,5}
        #   We can place 2x2x2 Strassen blocks:
        #   j blocks: {0,1}, {2,3}
        #   k blocks: {0,1}, {2,3}, {4,5}
        #   So 2 * 3 = 6 Strassen blocks, each rank 7 = 42
        #   Total: 24 + 42 = 66
        #
        # That's 66, which is worse than the current 56.
        #
        # Let me try a different decomposition. What if we do:
        # - Use Strassen on i in {0,1}, j in {0,1}, k in {0,1} → rank 7
        # - Use Strassen on i in {0,1}, j in {2,3}, k in {0,1} → rank 7
        # - Use Strassen on i in {0,1}, j in {0,1}, k in {2,3} → rank 7
        # - Use Strassen on i in {0,1}, j in {2,3}, k in {2,3} → rank 7
        # - Use Strassen on i in {0,1}, j in {0,1}, k in {4,5} → rank 7
        # - Use Strassen on i in {0,1}, j in {2,3}, k in {4,5} → rank 7
        # - i=2: schoolbook for all (j,k) → rank 24
        # Total: 7*6 + 24 = 66
        #
        # Same result. The Kronecker product approach gives 66, not 54.
        #
        # I think the directive's claim of rank 54 may be incorrect, or I'm
        # misunderstanding the decomposition. Let me just implement the best
        # structured approach I can and use peeling to try to reduce rank.
        #
        # Actually, let me try yet another approach. What if we overlap the
        # Strassen blocks more cleverly?
        #
        # For 3x4x6, the current best is rank 56 from the recursive approach.
        # Let me just use that and try to peel.
        
        cols = generate_strassen_decomposition()
        return cols

    # Generate the base Strassen decomposition
    cols = generate_strassen_decomposition()
    U, V, W = cols_to_matrices(cols)
    rank = len(cols)
    
    if not verify(U, V, W):
        cols = []
        direct_decompose(n, m, p, 0, 0, 0, n, m, p, cols)
        U, V, W = cols_to_matrices(cols)
        rank = len(cols)
    
    # Try multiple block placements
    if time.time() < deadline - 30:
        alt_cols, alt_rank = try_multiple_block_placements()
        if alt_cols is not None and alt_rank < rank:
            cols = alt_cols
            rank = alt_rank
            U, V, W = cols_to_matrices(cols)
    
    # Try peeling to reduce rank further
    if time.time() < deadline - 20:
        peeled = peel_columns(U, V, W, max_iters=3)
        if peeled is not None:
            U, V, W = peeled
            rank = U.shape[1]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
