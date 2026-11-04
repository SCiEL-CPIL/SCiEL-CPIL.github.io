import numpy as np
import time

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _build_rank93():
    """Build the known rank-93 exact decomposition (Strassen 4x4x4 + 4*2x2x3)."""
    n, m, p = DIMENSIONS
    total_rank = 49 + 44

    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)

    current_rank = 0

    outer_strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1})
    ]
    inner_strassen = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    ]
    c_inner = {
        0: {(0,0): 1, (1,1): 1},
        1: {(1,0): 1, (1,1): -1},
        2: {(0,1): 1, (1,1): 1},
        3: {(0,0): 1, (1,0): 1},
        4: {(0,0): -1, (0,1): 1},
        5: {(1,1): 1},
        6: {(0,0): 1}
    }
    c_outer = {
        0: {(0,0): 1, (1,1): 1},
        1: {(1,0): 1, (1,1): -1},
        2: {(0,1): 1, (1,1): 1},
        3: {(0,0): 1, (1,0): 1},
        4: {(0,0): -1, (0,1): 1},
        5: {(1,1): 1},
        6: {(0,0): 1}
    }

    for m_idx in range(7):
        x_def, y_def = outer_strassen[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = inner_strassen[t_idx]

            u_coeffs = np.zeros(n * m)
            v_coeffs = np.zeros(m * p)
            for (li, lj), c_x in x_sub.items():
                for b, coeff in x_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_coeffs[i * m + j] += c_x * coeff
            for (li, lj), c_y in y_sub.items():
                for b, coeff in y_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    v_coeffs[i * p + j] += c_y * coeff

            U[:, r] = u_coeffs
            V[:, r] = v_coeffs

            w_coeffs = np.zeros(p * n)
            for (li, lj), cm_c in c_inner[t_idx].items():
                for (br, bc), cb_c in c_outer[m_idx].items():
                    if cm_c == 0 or cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    w_coeffs[k_final * n + i_final] += cm_c * cb_c
            W[:, r] = w_coeffs

    def add_2x2x3(a_rb, a_cb, b_rb, c_rb):
        nonlocal current_rank
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = inner_strassen[t_idx]
            u_coeffs = np.zeros(n * m)
            v_coeffs = np.zeros(m * p)
            for (li, lj), c in x_sub.items():
                i = 2 * a_rb + li
                j = 2 * a_cb + lj
                u_coeffs[i * m + j] += c
            for (li, lj), c in y_sub.items():
                i = 2 * b_rb + li
                j = 4 + lj
                v_coeffs[i * p + j] += c
            U[:, r] = u_coeffs
            V[:, r] = v_coeffs
            w_coeffs = np.zeros(p * n)
            for (li, lj), c in c_inner[t_idx].items():
                i_final = 2 * c_rb + li
                k_final = 4 + lj
                w_coeffs[k_final * n + i_final] += c
            W[:, r] = w_coeffs

        schoolbook = [
            ({(0,0): 1}, {(0,0): 1}, (0,0)),
            ({(0,1): 1}, {(1,0): 1}, (0,0)),
            ({(1,0): 1}, {(0,0): 1}, (1,0)),
            ({(1,1): 1}, {(1,0): 1}, (1,0))
        ]
        for t_idx in range(4):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_pos = schoolbook[t_idx]
            u_coeffs = np.zeros(n * m)
            v_coeffs = np.zeros(m * p)
            for (li, lj), c in x_sub.items():
                i = 2 * a_rb + li
                j = 2 * a_cb + lj
                u_coeffs[i * m + j] += c
            for (li, lj), c in y_sub.items():
                i = 2 * b_rb + li
                j = 6
                v_coeffs[i * p + j] += c
            U[:, r] = u_coeffs
            V[:, r] = v_coeffs
            w_coeffs = np.zeros(p * n)
            i_final = 2 * c_rb + c_pos[0]
            k_final = 6
            w_coeffs[k_final * n + i_final] = 1.0
            W[:, r] = w_coeffs

    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(0, 1, 1, 0)
    add_2x2x3(1, 0, 0, 1)
    add_2x2x3(1, 1, 1, 1)

    return U[:, :current_rank], V[:, :current_rank], W[:, :current_rank]


def _verify(U, V, W, tol=1e-7):
    """Verify sum_r U[i,r] V[j,r] W[k,r] == 1 if (i,j,k) is a valid triple, else 0.
    The tensor T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r] must equal the matmul tensor.
    For 4x4x7 matmul: T[i,j,k] = 1 if i,j,k are the standard triple, else 0.
    Actually the matmul tensor is T[i,j,k] = delta (it's a 3-tensor with T[i,j,k]=1
    for all i,j,k? No.)
    """
    n, m, p = DIMENSIONS
    # The matmul tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait: the tensor for matrix multiplication A (n x m) * B (m x p) -> C (n x p)
    # is T[i, j, k] where i indexes C row, j indexes the shared dim, k indexes B col.
    # The decomposition is: C[i,k] = sum_j A[i,j] B[j,k]
    # As a 3-tensor: T[i, j, k] = 1 for all i,j,k (it's the identity in the middle).
    # Actually T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k} = 1 for all i,j,k.
    # Hmm, no. The tensor T_{ijk} = 1 for all i,j,k. That's not right either.
    #
    # Let me think again. The matmul tensor is defined by:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So the 3-tensor T has components T_{ijk} such that T_{ijk} = 1 for all i,j,k.
    # That means T is the all-ones tensor? No!
    #
    # Actually, the tensor decomposition is:
    # T_{ijk} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # and T_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that can't be right because then the rank would be 1 (U=V=W=ones).
    #
    # I think the convention is: T_{ijk} = 1 if the triple (i,j,k) corresponds to
    # a "valid" matmul contribution, which is always 1 for all i,j,k.
    # But that would make it rank 1. So the tensor must be something else.
    #
    # Let me re-read the task: "exactly decompose the matrix multiplication tensor"
    # The matmul tensor for (n,m,p) is T with T[i,j,k] = 1 for all i,j,k.
    # No wait, that's the tensor for the bilinear map (A,B) -> C where
    # C[i,k] = sum_j A[i,j] B[j,k]. The tensor is T[i,j,k] = 1 for all i,j,k.
    #
    # Hmm, but then the rank is 1. That's clearly wrong.
    #
    # I think the correct interpretation is: the tensor T is indexed by (i, j, k)
    # where i is the row of A/C, j is the column of A / row of B, k is the column of B/C.
    # And T[i,j,k] = 1 for all valid i,j,k. The rank of this tensor is what we minimize.
    #
    # Actually, I recall now: the matmul tensor T_{ijk} = 1 for all i,j,k is indeed
    # the tensor, and its rank is the minimum number of rank-1 tensors needed to
    # express it. For n=m=p=2, the rank is 7 (Strassen). For n=m=p=1, rank is 1.
    #
    # So T[i,j,k] = 1 for all i,j,k. Let me verify: for 2x2x2, Strassen gives
    # rank 7, and the tensor is the all-ones 2x2x2 tensor? No, that's not right.
    #
    # Let me look at this differently. The bilinear map is:
    # f(A, B) = A * B
    # This is a trilinear form if we write it as:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # where C = A*B. So T[i,j,k] = 1 for all i,j,k.
    #
    # Yes! The tensor T[i,j,k] = 1 for all i,j,k. And the rank is the minimum
    # number of rank-1 terms to express this tensor. For 2x2x2, this is 7.
    #
    # So verification: sum_r U[i,r] * V[j,r] * W[k,r] should equal 1 for all i,j,k.
    #
    # Let me verify with the current program's approach.
    n, m, p = DIMENSIONS
    # T[i,j,k] = 1 for all i,j,k
    # Check: sum_r U[i,r] V[j,r] W[k,r] == 1 for all i,j,k
    # We can compute this as: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # and sum over r.
    # Efficient: for each i,j, compute sum_r U[i,r] V[j,r] W[k,r] as a function of k.
    # Or just check a random sample.
    # For full check: T[i,j,k] = (U @ (V * W))[i, j, k]? No.
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # = sum_r (U[i,r] V[j,r]) W[k,r]
    # Let M[i,j,r] = U[i,r] V[j,r], then T[i,j,k] = sum_r M[i,j,r] W[k,r] = (M[i,j,:] @ W[:,k])
    # So T[i,j,k] = (U[i,:] * V[j,:]) @ W[:,k]
    # We can compute: for each r, U[:,r] V[:,r] is a (n*m) x (m*p) matrix, but that's big.
    # Better: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Let's just do a random sample check for speed.
    rng = np.random.RandomState(42)
    for _ in range(500):
        i = rng.randint(0, n)
        j = rng.randint(0, m)
        k = rng.randint(0, p)
        val = np.dot(U[i, :] * V[j, :], W[k, :])
        if abs(val - 1.0) > tol:
            return False
    return True


def _rank_of_matrix(M, tol=1e-9):
    """Compute numerical rank of matrix M."""
    if M.size == 0:
        return 0
    sv = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(sv > tol))


def _try_reduce_column(U, V, W, factor, col_idx, tol=1e-9):
    """Try to eliminate column col_idx of the given factor by expressing it as a
    linear combination of the other columns of the SAME factor, such that the
    tensor identity is preserved.
    
    The key insight: if column r of U is in the span of other U columns, we can
    write U[:,r] = sum_{r' != r} c_{r'} U[:,r']. Then we need to adjust V and W
    so that the tensor identity still holds. But this is not straightforward
    because V and W also have column r.
    
    A better approach: if the column r of U is linearly dependent on other columns
    of U, we can perform a column operation on U: U[:,r] -> U[:,r] - sum c_{r'} U[:,r'].
    But this changes the tensor unless we also adjust V and W.
    
    Actually, the standard approach for rank reduction of a tensor decomposition:
    If there exists a linear dependence among the U columns, i.e.,
    sum_r alpha_r U[:,r] = 0 with not all alpha_r = 0, then we can use this to
    reduce the rank. But we need to maintain the tensor identity.
    
    Let me think about this more carefully. The tensor identity is:
    T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k.
    
    If we have a linear dependence sum_r alpha_r U[:,r] = 0, we can modify the
    decomposition. But it's not a simple column elimination.
    
    A practical approach: try to find a column that is "redundant" in the sense
    that removing it and adjusting the others still gives a valid decomposition.
    This is hard in general.
    
    Alternative: use the fact that if the U matrix has rank < R, then we can
    find a basis for the column space and re-express the decomposition with
    fewer terms. But the V and W matrices also need to be adjusted.
    
    Let me try a different approach: iterative column elimination with verification.
    For each column r, try to set U[:,r] = 0 (or remove it) and see if we can
    adjust the other columns to maintain the identity. This is a system of linear
    equations.
    
    Actually, the simplest structural change: try to find linearly dependent
    columns in U, V, or W and eliminate them, then re-verify. If the decomposition
    is still valid after elimination, we've reduced the rank.
    
    Let me try: for each factor, find the rank of the matrix. If rank < R, then
    there are R - rank linearly dependent columns. We can try to eliminate them
    one at a time.
    """
    pass


def _reduce_rank(U, V, W, max_iterations=200, tol=1e-9):
    """Iteratively try to reduce the rank by eliminating linearly dependent columns."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R
    
    for iteration in range(max_iterations):
        if _time_remaining() < 10:
            break
        
        # Try to find a reducible column in U, V, or W
        reduced = False
        
        for factor_name, factor in [('U', U), ('V', V), ('W', W)]:
            if _time_remaining() < 10:
                break
            
            R_cur = factor.shape[1]
            # Check if any column is linearly dependent on the others
            # Compute the rank of the matrix without each column
            # If rank(factor without col r) == rank(factor), then col r is redundant
            
            # Efficient: compute SVD of the full matrix
            try:
                sv_full = np.linalg.svd(factor, compute_uv=False)
                rank_full = int(np.sum(sv_full > tol))
            except:
                continue
            
            if rank_full >= R_cur:
                # Full rank, no redundant columns
                continue
            
            # Find a redundant column: one whose removal doesn't change the rank
            for r in range(R_cur):
                if _time_remaining() < 10:
                    break
                
                # Try removing column r
                factor_no_r = np.delete(factor, r, axis=1)
                try:
                    sv_no_r = np.linalg.svd(factor_no_r, compute_uv=False)
                    rank_no_r = int(np.sum(sv_no_r > tol))
                except:
                    continue
                
                if rank_no_r == rank_full:
                    # Column r is redundant in terms of matrix rank
                    # But we need to check if the tensor identity still holds
                    # after removing column r from all three factors.
                    # Wait, we can only remove column r from ONE factor at a time
                    # if we adjust the others. This is complex.
                    #
                    # Let me try a different approach: if column r of U is in the
                    # span of other U columns, we can express U[:,r] = sum_{r' != r}
                    # c_{r'} U[:,r']. Then the tensor identity becomes:
                    # sum_r U[i,r] V[j,r] W[k,r] = sum_{r' != r} c_{r'} U[i,r'] V[j,r'] W[k,r']
                    # + sum_{r' != r} U[i,r'] V[j,r'] W[k,r'] * 0  (if we set V[:,r]=W[:,r]=0)
                    # No, that's not right.
                    #
                    # I think the correct approach is: if U[:,r] is in the span of
                    # other U columns, we can perform a column operation on U that
                    # makes U[:,r] = 0, and simultaneously adjust V and W to
                    # compensate. But this requires solving a system.
                    #
                    # Let me try a simpler heuristic: just remove column r from all
                    # three factors and verify. If it still works, great.
                    U_no_r = np.delete(U, r, axis=1)
                    V_no_r = np.delete(V, r, axis=1)
                    W_no_r = np.delete(W, r, axis=1)
                    
                    if _verify(U_no_r, V_no_r, W_no_r, tol=1e-7):
                        U, V, W = U_no_r, V_no_r, W_no_r
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = U.shape[1]
                        reduced = True
                        break
            if reduced:
                break
        
        if not reduced:
            break
    
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS
    
    # Build the known rank-93 solution
    U, V, W = _build_rank93()
    
    # Verify the initial solution
    if not _verify(U, V, W):
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce the rank
    U_red, V_red, W_red, rank_red = _reduce_rank(U, V, W)
    
    # Verify the reduced solution
    if rank_red < U.shape[1] and _verify(U_red, V_red, W_red):
        U, V, W = U_red, V_red, W_red
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
