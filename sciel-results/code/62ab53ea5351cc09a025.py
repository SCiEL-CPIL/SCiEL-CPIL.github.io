"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _try_merge_two_terms(U, V, W, T, r1, r2, tol=1e-9):
    """
    Try to merge two rank-1 terms into one by checking if their combined
    contribution can be expressed as a single rank-1 tensor.
    Returns (True, merged_U_col, merged_V_col, merged_W_col) if mergeable, else (False, None, None, None).
    """
    # The sum of two rank-1 tensors u1⊗v1⊗w1 + u2⊗v2⊗w2 is rank-1
    # iff the three vectors in each mode are linearly dependent (rank 1 in each mode).
    # Specifically, we need: u1 and u2 linearly dependent, v1 and v2 linearly dependent,
    # w1 and w2 linearly dependent, OR some other rank-1 condition.
    
    # A more general check: the 2x2x2 tensor formed by the two terms has rank 1
    # iff all 2x2 minors of the unfolding are zero.
    
    u1, u2 = U[:, r1], U[:, r2]
    v1, v2 = V[:, r1], V[:, r2]
    w1, w2 = W[:, r1], W[:, r2]
    
    # Check if the combined contribution equals a rank-1 tensor.
    # The combined contribution is: u1⊗v1⊗w1 + u2⊗v2⊗w2
    # This is rank 1 iff there exist scalars α, β, γ such that
    # u1 = α*u', v1 = β*v', w1 = γ*w' and u2 = α'*u', v2 = β'*v', w2 = γ'*w'
    # for some u', v', w'. Equivalently, the vectors (u1, u2) span a 1-dim space,
    # (v1, v2) span a 1-dim space, and (w1, w2) span a 1-dim space.
    
    # Check linear dependence of u1, u2
    # If u1 and u2 are linearly dependent, one is a scalar multiple of the other.
    # Similarly for v and w.
    
    # For u: check if rank([u1, u2]) == 1
    U_pair = np.column_stack([u1, u2])
    # Find nonzero entries
    nz_u = np.nonzero(np.abs(U_pair).max(axis=1) > tol)[0]
    if len(nz_u) >= 2:
        # Check if rows are proportional
        row0 = U_pair[nz_u[0]]
        # Check if all other nonzero rows are proportional to row0
        for idx in nz_u[1:]:
            row = U_pair[idx]
            # Find a nonzero element in row0 to use as pivot
            pivot = np.nonzero(np.abs(row0) > tol)[0]
            if len(pivot) == 0:
                return False, None, None, None
            p = pivot[0]
            if abs(row0[p]) < tol:
                return False, None, None, None
            scale = row[p] / row0[p]
            if not np.allclose(row, scale * row0, atol=tol):
                return False, None, None, None
    elif len(nz_u) == 1:
        # Only one nonzero row, so they're trivially dependent (one or both zero in that row)
        pass
    else:
        # Both zero, trivially dependent
        pass
    
    # Same check for v
    V_pair = np.column_stack([v1, v2])
    nz_v = np.nonzero(np.abs(V_pair).max(axis=1) > tol)[0]
    if len(nz_v) >= 2:
        row0 = V_pair[nz_v[0]]
        for idx in nz_v[1:]:
            row = V_pair[idx]
            pivot = np.nonzero(np.abs(row0) > tol)[0]
            if len(pivot) == 0:
                return False, None, None, None
            p = pivot[0]
            if abs(row0[p]) < tol:
                return False, None, None, None
            scale = row[p] / row0[p]
            if not np.allclose(row, scale * row0, atol=tol):
                return False, None, None, None
    
    # Same check for w
    W_pair = np.column_stack([w1, w2])
    nz_w = np.nonzero(np.abs(W_pair).max(axis=1) > tol)[0]
    if len(nz_w) >= 2:
        row0 = W_pair[nz_w[0]]
        for idx in nz_w[1:]:
            row = W_pair[idx]
            pivot = np.nonzero(np.abs(row0) > tol)[0]
            if len(pivot) == 0:
                return False, None, None, None
            p = pivot[0]
            if abs(row0[p]) < tol:
                return False, None, None, None
            scale = row[p] / row0[p]
            if not np.allclose(row, scale * row0, atol=tol):
                return False, None, None, None
    
    # If all three pairs are linearly dependent, the sum is rank 1.
    # The merged term is simply the sum (since it's already rank 1).
    merged_u = u1 + u2
    merged_v = v1 + v2
    merged_w = w1 + w2
    
    # Verify the merge is exact
    # Check that u1⊗v1⊗w1 + u2⊗v2⊗w2 == merged_u⊗merged_v⊗merged_w
    left = np.einsum('a,b,c->abc', u1, v1, w1) + np.einsum('a,b,c->abc', u2, v2, w2)
    right = np.einsum('a,b,c->abc', merged_u, merged_v, merged_w)
    if not np.allclose(left, right, atol=tol):
        return False, None, None, None
    
    return True, merged_u, merged_v, merged_w

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
    
    # Index mapping:
    # a: A[i,j] -> i*3+j
    #   a0=A[0,0], a1=A[0,1], a2=A[0,2], a3=A[1,0], a4=A[1,1], a5=A[1,2]
    # b: B[j,k] -> j*3+k
    #   b0=B[0,0], b1=B[0,1], b2=B[0,2], b3=B[1,0], b4=B[1,1], b5=B[1,2],
    #   b6=B[2,0], b7=B[2,1], b8=B[2,2]
    # c: C[i,k] -> k*2+i
    #   c0=C[0,0], c1=C[1,0], c2=C[0,1], c3=C[1,1], c4=C[0,2], c5=C[1,2]
    
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    # A1 = [[a0, a1], [a3, a4]]
    # B1_left = [[b0, b1], [b3, b4]]
    # Output: C[0,0]=c0, C[1,0]=c1, C[0,1]=c2, C[1,1]=c3
    
    # M1 = (a0+a4)*(b0+b4)
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    # M2 = (a3+a4)*b0
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    # M3 = a0*(b1-b4)
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    # M4 = a4*(b3-b0)
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    # M5 = (a0+a1)*b4
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    # M6 = (a3-a0)*(b0+b1)
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    # M7 = (a1-a4)*(b3+b4)
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    # C[0,2]=c4, C[1,2]=c5
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    set_term(7, {0:1}, {2:1}, {4:1})   # A[0,0]*B[0,2] -> C[0,2]
    set_term(8, {1:1}, {5:1}, {4:1})   # A[0,1]*B[1,2] -> C[0,2]
    set_term(9, {3:1}, {2:1}, {5:1})   # A[1,0]*B[0,2] -> C[1,2]
    set_term(10, {4:1}, {5:1}, {5:1})  # A[1,1]*B[1,2] -> C[1,2]
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    # C[i,k] += A[i,2]*B[2,k]
    set_term(11, {2:1}, {6:1}, {0:1})  # A[0,2]*B[2,0] -> C[0,0]
    set_term(12, {2:1}, {7:1}, {2:1})  # A[0,2]*B[2,1] -> C[0,1]
    set_term(13, {2:1}, {8:1}, {4:1})  # A[0,2]*B[2,2] -> C[0,2]
    set_term(14, {5:1}, {6:1}, {1:1})  # A[1,2]*B[2,0] -> C[1,0]
    set_term(15, {5:1}, {7:1}, {3:1})  # A[1,2]*B[2,1] -> C[1,1]
    set_term(16, {5:1}, {8:1}, {5:1})  # A[1,2]*B[2,2] -> C[1,2]
    
    # Build tensor for verification
    T = _build_tensor(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Attempt to reduce rank by merging pairs of terms
    # This is a best-effort polish; rank 17 is proven optimal, but we try anyway.
    if time.time() < deadline - 5:
        current_rank = rank
        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        
        # Try merging all pairs, starting from the end
        merged = True
        while merged and time.time() < deadline - 5:
            merged = False
            for r1 in range(current_rank - 1, -1, -1):
                if time.time() >= deadline - 5:
                    break
                for r2 in range(r1 - 1, -1, -1):
                    if time.time() >= deadline - 5:
                        break
                    success, mu, mv, mw = _try_merge_two_terms(U_cur, V_cur, W_cur, T, r1, r2)
                    if success:
                        # Replace terms r1 and r2 with merged term
                        U_cur[:, r1] = mu
                        V_cur[:, r1] = mv
                        W_cur[:, r1] = mw
                        # Remove term r2 by shifting
                        if r2 < current_rank - 1:
                            U_cur[:, r2:current_rank-1] = U_cur[:, r2+1:current_rank]
                            V_cur[:, r2:current_rank-1] = V_cur[:, r2+1:current_rank]
                            W_cur[:, r2:current_rank-1] = W_cur[:, r2+1:current_rank]
                        current_rank -= 1
                        merged = True
                        # Verify after merge
                        if not _verify_decomposition(U_cur[:, :current_rank], V_cur[:, :current_rank], W_cur[:, :current_rank], T):
                            # Revert
                            U_cur = U.copy()
                            V_cur = V.copy()
                            W_cur = W.copy()
                            current_rank = rank
                            merged = False
                            break
                        break  # Restart pair search after a merge
        
        if current_rank < rank:
            U = U_cur[:, :current_rank]
            V = V_cur[:, :current_rank]
            W = W_cur[:, :current_rank]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
