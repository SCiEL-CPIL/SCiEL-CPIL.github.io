import time
import random
import itertools

def construct():
    _t0 = time.time()
    DEADLINE = _t0 + 115.0  # safety margin before 150s kill
    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m x p

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def build_decomposition(a_row_splits, a_col_splits, b_row_splits, b_col_splits, 
                            small_block_rank_func, large_block_rank_func):
        """Build a decomposition given partition strategies.
        a_row_splits: list of row sizes for A, summing to n
        a_col_splits: list of col sizes for A, summing to m (must match b_row_splits)
        b_row_splits: list of row sizes for B, summing to m (must match a_col_splits)
        b_col_splits: list of col sizes for B, summing to p
        """
        RANK = 0
        U = []
        V = []
        W = []
        col = 0

        # Compute total rank
        a_row_starts = [0]
        for s in a_row_splits[:-1]:
            a_row_starts.append(a_row_starts[-1] + s)
        a_col_starts = [0]
        for s in a_col_splits[:-1]:
            a_col_starts.append(a_col_starts[-1] + s)
        b_row_starts = [0]
        for s in b_row_splits[:-1]:
            b_row_starts.append(b_row_starts[-1] + s)
        b_col_starts = [0]
        for s in b_col_splits[:-1]:
            b_col_starts.append(b_col_starts[-1] + s)

        # For each A row block, A col block, B row block, B col block
        # where A col block index == B row block index
        for ar_i, ar_size in enumerate(a_row_splits):
            ar_start = a_row_starts[ar_i]
            for ac_i, ac_size in enumerate(a_col_splits):
                ac_start = a_col_starts[ac_i]
                # B row block must match A col block
                br_start = b_row_starts[ac_i]
                br_size = b_row_splits[ac_i]
                for bc_i, bc_size in enumerate(b_col_splits):
                    bc_start = b_col_starts[bc_i]
                    # Block: A[ar_start:ar_start+ar_size, ac_start:ac_start+ac_size] 
                    #        * B[br_start:br_start+br_size, bc_start:bc_start+bc_size]
                    #        -> C[ar_start:ar_start+ar_size, bc_start:bc_start+bc_size]
                    if ar_size == 1 and br_size == 2:
                        # 1x2 * 2x2 -> 1x2, rank 4 (schoolbook)
                        rank = 4
                        RANK += rank
                    elif ar_size == 2 and br_size == 2:
                        # 2x2 * 2x2 -> 2x2, rank 7 (Strassen)
                        rank = 7
                        RANK += rank
                    else:
                        # Fallback: schoolbook
                        rank = ar_size * br_size * bc_size  # This is wrong, let me think
                        # Actually for general blocks, schoolbook rank = ar_size * br_size * bc_size / bc_size... 
                        # No, schoolbook for (a×b)*(b×c) -> a×c has rank a*b*c
                        # But we only need the specific structure
                        rank = ar_size * bc_size * br_size
                        RANK += rank

        # Initialize matrices
        U = [[0] * RANK for _ in range(n * m)]
        V = [[0] * RANK for _ in range(m * p)]
        W = [[0] * RANK for _ in range(p * n)]
        col = 0

        def _add_term(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for u_i, u_c in zip(u_indices, u_coeffs):
                U[u_i][col] += u_c
            for v_i, v_c in zip(v_indices, v_coeffs):
                V[v_i][col] += v_c
            for w_i, w_c in zip(w_indices, w_coeffs):
                W[w_i][col] += w_c
            col += 1

        # Build terms
        for ar_i, ar_size in enumerate(a_row_splits):
            ar_start = a_row_starts[ar_i]
            for ac_i, ac_size in enumerate(a_col_splits):
                ac_start = a_col_starts[ac_i]
                br_start = b_row_starts[ac_i]
                br_size = b_row_splits[ac_i]
                for bc_i, bc_size in enumerate(b_col_splits):
                    bc_start = b_col_starts[bc_i]

                    if ar_size == 1 and br_size == 2:
                        # 1x2 * 2x2 -> 1x2, schoolbook rank 4
                        A0 = _idx_u(ar_start, ac_start)
                        A1 = _idx_u(ar_start, ac_start + 1)
                        B00 = _idx_v(br_start, bc_start)
                        B01 = _idx_v(br_start, bc_start + 1)
                        B10 = _idx_v(br_start + 1, bc_start)
                        B11 = _idx_v(br_start + 1, bc_start + 1)
                        C0 = _idx_w(bc_start, ar_start)
                        C1 = _idx_w(bc_start + 1, ar_start)

                        # M1 = A0 * B00 -> C0
                        _add_term([A0], [B00], [C0], [1], [1], [1])
                        # M2 = A1 * B10 -> C0
                        _add_term([A1], [B10], [C0], [1], [1], [1])
                        # M3 = A0 * B01 -> C1
                        _add_term([A0], [B01], [C1], [1], [1], [1])
                        # M4 = A1 * B11 -> C1
                        _add_term([A1], [B11], [C1], [1], [1], [1])

                    elif ar_size == 2 and br_size == 2:
                        # 2x2 * 2x2 -> 2x2, Strassen rank 7
                        A00 = _idx_u(ar_start, ac_start)
                        A01 = _idx_u(ar_start, ac_start + 1)
                        A10 = _idx_u(ar_start + 1, ac_start)
                        A11 = _idx_u(ar_start + 1, ac_start + 1)
                        B00 = _idx_v(br_start, bc_start)
                        B01 = _idx_v(br_start, bc_start + 1)
                        B10 = _idx_v(br_start + 1, bc_start)
                        B11 = _idx_v(br_start + 1, bc_start + 1)
                        C00 = _idx_w(bc_start, ar_start)
                        C01 = _idx_w(bc_start + 1, ar_start)
                        C10 = _idx_w(bc_start, ar_start + 1)
                        C11 = _idx_w(bc_start + 1, ar_start + 1)

                        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
                        _add_term([A00, A11], [B00, B11], [C00, C11], [1, 1], [1, 1], [1, 1])
                        # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
                        _add_term([A10, A11], [B00], [C10, C11], [1, 1], [1], [1, -1])
                        # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
                        _add_term([A00], [B01, B11], [C01, C11], [1], [1, -1], [1, 1])
                        # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
                        _add_term([A11], [B10, B00], [C00, C10], [1], [1, -1], [1, 1])
                        # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
                        _add_term([A00, A01], [B11], [C00, C01], [1, 1], [1], [-1, 1])
                        # M6 = (A10-A00)(B00+B01) -> C11 += 1
                        _add_term([A10, A00], [B00, B01], [C11], [1, -1], [1, 1], [1])
                        # M7 = (A01-A11)(B10+B11) -> C00 += 1
                        _add_term([A01, A11], [B10, B11], [C00], [1, -1], [1, 1], [1])

                    else:
                        # Schoolbook for general case
                        for di in range(ar_size):
                            for dj in range(bc_size):
                                for dk in range(br_size):
                                    u_i = _idx_u(ar_start + di, ac_start + dk)
                                    v_i = _idx_v(br_start + dk, bc_start + dj)
                                    w_i = _idx_w(bc_start + dj, ar_start + di)
                                    _add_term([u_i], [v_i], [w_i], [1], [1], [1])

        assert col == RANK, f"Expected {RANK} terms, got {col}"
        return U, V, W, RANK

    def verify(U, V, W, rank, n=3, m=4, p=6):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # Our tensor: sum_c U[i,c] * V[k,c] * W[j*n+i, c] should equal delta_{i,k,j} pattern
        # Actually, the matrix multiplication tensor T[i,k,j] = 1 if we're computing C[i,j] from A[i,k]*B[k,j]
        # The decomposition should satisfy: sum_c U[i,c] * V[k,c] * W[j*n+i, c] = 1 if (i,k,j) is a valid triple
        # Wait, the tensor for matrix multiplication is: T[i,k,j] = 1 for all i,k,j (it's the identity in the sense that
        # C[i,j] = sum_k A[i,k] B[k,j], so the tensor has T[i,k,j] = 1 for all valid (i,k,j))
        # Actually no. The tensor is defined such that:
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # In tensor form: sum_c U[i,c] V[k,c] W[j,c] = T[i,k,j] where T[i,k,j] = 1 for all i,k,j
        # Wait, that's not right either. Let me think again.
        #
        # The matrix multiplication tensor T ∈ R^{n×m × m×p × p×n} (with W indexed by j*n+i)
        # is defined by: T[i*m+k, k*p+j, j*n+i] = 1 for all valid i,k,j, and 0 otherwise.
        # But since we're using the trace convention where W is indexed by (j,i) -> j*n+i,
        # the tensor entry at (u_idx, v_idx, w_idx) where u_idx = i*m+k, v_idx = k*p+j, w_idx = j*n+i
        # should be 1 if the indices are consistent (i.e., the k in u matches the k in v, 
        # and the j in v matches the j in w, and the i in u matches the i in w), else 0.
        #
        # For our decomposition: sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c] = T[u_idx, v_idx, w_idx]
        # where T[u_idx, v_idx, w_idx] = 1 if u_idx = i*m+k, v_idx = k*p+j, w_idx = j*n+i for some i,k,j,
        # and 0 otherwise. But since we only evaluate at valid (i,k,j) triples, T = 1 for all such triples.
        #
        # So we need: sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c] = 1 for all i,k,j.
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + k][c] * V[k * p + j][c] * W[j * n + i][c]
                    if s != 1:
                        return False
        return True

    def try_reduce_rank(U, V, W, rank, n=3, m=4, p=6, max_iters=500):
        """Try to reduce rank by dropping terms that are linearly dependent."""
        MOD = 11
        rng = random.Random(12345)
        
        def _build_tensor_matrix(U, V, W, rank):
            """Build the 72xrank matrix over GF(11)."""
            rows = n * m * p
            M = [[0] * rank for _ in range(rows)]
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        row_idx = i * m * p + k * p + j
                        for c in range(rank):
                            M[row_idx][c] = (U[i * m + k][c] * V[k * p + j][c] * W[j * n + i][c]) % MOD
            return M

        def _find_null_vector(M, rank):
            """Find a null vector of M over GF(11)."""
            rows = len(M)
            A = [row[:] for row in M]
            pivot_cols = []
            row_ptr = 0
            
            for col_idx in range(rank):
                if row_ptr >= rows:
                    break
                # Find pivot
                pivot_row = -1
                for r in range(row_ptr, rows):
                    if A[r][col_idx] != 0:
                        pivot_row = r
                        break
                if pivot_row == -1:
                    continue
                A[row_ptr], A[pivot_row] = A[pivot_row], A[row_ptr]
                inv = pow(A[row_ptr][col_idx], MOD - 2, MOD)
                for c in range(col_idx, rank):
                    A[row_ptr][c] = (A[row_ptr][c] * inv) % MOD
                for r in range(rows):
                    if r == row_ptr:
                        continue
                    factor = A[r][col_idx]
                    if factor != 0:
                        for c in range(col_idx, rank):
                            A[r][c] = (A[r][c] - factor * A[row_ptr][c]) % MOD
                pivot_cols.append(col_idx)
                row_ptr += 1
            
            free_cols = [c for c in range(rank) if c not in pivot_cols]
            if not free_cols:
                return None
            
            # Find one null vector
            alpha = [0] * rank
            fc = free_cols[0]
            alpha[fc] = 1
            for pr, pc in enumerate(pivot_cols):
                s = 0
                for f in free_cols:
                    s = (s + A[pr][f] * alpha[f]) % MOD
                alpha[pc] = (-s) % MOD
            
            # Verify
            for r in range(rows):
                s = 0
                for c in range(rank):
                    s = (s + M[r][c] * alpha[c]) % MOD
                if s != 0:
                    return None
            return alpha

        best_U, best_V, best_W, best_rank = U, V, W, rank
        current_U, current_V, current_W, current_rank = U, V, W, rank
        
        for iteration in range(max_iters):
            if time.time() > DEADLINE - 10:
                break
            
            # Build tensor matrix and find null vector
            M = _build_tensor_matrix(current_U, current_V, current_W, current_rank)
            alpha = _find_null_vector(M, current_rank)
            
            if alpha is None:
                break  # No more dependencies
            
            # Find a term to eliminate
            # We want to eliminate the term with the largest coefficient in magnitude
            # (over integers, not GF(11))
            # But the null vector is over GF(11), so we need to interpret it carefully.
            # 
            # The dependency is: sum_c alpha_c * T_c = 0 (over GF(11))
            # This means: sum_c alpha_c * U_c ⊗ V_c ⊗ W_c = 0 (over GF(11))
            # But we need this to hold over the integers, not just GF(11).
            #
            # If the entries are small integers, the GF(11) dependency might not
            # correspond to an integer dependency. So this approach is flawed.
            #
            # Alternative: try to drop a term and check if the remaining terms
            # still span the tensor space.
            
            # Try dropping each term and check validity
            for c_drop in range(current_rank):
                if alpha[c_drop] == 0:
                    continue
                
                # Remove term c_drop
                keep = [c for c in range(current_rank) if c != c_drop]
                new_rank = current_rank - 1
                U_new = [[current_U[i][c] for c in keep] for i in range(n * m)]
                V_new = [[current_V[i][c] for c in keep] for i in range(m * p)]
                W_new = [[current_W[i][c] for c in keep] for i in range(p * n)]
                
                if verify(U_new, V_new, W_new, new_rank):
                    current_U, current_V, current_W, current_rank = U_new, V_new, W_new, new_rank
                    best_U, best_V, best_W, best_rank = current_U, current_V, current_W, current_rank
                    break  # Restart the loop with the new state
        
        return best_U, best_V, best_W, best_rank

    # Generate candidate partition strategies
    candidates = []
    
    # Strategy 1: Current best (1+2 rows, 2+2 cols, 2+2 B rows, 2+2+2 B cols)
    candidates.append(([1, 2], [2, 2], [2, 2], [2, 2, 2]))
    
    # Strategy 2: 2+1 rows, 2+2 cols, 2+2 B rows, 2+2+2 B cols
    candidates.append(([2, 1], [2, 2], [2, 2], [2, 2, 2]))
    
    # Strategy 3: 1+1+1 rows, 2+2 cols, 2+2 B rows, 2+2+2 B cols
    candidates.append(([1, 1, 1], [2, 2], [2, 2], [2, 2, 2]))
    
    # Strategy 4: 1+2 rows, 1+1+2 cols, 1+1+2 B rows, 2+2+2 B cols
    candidates.append(([1, 2], [1, 1, 2], [1, 1, 2], [2, 2, 2]))
    
    # Strategy 5: 2+1 rows, 1+1+2 cols, 1+1+2 B rows, 2+2+2 B cols
    candidates.append(([2, 1], [1, 1, 2], [1, 1, 2], [2, 2, 2]))
    
    # Strategy 6: 1+2 rows, 2+2 cols, 2+2 B rows, 1+1+2+2 B cols (asymmetric B cols)
    candidates.append(([1, 2], [2, 2], [2, 2], [1, 1, 2, 2]))
    
    # Strategy 7: 1+2 rows, 2+2 cols, 2+2 B rows, 3+3 B cols
    candidates.append(([1, 2], [2, 2], [2, 2], [3, 3]))
    
    # Strategy 8: 2+1 rows, 2+2 cols, 2+2 B rows, 3+3 B cols
    candidates.append(([2, 1], [2, 2], [2, 2], [3, 3]))
    
    # Strategy 9: 1+1+1 rows, 1+1+2 cols, 1+1+2 B rows, 2+2+2 B cols
    candidates.append(([1, 1, 1], [1, 1, 2], [1, 1, 2], [2, 2, 2]))
    
    # Strategy 10: 1+2 rows, 3+1 cols, 3+1 B rows, 2+2+2 B cols
    candidates.append(([1, 2], [3, 1], [3, 1], [2, 2, 2]))
    
    # Strategy 11: 2+1 rows, 3+1 cols, 3+1 B rows, 2+2+2 B cols
    candidates.append(([2, 1], [3, 1], [3, 1], [2, 2, 2]))
    
    # Strategy 12: 1+2 rows, 2+2 cols, 2+2 B rows, 1+2+3 B cols
    candidates.append(([1, 2], [2, 2], [2, 2], [1, 2, 3]))
    
    # Strategy 13: 2+1 rows, 2+2 cols, 2+2 B rows, 1+2+3 B cols
    candidates.append(([2, 1], [2, 2], [2, 2], [1, 2, 3]))
    
    # Strategy 14: 1+2 rows, 1+3 cols, 1+3 B rows, 2+2+2 B cols
    candidates.append(([1, 2], [1, 3], [1, 3], [2, 2, 2]))
    
    # Strategy 15: 2+1 rows, 1+3 cols, 1+3 B rows, 2+2+2 B cols
    candidates.append(([2, 1], [1, 3], [1, 3], [2, 2, 2]))
    
    best_result = None
    best_rank = float('inf')
    
    for a_rows, a_cols, b_rows, b_cols in candidates:
        if time.time() > DEADLINE - 20:
            break
        
        # Validate that a_cols and b_rows match
        if a_cols != b_rows:
            continue
        if sum(a_rows) != n or sum(a_cols) != m or sum(b_rows) != m or sum(b_cols) != p:
            continue
        
        try:
            U, V, W, rank = build_decomposition(a_rows, a_cols, b_rows, b_cols, None, None)
            
            if not verify(U, V, W, rank):
                continue
            
            # Try to reduce rank
            U, V, W, rank = try_reduce_rank(U, V, W, rank)
            
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W, rank)
                
        except Exception as e:
            continue
    
    if best_result is None:
        # Fallback to the known good 66-rank decomposition
        U, V, W, rank = build_decomposition([1, 2], [2, 2], [2, 2], [2, 2, 2], None, None)
        assert verify(U, V, W, rank)
        return {"U": U, "V": V, "W": W}
    
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}
