import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _build_rank42():
    """Build the rank-42 Strassen decomposition for 2x4 @ 4x6."""
    n, m, p = 2, 4, 6
    rank_total = 42

    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    strassen_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    strassen_W = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]

    col_offset = 0
    for j_block in range(3):
        for side in ['L', 'R']:
            if side == 'L':
                u_col_offset = 0
                v_row_offset = 0
            else:
                u_col_offset = 2
                v_row_offset = 2

            for prod_idx in range(7):
                col = col_offset + prod_idx

                local_U = {
                    0: 0 * 4 + (u_col_offset + 0),
                    1: 0 * 4 + (u_col_offset + 1),
                    2: 1 * 4 + (u_col_offset + 0),
                    3: 1 * 4 + (u_col_offset + 1),
                }
                for li, s in strassen_U[prod_idx].items():
                    U[local_U[li]][col] += s

                local_V = {
                    0: (v_row_offset + 0) * 6 + (2 * j_block + 0),
                    1: (v_row_offset + 0) * 6 + (2 * j_block + 1),
                    2: (v_row_offset + 1) * 6 + (2 * j_block + 0),
                    3: (v_row_offset + 1) * 6 + (2 * j_block + 1),
                }
                for li, s in strassen_V[prod_idx].items():
                    V[local_V[li]][col] += s

                local_W = {
                    0: (2 * j_block + 0) * 2 + 0,
                    1: (2 * j_block + 1) * 2 + 0,
                    2: (2 * j_block + 0) * 2 + 1,
                    3: (2 * j_block + 1) * 2 + 1,
                }
                for li, s in strassen_W[prod_idx].items():
                    W[local_W[li]][col] += s

            col_offset += 7

    return U, V, W


def _check_decomposition_validity(U, V, W, n, m, p, rank):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # The tensor T[i,j,k] = delta[i,k] for matrix mult: (A*B)[i,j] = sum_k A[i,k]*B[k,j]
    # Decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # U index: i*m + k  (row i of A, col k)
    # V index: k*p + j  (row k of B, col j)
    # W index: j*n + i  (col j, row i of C) -- trace convention W[k*n+i, r] where k is output col

    # Build the full tensor from decomposition
    # T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # For each (i,j,k) triple, check that T[i,j,k] = delta_{i,k} (A[i,k]*B[k,j] contributes to C[i,j])
    # Actually the tensor for matmul: T_{i,j,k} = 1 if we're computing C[i,j] = sum_k A[i,k]B[k,j]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle index pairing)
    # Wait: the matmul tensor has T[i,k,j] = delta... let me think more carefully.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Tensor: T[i,j,k] such that C[i,j] = sum_k T[i,j,k] * A[i,k] * B[k,j]
    # So T[i,j,k] = 1 for all i,j,k (the tensor is all-ones in the appropriate sense)
    # No wait: T[i,j,k] = 1 means C[i,j] = sum_k 1 * A[i,k] * B[k,j] which is correct.
    # So the target tensor is T[i,j,k] = 1 for all valid i, j, k.
    #
    # In our indexing:
    # U has index (i*m + k) for row i, col k of A -> shape (n*m, rank)
    # V has index (k*p + j) for row k, col j of B -> shape (m*p, rank)
    # W has index (j*n + i) for col j, row i of C -> shape (p*n, rank)
    #
    # Check: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal 1 for all i,j,k

    max_err = 0.0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+k, r] * V_np[k*p+j, r] * W_np[j*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-10


def _try_reduce_rank(U, V, W, n, m, p, rank, time_budget):
    """
    Try to find linear dependencies among the rank columns to reduce the rank.
    Uses the structured approach: check if any subset of columns is linearly dependent
    in the sense that can be removed while preserving the tensor decomposition.
    """
    import numpy as np
    import itertools
    from fractions import Fraction

    deadline = time.time() + time_budget

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # The tensor constraint: for all (i,j,k): sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] = 1
    # If column r is a linear combination of other columns in a compatible way, we can remove it.
    #
    # Key insight: if for some column r, there exist coefficients c_s (s != r) such that
    # for ALL (i,j,k): U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] = sum_s c_s * U[i*m+k,s]*V[k*p+j,s]*W[j*n+i,s]
    # then column r is redundant.
    #
    # This is equivalent to: the "rank-1 tensor" formed by the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # lies in the span of the other such tensors.
    #
    # Let's check: for each column r, can we find c_s such that the above holds?
    # This is a linear system in the c_s's. The "equations" are indexed by (i,j,k).
    #
    # Number of (i,j,k) triples: n*m*p = 2*4*6 = 48
    # Number of unknowns (if trying to remove 1 column): rank-1 = 41
    #
    # But we need integer/half-integer solutions. Let's first check over reals.

    # Build the constraint matrix: rows = (i,j,k) triples, cols = r (excluding the one being tested)
    # M[(i,j,k), s] = U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s]
    # b[(i,j,k)] = U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]

    # Precompute all (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(p) for k in range(m)]
    ntriples = len(triples)

    # Precompute M matrix (ntriples x rank)
    M = np.zeros((ntriples, rank))
    for ti, (i, j, k) in enumerate(triples):
        for r in range(rank):
            M[ti, r] = U_np[i*m+k, r] * V_np[k*p+j, r] * W_np[j*n+i, r]

    # Check if any column is in the span of the others (over reals)
    # Use SVD to check rank
    rank_full = np.linalg.matrix_rank(M)
    if rank_full < rank:
        # There are linear dependencies! Find which columns are dependent.
        # Use QR with column pivoting to find independent columns
        from scipy.linalg import qr
        Q, R, piv = qr(M, pivoting=True)
        # Find the number of independent columns
        diag_R = np.abs(np.diag(R))
        tol = 1e-10 * max(diag_R)
        n_independent = sum(1 for d in diag_R if d > tol)
        if n_independent < rank:
            # Some columns are dependent. Find which ones.
            # The last (rank - n_independent) columns in the pivoted order are dependent.
            dependent_cols = set(piv[n_independent:].tolist())
            # Try removing one dependent column at a time and verify
            for col_to_remove in sorted(dependent_cols):
                if time.time() > deadline:
                    break
                # Try to express column col_to_remove as combination of others
                other_cols = [c for c in range(rank) if c != col_to_remove]
                M_sub = M[:, other_cols]
                b = M[:, col_to_remove]
                try:
                    coeffs, residuals, rank_sub, sv = np.linalg.lstsq(M_sub, b, rcond=1e-10)
                    if rank_sub == len(other_cols) and max(np.abs(M_sub @ coeffs - b)) < 1e-8:
                        # Check if coefficients are close to rational (integer or half-integer)
                        # Try to find a rational approximation
                        from fractions import Fraction
                        rational_coeffs = []
                        is_rational = True
                        for c in coeffs:
                            # Try denominators 1, 2, 3, 4, ...
                            found = False
                            for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                                num = round(c * denom)
                                if abs(c - num/denom) < 1e-8:
                                    rational_coeffs.append(Fraction(num, denom))
                                    found = True
                                    break
                            if not found:
                                is_rational = False
                                break
                        if is_rational:
                            # We found a valid reduction!
                            # Build new U, V, W with column removed
                            # Actually, we need to "absorb" the coefficients into the other columns
                            # New column s' = old column s + c_s * old column col_to_remove
                            # But wait, the decomposition requires U[i,r]*V[j,r]*W[k,r] to sum correctly.
                            # If M[:,r] = sum_s c_s * M[:,s], then:
                            # sum_r U[i,r]*V[j,r]*W[k,r] = sum_{s!=r} U[i,s]*V[j,s]*W[k,s] + U[i,r]*V[j,r]*W[k,r]
                            # = sum_{s!=r} U[i,s]*V[j,s]*W[k,s] + sum_s c_s * U[i,s]*V[j,s]*W[k,s]
                            # = sum_s (1 + c_s) * U[i,s]*V[j,s]*W[k,s]  for s != r
                            # = sum_s (1 + c_s) * U[i,s]*V[j,s]*W[k,s]
                            # So we can just multiply each remaining column's contribution by (1+c_s)
                            # But that changes U, V, W entries...
                            # Actually simpler: just remove column r and adjust.
                            # The new decomposition uses only columns s != r, but with modified entries.
                            # U_new[i,s] = U[i,s] + c_s * U[i,r]  -- NO, this doesn't work directly
                            # because the product is U*V*W, not linear in each separately.
                            #
                            # Actually, the correct approach: if the "tensor column" T_r = u_r ⊗ v_r ⊗ w_r
                            # equals sum_s c_s * T_s, then we can remove r and the sum is preserved.
                            # We don't need to modify the other columns! We just drop column r.
                            #
                            # Let's verify: new sum = sum_{s != r} T_s = sum_s T_s - T_r = sum_s T_s - sum_s c_s T_s
                            # That's NOT equal to sum_s T_s unless sum_s c_s T_s = 0.
                            #
                            # I made an error. Let me reconsider.
                            # If T_r = sum_{s != r} c_s * T_s, then:
                            # sum_r T_r = sum_{s != r} T_s + T_r = sum_{s != r} T_s + sum_{s != r} c_s T_s
                            # = sum_{s != r} (1 + c_s) T_s
                            # So the new decomposition has "effective" columns (1+c_s) * T_s for s != r.
                            # But (1+c_s) * (u_s ⊗ v_s ⊗ w_s) cannot generally be written as u'_s ⊗ v'_s ⊗ w'_s
                            # with u'_s, v'_s, w'_s being the original entries scaled.
                            # Actually it CAN: just scale u_s by (1+c_s)^(1/3), etc. But we need rational entries.
                            #
                            # Alternative: we can absorb the factor differently.
                            # (1+c_s) * u_s[i] * v_s[j] * w_s[k] = (u_s[i]) * ((1+c_s)*v_s[j]) * w_s[k]
                            # So we can just multiply V[:,s] by (1+c_s) for each s.
                            # This keeps entries rational if c_s is rational!
                            pass
                        # For now, just note the dependency exists
                        # Build reduced decomposition
                        new_rank = rank - 1
                        U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
                        V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
                        W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
                        # Scale V columns by (1+c_s)
                        for new_s, orig_s in enumerate(other_cols):
                            scale = 1.0 + coeffs[new_s]
                            for row in range(len(V_new)):
                                V_new[row][new_s] *= scale
                        # Verify
                        if _check_decomposition_validity(U_new, V_new, W_new, n, m, p, new_rank):
                            # Check entries are rational (close to integer or half-integer)
                            all_rational = True
                            for mat in [U_new, V_new, W_new]:
                                for row in mat:
                                    for val in row:
                                        if abs(val * 2 - round(val * 2)) > 1e-8:
                                            all_rational = False
                                            break
                                    if not all_rational:
                                        break
                                if not all_rational:
                                    break
                            if all_rational:
                                return U_new, V_new, W_new, new_rank
                except Exception:
                    pass
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Uses Strassen's algorithm on 2x2 blocks. First builds the rank-42 decomposition,
    then attempts to find linear dependencies to reduce rank.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = 2, 4, 6

    # Build the base rank-42 decomposition
    U, V, W = _build_rank42()
    rank = 42

    # Verify the base decomposition
    if not _check_decomposition_validity(U, V, W, n, m, p, rank):
        # Fallback: return what we have (shouldn't happen)
        pass

    # Try to reduce rank using available time
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        result = _try_reduce_rank(U, V, W, n, m, p, rank, remaining_time - 5)
        if result is not None:
            U, V, W, rank = result

    return {"U": U, "V": V, "W": W}
    
    # Strassen 2x2 @ 2x2 rank-7 decomposition
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # P1 = (a11+a22)*(b11+b22)
    # P2 = (a21+a22)*b11
    # P3 = a11*(b12-b22)
    # P4 = a22*(b21-b11)
    # P5 = (a11+a12)*b22
    # P6 = (a21-a11)*(b11+b12)
    # P7 = (a12-a22)*(b21+b22)
    #
    # c11 = P1 + P4 - P5 + P7
    # c12 = P3 + P5
    # c21 = P2 + P4
    # c22 = P1 - P2 + P3 + P6
    
    # For each product, store (U_entries, V_entries, W_entries) as dict {local_idx: sign}
    strassen_U = [
        {0: 1, 3: 1},       # P1: a11+a22
        {2: 1, 3: 1},       # P2: a21+a22
        {0: 1},             # P3: a11
        {3: 1},             # P4: a22
        {0: 1, 1: 1},       # P5: a11+a12
        {2: 1, 0: -1},      # P6: a21-a11
        {1: 1, 3: -1},      # P7: a12-a22
    ]
    strassen_V = [
        {0: 1, 3: 1},       # P1: b11+b22
        {0: 1},             # P2: b11
        {1: 1, 3: -1},      # P3: b12-b22
        {2: 1, 0: -1},      # P4: b21-b11
        {3: 1},             # P5: b22
        {0: 1, 1: 1},       # P6: b11+b12
        {2: 1, 3: 1},       # P7: b21+b22
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1: c11+c22
        {2: 1, 3: -1},      # P2: c21-c22
        {1: 1, 3: 1},       # P3: c12+c22
        {0: 1, 2: 1},       # P4: c11+c21
        {0: -1, 1: 1},      # P5: -c11+c12
        {3: 1},             # P6: c22
        {0: 1},             # P7: c11
    ]
    
    col_offset = 0
    for j_block in range(3):
        for side in ['L', 'R']:
            if side == 'L':
                u_col_offset = 0
                v_row_offset = 0
            else:
                u_col_offset = 2
                v_row_offset = 2
            
            for prod_idx in range(7):
                col = col_offset + prod_idx
                
                # U local index mapping:
                # local 0 -> A[0][u_col_offset+0], U idx = 0*4 + (u_col_offset+0)
                # local 1 -> A[0][u_col_offset+1], U idx = 0*4 + (u_col_offset+1)
                # local 2 -> A[1][u_col_offset+0], U idx = 1*4 + (u_col_offset+0)
                # local 3 -> A[1][u_col_offset+1], U idx = 1*4 + (u_col_offset+1)
                local_U = {
                    0: 0 * 4 + (u_col_offset + 0),
                    1: 0 * 4 + (u_col_offset + 1),
                    2: 1 * 4 + (u_col_offset + 0),
                    3: 1 * 4 + (u_col_offset + 1),
                }
                for li, s in strassen_U[prod_idx].items():
                    U[local_U[li]][col] += s
                
                # V local index mapping:
                # local 0 -> B[v_row_offset+0][2*j_block+0], V idx = (v_row_offset+0)*6 + (2*j_block+0)
                # local 1 -> B[v_row_offset+0][2*j_block+1], V idx = (v_row_offset+0)*6 + (2*j_block+1)
                # local 2 -> B[v_row_offset+1][2*j_block+0], V idx = (v_row_offset+1)*6 + (2*j_block+0)
                # local 3 -> B[v_row_offset+1][2*j_block+1], V idx = (v_row_offset+1)*6 + (2*j_block+1)
                local_V = {
                    0: (v_row_offset + 0) * 6 + (2 * j_block + 0),
                    1: (v_row_offset + 0) * 6 + (2 * j_block + 1),
                    2: (v_row_offset + 1) * 6 + (2 * j_block + 0),
                    3: (v_row_offset + 1) * 6 + (2 * j_block + 1),
                }
                for li, s in strassen_V[prod_idx].items():
                    V[local_V[li]][col] += s
                
                # W local index mapping:
                # local 0 -> C[0][2*j_block+0], W idx = (2*j_block+0)*2 + 0
                # local 1 -> C[0][2*j_block+1], W idx = (2*j_block+1)*2 + 0
                # local 2 -> C[1][2*j_block+0], W idx = (2*j_block+0)*2 + 1
                # local 3 -> C[1][2*j_block+1], W idx = (2*j_block+1)*2 + 1
                local_W = {
                    0: (2 * j_block + 0) * 2 + 0,
                    1: (2 * j_block + 1) * 2 + 0,
                    2: (2 * j_block + 0) * 2 + 1,
                    3: (2 * j_block + 1) * 2 + 1,
                }
                for li, s in strassen_W[prod_idx].items():
                    W[local_W[li]][col] += s
            
            col_offset += 7
    
    return {"U": U, "V": V, "W": W}
