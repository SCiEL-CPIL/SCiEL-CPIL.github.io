"""Rank-minimal exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Start from the rank-42 Strassen block decomposition, then attempt to
reduce rank by identifying and eliminating linearly dependent rank-1 terms.
Also try structured searches for lower-rank decompositions.

Partition:
  A (2x4) = [AL | AR], each 2x2
  B (4x6) = [B1 | B2 | B3], each 4x2
  C = A @ B, so C[:, j_block*2:(j_block+1)*2] = AL @ B_j_top + AR @ B_j_bottom

  Six 2x2@2x2 products, each via Strassen => rank 42.

Uses exact integer arithmetic throughout.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def _strassen_factors():
    """Return (U1, V1, W1) for 2x2x2 Strassen, shapes (4,7),(4,7),(4,7).
    U1 row: i*2+j, V1 row: j*2+k, W1 row: k*2+i (trace convention).
    All entries are integers in {-1, 0, 1}."""
    U1 = [[0] * 7 for _ in range(4)]
    V1 = [[0] * 7 for _ in range(4)]
    W1 = [[0] * 7 for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1  # C00 += M1
    W1[3][0] = 1  # C11 += M1

    # M2 = (A10+A11)B00
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1  # C10 += M2
    W1[3][1] = -1 # C11 -= M2

    # M3 = A00(B01-B11)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1  # C01 += M3
    W1[3][2] = 1  # C11 += M3

    # M4 = A11(B10-B00)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1  # C00 += M4
    W1[1][3] = 1  # C10 += M4

    # M5 = (A00+A01)B11
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1 # C00 -= M5
    W1[2][4] = 1  # C01 += M5

    # M6 = (A10-A00)(B00+B01)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1  # C11 += M6

    # M7 = (A01-A11)(B10+B11)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1  # C00 += M7

    return U1, V1, W1


def _build_rank42():
    """Build rank-42 decomposition via 6 Strassen blocks. All integer."""
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 42

    U = [[0] * total_rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * total_rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * total_rank for _ in range(p * n)]  # (12, 42)

    # Six blocks: for j_block in 0..2 (which 4x2 block of B),
    #             for side in 0..1 (AL or AR, i.e., which 2x2 of A)
    for j_block in range(3):
        for side in range(2):
            b = j_block * 2 + side
            col_offset = b * 7

            # U contribution: A[i, a] where a = side * 2 + j_local
            for i in range(2):
                for j_local in range(2):
                    a = side * 2 + j_local
                    for s in range(7):
                        coeff = U1[i * 2 + j_local][s]
                        if coeff != 0:
                            U[i * m + a][col_offset + s] += coeff

            # V contribution: B[a, bcol] where a = side * 2 + j_local, bcol = j_block * 2 + k_local
            for j_local in range(2):
                a = side * 2 + j_local
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = V1[j_local * 2 + k_local][s]
                        if coeff != 0:
                            V[a * p + bcol][col_offset + s] += coeff

            # W contribution: C[i, bcol] where i in 0..1, bcol = j_block * 2 + k_local
            for i in range(2):
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = W1[k_local * 2 + i][s]
                        if coeff != 0:
                            W[bcol * n + i][col_offset + s] += coeff

    return U, V, W


def _verify_exact(U, V, W, n, m, p):
    """Verify with exact integer arithmetic: sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for s in range(rank):
                    val += U[i * m + j][s] * V[j * p + k][s] * W[k * n + i][s]
                if val != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    U, V, W = _build_rank42()

    # Verify correctness with exact integer arithmetic
    if not _verify_exact(U, V, W, n, m, p):
        # Fallback: schoolbook rank 48 (should never be reached)
        schoolbook_rank = n * m * p
        U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj][col] = 1
                    V_fb[jj * p + kk][col] = 1
                    W_fb[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Try to reduce rank by finding linear dependencies
    best_rank = len(U[0])
    best_U, best_V, best_W = U, V, W

    if time.time() < deadline - 20.0:
        reduced = _try_reduce_rank(U, V, W, n, m, p)
        if reduced is not None:
            rU, rV, rW = reduced
            if len(rU[0]) < best_rank and _verify_exact(rU, rV, rW, n, m, p):
                best_rank = len(rU[0])
                best_U, best_V, best_W = rU, rV, rW

    # Try structured search for even lower rank
    if time.time() < deadline - 10.0:
        found = _structured_search(n, m, p, best_rank - 1, deadline)
        if found is not None:
            fU, fV, fW = found
            if _verify_exact(fU, fV, fW, n, m, p):
                best_rank = len(fU[0])
                best_U, best_V, best_W = fU, fV, fW

    return {"U": best_U, "V": best_V, "W": best_W}


def _try_reduce_rank(U, V, W, n, m, p):
    """Try to reduce rank by finding linear dependencies among rank-1 terms.
    
    For each term s, check if u_s ⊗ v_s ⊗ w_s can be expressed as a linear
    combination of other terms. If so, we can potentially eliminate it.
    
    Returns (U', V', W') with reduced rank if successful, else None.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor T as sum of rank-1 terms
    # T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    
    # For each term, try to see if it's redundant
    # A term s is redundant if removing it and adjusting others still gives T
    # This is equivalent to checking if u_s ⊗ v_s ⊗ w_s is in the span of others
    
    # Simpler approach: try removing each term and see if the remaining
    # terms can still decompose T (by solving for new W values)
    
    # Actually, let's try a different approach: find if any column of U, V, W
    # is a linear combination of others in a way that preserves the tensor
    
    # The tensor T is fixed. If we have terms (u_s, v_s, w_s) for s=1..r,
    # and we want to remove term r, we need:
    # u_r ⊗ v_r ⊗ w_r = sum_{s=1}^{r-1} c_s (u_s ⊗ v_s ⊗ w_s)
    # This is a system of equations. Let's check if any term is dependent.
    
    # Flatten each rank-1 tensor
    def flatten_term(u, v, w, n, m, p):
        """Flatten u⊗v⊗w into a vector of length n*m*p."""
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i * m * p + j * p + k] = u[i * m + j] * v[j * p + k] * w[k * n + i]
        return vec
    
    terms = []
    for s in range(rank):
        u_s = U_arr[:, s]
        v_s = V_arr[:, s]
        w_s = W_arr[:, s]
        terms.append(flatten_term(u_s, v_s, w_s, n, m, p))
    
    terms_arr = np.array(terms)
    
    # Check rank of the term matrix
    # If rank < number of terms, there are dependencies
    term_rank = np.linalg.matrix_rank(terms_arr, tol=1e-10)
    
    if term_rank < rank:
        # There are dependencies. Try to find a subset of terms that spans the same space
        # and still sums to T
        T_flat = np.zeros(n * m * p)
        for s in range(rank):
            T_flat += terms_arr[s]
        
        # Find a minimal subset that sums to T
        # This is a combinatorial problem, but for small reductions we can try greedily
        # Try removing each term and checking if the rest can be rescaled to sum to T
        
        # Actually, the simplest check: if term s is a linear combination of others,
        # we can't just remove it - we need the sum to remain T.
        # The sum constraint is: sum_s t_s = T
        # If we remove term r, we need sum_{s≠r} t_s = T - t_r
        # But T - t_r may not be representable with the remaining terms.
        
        # Let's try a different approach: solve for coefficients
        # We want to find c_s such that sum_s c_s * t_s = T and as many c_s = 0 as possible
        
        # This is an underdetermined system. Use least squares with sparsity.
        # For now, just report the rank deficiency and try simple eliminations
        
        # Try removing each term one at a time
        for remove_idx in range(rank):
            remaining = [i for i in range(rank) if i != remove_idx]
            T_remaining = T_flat - terms_arr[remove_idx]
            
            # Check if T_remaining is in the span of remaining terms
            if len(remaining) == 0:
                continue
            
            rem_arr = terms_arr[remaining]
            # Solve rem_arr^T * x = T_remaining (least squares)
            # Actually we need: sum_{i in remaining} x_i * rem_arr[i] = T_remaining
            # This is rem_arr.T @ x = T_remaining
            
            try:
                x, residuals, rank_rem, sv = np.linalg.lstsq(rem_arr.T, T_remaining, rcond=1e-10)
                # Check if the solution is close to all 1s (which would mean no change needed)
                # Or just check if the reconstruction is good
                recon = rem_arr.T @ x
                if np.allclose(recon, T_remaining, atol=1e-8):
                    # We can express T_remaining as a combination of remaining terms
                    # But we need the new coefficients to give integer/half-integer results
                    # For now, this indicates a potential reduction
                    # Build the new decomposition with adjusted W
                    new_U = U_arr[:, remaining]
                    new_V = V_arr[:, remaining]
                    # W needs to be adjusted: new_w_s = x_s * old_w_s? No, that's not right.
                    # The issue is that we can't simply scale W because the tensor
                    # decomposition is not linear in that way.
                    # Actually, if t_s = u_s ⊗ v_s ⊗ w_s, and we want
                    # sum c_s t_s = T, then we can set new_w_s = c_s * w_s
                    # only if c_s is a scalar. Let's check.
                    
                    # The new decomposition would be:
                    # sum_s c_s * (u_s ⊗ v_s ⊗ w_s) = sum_s (u_s ⊗ v_s ⊗ (c_s * w_s))
                    # So new W column s = c_s * old W column s
                    
                    new_W = W_arr[:, remaining] * x[np.newaxis, :]
                    
                    # Check if entries are "nice" (integer or half-integer)
                    # For now, just accept if the values are close to half-integers
                    is_nice = True
                    for mat in [new_U, new_V, new_W]:
                        for val in mat:
                            if abs(val * 2 - round(val * 2)) > 1e-6:
                                is_nice = False
                                break
                        if not is_nice:
                            break
                    
                    if is_nice:
                        # Convert to lists with rounded half-integers
                        def to_list(arr):
                            return [[round(val * 2) / 2 for val in row] for row in arr]
                        
                        new_U_list = to_list(new_U)
                        new_V_list = to_list(new_V)
                        new_W_list = to_list(new_W)
                        
                        return new_U_list, new_V_list, new_W_list
            except Exception:
                continue
    
    return None


def _structured_search(n, m, p, max_rank, deadline):
    """Try structured search for a decomposition with rank < max_rank.
    
    This is a placeholder for a more sophisticated search.
    Returns (U, V, W) if found, else None.
    """
    # For now, this is a stub. A real implementation would use
    # algebraic geometry or numerical optimization to find lower-rank
    # decompositions.
    return None
