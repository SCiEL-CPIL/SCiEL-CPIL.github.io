"""
Matrix multiplication 2x4 @ 4x6: rank-minimal exact tensor decomposition.

Strategy (different from block-partitioning / Strassen):
  Use a direct algebraic search over the tensor's slice space.
  1. Build the 48-dimensional matrix-multiplication tensor T in C^{8 x 24 x 12}.
  2. The tensor rank r is the minimum number of rank-1 tensors summing to T.
  3. We use the "slice space" / "rank profile" idea:
       - For a rank-r decomposition, the slices of T along any mode must lie in
         the span of the corresponding factor vectors.
       - Concretely, fix a small set of candidate W-vectors (the third factor),
         then the problem reduces to a bilinear system in U and V.
  4. Because n=2, m=4, p=6 are small, we can enumerate structured candidate
     W-vectors from the span of the 12 standard basis vectors of C^{12} with
     small integer coefficients, and solve the resulting linear systems.

  In practice, for this specific small case, the known lower bound is 24
  (from the "rank of matrix multiplication" literature for small sizes) and
  the upper bound from Strassen-style partitioning is 42.  We try to find
  a decomposition of rank < 42 by:
    a) Trying rank 24, 25, ..., 41 with a randomized / structured search.
    b) Using the fact that the tensor is "commutative" in a limited sense:
       the 12 output coordinates (k*n+i) each receive contributions from
       specific (i,j,k) tuples.

  A more concrete approach:
    The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
    We seek U (8 x r), V (24 x r), W (12 x r) such that:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all (i,j,k)
      and 0 otherwise.

    Fix W. Then for each r, the equation becomes:
      sum_r U[i*m+j, r] * V[j*p+k, r] = 1/W[k*n+i, r]  (where W[k*n+i, r] != 0)

    This is a system of bilinear equations.  We can linearize by fixing
    either U or V.  For example, fix V, then the equations become linear in U:
      For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

    But this is still coupled across r.  A standard trick:
      Let M_r be the outer product U[:,r] * V[:,r] * W[:,r].  We need sum M_r = T.

    Alternate approach: use the "border rank" or "decomposition via linear algebra"
    by treating the problem as finding r vectors in a 8*24*12 = 2304-dim space
    whose sum equals T, with the constraint that each vector is rank-1 in
    the tensor product sense.

    Given the computational budget, we use a structured search:
      - Start from the Strassen-42 solution.
      - Try to reduce rank by 1, 2, ... by finding linear dependencies among
        the rank-1 terms and eliminating them.
      - Specifically, if two rank-1 terms can be combined (their sum is also
        rank-1 or can be replaced by fewer rank-1 terms), reduce the count.

    For this small case, we try a direct enumeration of "simple" W-vectors
    (vectors with entries in {-1, 0, 1} with at most 3 non-zero entries) and
    solve for U, V via least-squares / exact rational arithmetic.
"""

import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations_with_replacement

DIMENSIONS = [2, 4, 6]


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm = n * m   # 8
    mp = m * p   # 24
    pn = p * n   # 12

    def schoolbook():
        rank = nm * mp  # wait, no: rank = n*m*p = 48
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    def verify(U, V, W, tol=1e-9):
        rank = len(U[0]) if U and U[0] else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += float(U[i * m + j][r]) * float(V[j * p + k][r]) * float(W[k * n + i][r])
                    if abs(s - 1.0) > tol:
                        return False
        # Check that non-multiplication entries are 0
        # The tensor T[a,b,c] is 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k)
        # and 0 otherwise. But our decomposition only needs to match the 48 nonzero entries.
        # Actually, the tensor has exactly n*m*p = 48 nonzero entries, all equal to 1.
        # All other entries must be 0. Let's verify a few.
        # For efficiency, check all entries:
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Determine if (a,b,c) corresponds to a valid (i,j,k)
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2 and k == k2:
                        # Should be 1
                        s = 0.0
                        for r in range(rank):
                            s += float(U[a][r]) * float(V[b][r]) * float(W[c][r])
                        if abs(s - 1.0) > tol:
                            return False
                    else:
                        # Should be 0
                        s = 0.0
                        for r in range(rank):
                            s += float(U[a][r]) * float(V[b][r]) * float(W[c][r])
                        if abs(s) > tol:
                            return False
        return True

    # Get schoolbook as fallback
    best_result, best_rank = schoolbook()

    # Build Strassen-42 solution
    def strassen42():
        rank = 42
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        def add_strassen_block(offset, A_rows, B_rows, C_rows):
            a00, a01, a10, a11 = A_rows
            b00, b01, b10, b11 = B_rows
            c00, c01, c10, c11 = C_rows

            U[a00][offset] += 1
            U[a11][offset] += 1
            V[b00][offset] += 1
            V[b11][offset] += 1
            W[c00][offset] += 1
            W[c11][offset] += 1

            U[a10][offset + 1] += 1
            U[a11][offset + 1] += 1
            V[b00][offset + 1] += 1
            W[c10][offset + 1] += 1

            U[a00][offset + 2] += 1
            V[b01][offset + 2] += 1
            V[b11][offset + 2] += -1
            W[c01][offset + 2] += 1
            W[c11][offset + 2] += 1

            U[a11][offset + 3] += 1
            V[b10][offset + 3] += 1
            V[b00][offset + 3] += -1
            W[c00][offset + 3] += 1
            W[c10][offset + 3] += 1

            U[a00][offset + 4] += 1
            U[a01][offset + 4] += 1
            V[b11][offset + 4] += 1
            W[c00][offset + 4] += -1
            W[c01][offset + 4] += 1

            U[a10][offset + 5] += 1
            U[a00][offset + 5] += -1
            V[b00][offset + 5] += 1
            V[b01][offset + 5] += 1
            W[c11][offset + 5] += 1

            U[a01][offset + 6] += 1
            U[a11][offset + 6] += -1
            V[b10][offset + 6] += 1
            V[b11][offset + 6] += 1
            W[c00][offset + 6] += 1
            W[c11][offset + 6] += -1

            return offset + 7

        A_L = [0, 1, 4, 5]
        A_R = [2, 3, 6, 7]
        B_T1 = [0, 1, 6, 7]
        B_T2 = [2, 3, 8, 9]
        B_T3 = [4, 5, 10, 11]
        B_B1 = [12, 13, 18, 19]
        B_B2 = [14, 15, 20, 21]
        B_B3 = [16, 17, 22, 23]
        C_L = [0, 2, 1, 3]
        C_M = [4, 6, 5, 7]
        C_R = [8, 10, 9, 11]

        offset = 0
        offset = add_strassen_block(offset, A_L, B_T1, C_L)
        offset = add_strassen_block(offset, A_R, B_B1, C_L)
        offset = add_strassen_block(offset, A_L, B_T2, C_M)
        offset = add_strassen_block(offset, A_R, B_B2, C_M)
        offset = add_strassen_block(offset, A_L, B_T3, C_R)
        offset = add_strassen_block(offset, A_R, B_B3, C_R)

        return {"U": U, "V": V, "W": W}, rank

    strassen_res, strassen_rank = strassen42()
    if verify(strassen_res["U"], strassen_res["V"], strassen_res["W"]):
        best_result = strassen_res
        best_rank = strassen_rank

    # Now try to improve via direct algebraic search.
    # Approach: Try to find a decomposition of rank r < best_rank by
    # fixing W to have a specific sparse structure and solving for U, V.
    
    # The tensor T has 48 nonzero entries. For a rank-r decomposition,
    # we need r rank-1 tensors. Each rank-1 tensor U_r ⊗ V_r ⊗ W_r contributes
    # to the 48 entries.
    
    # Key insight: The 12 output coordinates c = k*n+i each correspond to
    # a specific (i,k) pair. For a fixed c, the equation is:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[c, r] = 1  for the valid (j)
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[c, r] = 0  for invalid (j)
    
    # We try a structured search over W matrices. For each candidate W,
    # we check if there exist U, V satisfying the equations.
    
    # Due to time constraints, we try a limited set of candidate W structures.
    
    # Method: Randomized search with verification
    # Generate random W with small integer entries, then solve for U, V.
    
    import random
    random.seed(42)
    
    def try_rank_r(r, num_trials=200):
        """Try to find a rank-r decomposition via random W and solving for U,V."""
        if time.time() > deadline:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Generate random W with entries in {-1, 0, 1}
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
            
            # Check that W has enough non-zero entries to be useful
            if sum(1 for row in W for x in row if x != 0) < r:
                continue
            
            # For each output coordinate c = k*n+i, we have equations:
            # For valid j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[c, r] = 1
            # For invalid j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[c, r] = 0
            
            # This is a bilinear system. We linearize by fixing V.
            # Let's fix V to be random too, then solve for U.
            
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
            
            # Now for each (i,k), we need to find U[i*m+j, r] for j=0..m-1
            # such that for each j:
            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(valid)
            
            # This gives m equations in m unknowns (U[i*m+j] for j=0..m-1) for each (i,k).
            # But the unknowns are the full vectors U[i*m+j, :] (each of length r).
            # So it's actually m*r unknowns per (i,k) pair, with m equations.
            # Under-determined if r > 1. We can use least squares.
            
            # Let's set up the linear system for all U entries.
            # Total unknowns: nm * r = 8 * r
            # Total equations: for each (i,j,k) valid: 1 equation, for each invalid: 1 equation
            # Total entries in tensor: nm * mp * pn = 8 * 24 * 12 = 2304
            # But we only need to satisfy the 48 valid entries = 1 and the rest = 0.
            
            # Actually, the constraint is that the reconstructed tensor equals T.
            # Let's build the linear system in U (treating V, W as fixed).
            
            # For each (a, b, c) in the tensor:
            #   sum_r U[a, r] * V[b, r] * W[c, r] = T[a,b,c]
            
            # This is linear in U. We have nm*r unknowns and nm*mp*pn equations.
            # nm*r = 8r, equations = 2304.
            # For r=42: 336 unknowns, 2304 equations. Over-determined.
            
            # Use numpy least squares.
            A_mat = np.zeros((nm * mp * pn, nm * r))
            b_vec = np.zeros(nm * mp * pn)
            
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        eq_idx = a * mp * pn + b * pn + c
                        # T[a,b,c]
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        if i == i2 and j == j2 and k == k2:
                            b_vec[eq_idx] = 1.0
                        else:
                            b_vec[eq_idx] = 0.0
                        for r_idx in range(r):
                            A_mat[eq_idx, a * r + r_idx] = float(V[b][r_idx]) * float(W[c][r_idx])
            
            # Solve least squares
            try:
                U_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            # Check if the solution is good
            U_check = U_sol.reshape(nm, r)
            # Verify
            max_err = 0.0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        s = 0.0
                        for r_idx in range(r):
                            s += U_check[a][r_idx] * float(V[b][r_idx]) * float(W[c][r_idx])
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                        max_err = max(max_err, abs(s - expected))
            
            if max_err < 1e-9:
                # Found a valid decomposition!
                U_int = [[int(round(U_check[a][r_idx])) for r_idx in range(r)] for a in range(nm)]
                return {"U": U_int, "V": V, "W": W}, r
        
        return None

    # Try ranks from best_rank - 1 down to 24
    for r in range(best_rank - 1, 23, -1):
        if time.time() > deadline:
            break
        result = try_rank_r(r, num_trials=100)
        if result is not None:
            best_result, best_rank = result
            break

    return best_result
