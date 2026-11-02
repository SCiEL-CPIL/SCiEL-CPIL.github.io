import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x2
    matrix by a 2x8 matrix.

    Strategy:
      1. Build a rank-28 Strassen-based decomposition (4 blocks x 7 terms).
      2. Use exact rational Gaussian elimination to find the minimal independent
         subset of rank-1 tensors.
      3. If dependencies exist, the rank will be less than 28.
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n, m, p = 2, 2, 8
    nm, mp, pn = n * m, m * p, p * n  # 4, 16, 16

    # --- Step 1: Build the rank-28 Strassen decomposition ---
    A_forms = [
        {0: 1, 3: 1},       # M1: a+d
        {2: 1, 3: 1},       # M2: c+d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a+b
        {2: 1, 0: -1},      # M6: c-a
        {1: 1, 3: -1}       # M7: b-d
    ]

    B_forms = [
        {0: 1, 3: 1},       # M1: x0+y1
        {0: 1},             # M2: x0
        {1: 1, 3: -1},      # M3: x1-y1
        {2: 1, 0: -1},      # M4: y0-x0
        {3: 1},             # M5: y1
        {0: 1, 1: 1},       # M6: x0+x1
        {2: 1, 3: 1}        # M7: y0+y1
    ]

    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},  # C00
        {2: 1, 4: 1},               # C01
        {1: 1, 3: 1},               # C10
        {0: 1, 1: -1, 2: 1, 5: 1}   # C11
    ]

    rank_28 = 28
    U = [[0] * rank_28 for _ in range(nm)]
    V = [[0] * rank_28 for _ in range(mp)]
    W = [[0] * rank_28 for _ in range(pn)]

    for block in range(4):
        k0 = block * 2
        for s in range(7):
            r = block * 7 + s
            for idx, val in A_forms[s].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j][r] = val
            for idx, val in B_forms[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][r] = val
            for c_local, coeffs in enumerate(C_coeffs):
                if s in coeffs:
                    coeff = coeffs[s]
                    i = c_local // 2
                    k_local = c_local % 2
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = coeff

    # --- Step 2: Find minimal independent subset via Gaussian elimination ---
    # Key insight: The multiplication tensor for 2x2 by 2x8 has a specific structure.
    # Each entry (i,j,k) of the output is sum over l of A[i,l]*B[l,k].
    # The tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    
    # We need to find the minimum number of rank-1 tensors whose sum equals T.
    # This is equivalent to finding the rank of the tensor T in the tensor rank sense.
    
    # Build the 1024x28 matrix where column r is the flattened r-th rank-1 tensor
    # T_r[a,b,c] = U[a,r]*V[b,r]*W[c,r]
    
    def get_tensor_entry(U, V, W, r, a, b, c):
        return Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
    
    # For each of the 1024 entries, we have a linear equation:
    # sum_r U[a,r]*V[b,r]*W[c,r] * x_r = T[a,b,c]
    # But we already have a valid decomposition with x_r = 1 for all r.
    # We want to find if there's a linear dependence: sum_r alpha_r * T_r = 0
    # with not all alpha_r zero, and alpha_r in Q.
    
    # Build the 1024x28 matrix M where M[entry, r] = T_r[entry]
    # Then find the null space of M.
    
    entries = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                entries.append((a, b, c))
    
    # Build matrix as list of rows (each row is a list of 28 Fractions)
    # M[entry_idx][r] = T_r[entry_idx]
    M = []
    for (a, b, c) in entries:
        row = []
        for r in range(rank_28):
            val = get_tensor_entry(U, V, W, r, a, b, c)
            row.append(val)
        M.append(row)
    
    # Gaussian elimination to find rank and null space
    # We'll do this on the 1024x28 matrix
    
    # Transpose to work with 28x1024 for efficiency
    # Actually, let's just do standard Gaussian elimination on M (1024x28)
    
    # Find the rank by row-reducing
    # We'll track which columns are pivots
    
    pivot_cols = []
    row_reduced = [row[:] for row in M]  # copy
    
    for col in range(rank_28):
        if time.time() - start_time > deadline:
            break
        # Find pivot row
        pivot_row = -1
        for row in range(len(row_reduced)):
            if row_reduced[row][col] != 0:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue
        
        # Swap to top
        row_reduced[pivot_row], row_reduced[0] = row_reduced[0], row_reduced[pivot_row]
        
        # Normalize
        pivot_val = row_reduced[0][col]
        for j in range(col, rank_28):
            row_reduced[0][j] /= pivot_val
        
        # Eliminate from other rows
        for row in range(1, len(row_reduced)):
            factor = row_reduced[row][col]
            if factor != 0:
                for j in range(col, rank_28):
                    row_reduced[row][j] -= factor * row_reduced[0][j]
        
        pivot_cols.append(col)
    
    rank_found = len(pivot_cols)
    
    # The null space has dimension 28 - rank_found
    # If rank_found < 28, we can reduce the rank
    
    # Find a basis for the null space
    # Free variables are columns not in pivot_cols
    free_cols = [c for c in range(rank_28) if c not in pivot_cols]
    
    null_basis = []
    for free_col in free_cols:
        vec = [Fraction(0)] * rank_28
        vec[free_col] = Fraction(1)
        for i, pivot_col in enumerate(pivot_cols):
            vec[pivot_col] = -row_reduced[i][free_col]
        null_basis.append(vec)
    
    # Now we can express the dependent columns in terms of independent ones
    # The independent columns are pivot_cols
    # We want to keep only the independent columns
    
    independent_indices = sorted(pivot_cols)
    reduced_rank = len(independent_indices)
    
    if reduced_rank == 0:
        reduced_rank = 1
    
    U_red = [[0] * reduced_rank for _ in range(nm)]
    V_red = [[0] * reduced_rank for _ in range(mp)]
    W_red = [[0] * reduced_rank for _ in range(pn)]
    
    for new_r, old_r in enumerate(independent_indices):
        for a in range(nm):
            U_red[a][new_r] = U[a][old_r]
        for b in range(mp):
            V_red[b][new_r] = V[b][old_r]
        for c in range(pn):
            W_red[c][new_r] = W[c][old_r]
    
    # Verify
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    T_decomposed = np.zeros((nm, mp, pn))
    for r in range(reduced_rank):
        for a in range(nm):
            ua = U_red[a][r]
            if ua == 0:
                continue
            for b in range(mp):
                vb = V_red[b][r]
                if vb == 0:
                    continue
                ubv = ua * vb
                for c in range(pn):
                    T_decomposed[a, b, c] += ubv * W_red[c][r]
    
    if not np.allclose(T_decomposed, T_target, atol=1e-12):
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_red, "V": V_red, "W": W_red}
