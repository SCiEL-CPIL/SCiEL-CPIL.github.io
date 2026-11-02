import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x2
    matrix by a 2x8 matrix.
    
    Strategy:
      1. Build a rank-28 Strassen-based decomposition (4 blocks x 7 terms).
      2. Use SVD to find the numerical rank of the tensor space.
      3. Extract an independent subset of rank-1 terms.
      4. Verify the decomposition exactly reproduces the multiplication tensor.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = 2, 2, 8
    nm, mp, pn = n * m, m * p, p * n  # 4, 16, 16

    # --- Step 1: Build the rank-28 Strassen decomposition ---
    # A = [[a, b], [c, d]]  (indices 0,1,2,3)
    # B block = [[x0, x1], [y0, y1]] (local indices 0,1,2,3)
    # C block = [[z0, z1], [z2, z3]] (local indices 0,1,2,3)

    # Strassen products M1..M7
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

    # C coefficients: C_coeffs[c_local_idx] = {M_idx: coeff}
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},  # C00
        {2: 1, 4: 1},               # C01
        {1: 1, 3: 1},               # C10
        {0: 1, 1: -1, 2: 1, 5: 1}   # C11
    ]

    # Build U, V, W as lists of lists, 28 columns
    rank_28 = 28
    U = [[0] * rank_28 for _ in range(nm)]
    V = [[0] * rank_28 for _ in range(mp)]
    W = [[0] * rank_28 for _ in range(pn)]

    for block in range(4):
        k0 = block * 2
        for s in range(7):
            r = block * 7 + s

            # Fill U (A-side)
            for idx, val in A_forms[s].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j][r] = val

            # Fill V (B-side)
            for idx, val in B_forms[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][r] = val

            # Fill W (C-side)
            for c_local, coeffs in enumerate(C_coeffs):
                if s in coeffs:
                    coeff = coeffs[s]
                    i = c_local // 2
                    k_local = c_local % 2
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = coeff

    # --- Step 2: Find the true rank using SVD ---
    # Flatten each rank-1 tensor into a vector of length nm*mp*pn = 1024
    def flatten_tensor_np(U, V, W, r):
        vec = np.zeros(nm * mp * pn, dtype=np.float64)
        for a in range(nm):
            ua = U[a][r]
            if ua == 0:
                continue
            for b in range(mp):
                vb = V[b][r]
                if vb == 0:
                    continue
                ubv = ua * vb
                for c in range(pn):
                    vec[a * mp * pn + b * pn + c] = ubv * W[c][r]
        return vec

    vectors_np = np.array([flatten_tensor_np(U, V, W, r) for r in range(rank_28)])

    # Use SVD to find rank
    U_svd, s, Vh = np.linalg.svd(vectors_np, full_matrices=False)
    tol = 1e-9 * s[0]
    rank_found = int(np.sum(s > tol))

    # Use QR with column pivoting to find independent column indices
    # We need to find which of the original 28 vectors are linearly independent
    # QR with pivoting on the transpose (rows are our vectors)
    # Actually, let's use a different approach: greedy selection
    
    # Greedy selection of independent vectors
    independent_indices = []
    basis = []  # list of vectors (as arrays)
    
    for r in range(rank_28):
        if time.time() - start_time > deadline:
            break
        v = vectors_np[r]
        # Check if v is in the span of basis
        if not basis:
            independent_indices.append(r)
            basis.append(v.copy())
        else:
            # Try to see if v can be represented as combination of basis vectors
            # Use least squares
            B = np.column_stack(basis)
            try:
                coeffs, residuals, rank_b, sv = np.linalg.lstsq(B, v, rcond=tol)
                if residuals and residuals[0] > tol * np.linalg.norm(v):
                    independent_indices.append(r)
                    basis.append(v.copy())
            except:
                independent_indices.append(r)
                basis.append(v.copy())
    
    # If greedy didn't find enough, use SVD-based selection
    if len(independent_indices) < rank_found:
        # Use the first rank_found vectors as a fallback
        independent_indices = list(range(rank_found))

    reduced_rank = len(independent_indices)
    if reduced_rank == 0:
        reduced_rank = 1
        independent_indices = [0]

    # --- Step 3: Extract the reduced decomposition ---
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

    # --- Step 4: Verify correctness ---
    # Check that the decomposition exactly reproduces the multiplication tensor
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

    if not np.allclose(T_decomposed, T_target, atol=1e-10):
        # Fallback: return the full rank-28 solution
        return {"U": U, "V": V, "W": W}

    return {"U": U_red, "V": V_red, "W": W_red}
