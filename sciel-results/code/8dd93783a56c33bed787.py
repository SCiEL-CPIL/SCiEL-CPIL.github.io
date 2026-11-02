import time
import numpy as np
from fractions import Fraction
from itertools import combinations

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x2
    matrix by a 2x8 matrix.

    Strategy:
      1. Build a rank-28 Strassen-based decomposition (4 blocks x 7 terms).
      2. Use the C2xC2 group algebra structure to search for linear
         dependencies among the 28 rank-1 tensors that arise from
         character orthogonality.
      3. Use exact rational Gaussian elimination to verify and extract
         a minimal independent subset.
      4. If a rank-24 decomposition is found, verify it exactly.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = 2, 2, 8
    nm, mp, pn = n * m, m * p, p * n  # 4, 16, 16

    # --- Step 1: Build the rank-28 Strassen decomposition ---
    # A = [[a, b], [c, d]]  (indices 0,1,2,3 in row-major: a=0, b=1, c=2, d=3)
    # B block k: [[x0, x1], [y0, y1]] (local indices 0,1,2,3)
    # C block k: [[z0, z1], [z2, z3]] (local indices 0,1,2,3)

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
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
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

    # --- Helper: verify a decomposition ---
    def verify_decomposition(U_d, V_d, W_d, rank):
        """Check that the decomposition exactly reproduces the multiplication tensor."""
        T_target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0

        T_decomposed = np.zeros((nm, mp, pn))
        for r in range(rank):
            for a in range(nm):
                ua = U_d[a][r]
                if ua == 0:
                    continue
                for b in range(mp):
                    vb = V_d[b][r]
                    if vb == 0:
                        continue
                    ubv = ua * vb
                    for c in range(pn):
                        T_decomposed[a, b, c] += ubv * W_d[c][r]

        return np.allclose(T_decomposed, T_target, atol=1e-12)

    # --- Helper: flatten a rank-1 term into a vector ---
    def flatten_term(U_mat, V_mat, W_mat, r):
        """Flatten the r-th rank-1 term into a 1D list of Fractions."""
        vec = []
        for a in range(nm):
            ua = Fraction(U_mat[a][r])
            if ua == 0:
                vec.extend([Fraction(0)] * (mp * pn))
                continue
            for b in range(mp):
                vb = Fraction(V_mat[b][r])
                if vb == 0:
                    vec.extend([Fraction(0)] * pn)
                    continue
                ubv = ua * vb
                for c in range(pn):
                    vec.append(ubv * Fraction(W_mat[c][r]))
        return vec

    # --- Step 2: Search for rank-24 using group algebra structure ---
    # The 2x8 matrix B can be viewed as 4 blocks of 2x2, indexed by elements
    # of C2xC2 = {(0,0), (0,1), (1,0), (1,1)}.
    # The 28 Strassen terms have structure: for each block g in C2xC2 and
    # each Strassen step s in {0,...,6}, we have a term (g, s).
    # 
    # Key insight: The A-side is shared across all blocks. The B-side and
    # C-side depend on the block. We can look for linear dependencies of
    # the form: sum_{g,s} alpha_{g,s} * T_{g,s} = 0
    # where T_{g,s} is the rank-1 tensor for block g, step s.
    #
    # Due to the group structure, we can look for dependencies where
    # coefficients factor as beta_g * gamma_s or similar.
    #
    # Approach: Try all subsets of 24 terms (remove 4 terms) and check
    # if the remaining 24 are linearly independent AND span the same
    # tensor. This is expensive, so we use structure to prune.
    
    # First, let's try a more targeted approach: look for 4 terms that
    # can be removed such that the remaining 24 still span the tensor.
    # We'll check if any 4 terms are linearly dependent with the others.
    
    # Build the 28 flattened vectors
    vectors = []
    for r in range(rank_28):
        if time.time() - start_time > deadline:
            break
        vectors.append(flatten_term(U, V, W, r))
    
    if len(vectors) < rank_28:
        # Timeout during vector construction, fall back to Gaussian elimination
        independent_indices = []
        basis = []
        for r in range(len(vectors)):
            v = vectors[r][:]
            dim = len(v)
            for b_idx, b_vec in basis:
                pivot_pos = -1
                for pos in range(dim):
                    if b_vec[pos] != 0:
                        pivot_pos = pos
                        break
                if pivot_pos < 0:
                    continue
                if v[pivot_pos] != 0:
                    factor = v[pivot_pos] / b_vec[pivot_pos]
                    for pos in range(pivot_pos, dim):
                        v[pos] -= factor * b_vec[pos]
            is_zero = True
            for val in v:
                if val != 0:
                    is_zero = False
                    break
            if not is_zero:
                independent_indices.append(r)
                basis.append((r, v))
        
        reduced_rank = len(independent_indices)
        if reduced_rank == 0:
            return {"U": U, "V": V, "W": W}
        
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
        
        if verify_decomposition(U_red, V_red, W_red, reduced_rank):
            return {"U": U_red, "V": V_red, "W": W_red}
        return {"U": U, "V": V, "W": W}
    
    # --- Step 3: Try to find a rank-24 decomposition ---
    # We need to find 4 indices to remove such that the remaining 24
    # vectors still span the same space as all 28.
    # 
    # This is equivalent to finding 4 vectors that are in the span of
    # the other 24. We can check this by: for each candidate set of 4
    # to remove, check if each of those 4 is in the span of the remaining 24.
    #
    # To make this tractable, we use the group structure. The terms are
    # indexed by (block, strassen_step) where block in {0,1,2,3} and
    # strassen_step in {0,...,6}.
    #
    # Hypothesis: There may be dependencies that respect the block structure.
    # For example, if we remove one term from each block, the dependency
    # might factor as a product of a block-dependent coefficient and a
    # step-dependent coefficient.
    
    # Try removing one term from each block (4 terms total)
    # There are 7^4 = 2401 such combinations. For each, check if the
    # 4 removed terms are in the span of the remaining 24.
    
    best_rank = 28
    best_indices = None
    
    for s0 in range(7):
        if time.time() - start_time > deadline:
            break
        for s1 in range(7):
            if time.time() - start_time > deadline:
                break
            for s2 in range(7):
                if time.time() - start_time > deadline:
                    break
                for s3 in range(7):
                    if time.time() - start_time > deadline:
                        break
                    
                    # Terms to remove: (0,s0), (1,s1), (2,s2), (3,s3)
                    remove = [s0, 7+s1, 14+s2, 21+s3]
                    keep = [r for r in range(28) if r not in remove]
                    
                    # Check if the 28 vectors span the same space as the 24 kept vectors
                    # This is equivalent to: each removed vector is in the span of the kept vectors
                    # We can check this by doing Gaussian elimination on all 28 and seeing
                    # if the removed ones become zero when reduced by the kept ones.
                    
                    # Build a combined matrix: first the 24 kept vectors, then the 4 removed
                    # If the rank of all 28 equals the rank of the 24 kept, then we can remove them.
                    
                    # To save time, we'll do a quick check: compute the rank of the 24 kept
                    # and the rank of all 28. If they're equal, we found a valid removal.
                    
                    # This is expensive to do naively. Instead, we'll check if the 4 removed
                    # vectors are linearly dependent with the 24 kept by checking if
                    # the 4x4 matrix of "coordinates" has determinant 0.
                    
                    # Alternative: Just do Gaussian elimination on all 28, but process
                    # the kept ones first, then see if the removed ones reduce to zero.
                    
                    # For efficiency, let's just try a subset of these combinations.
                    # We'll prioritize combinations where the removed terms have some
                    # symmetry.
                    
                    # Actually, let's just do the full Gaussian elimination check
                    # for each combination, but break early if we find one that works.
                    
                    # Process kept vectors first
                    basis = []
                    for r in keep:
                        v = vectors[r][:]
                        dim = len(v)
                        for b_idx, b_vec in basis:
                            pivot_pos = -1
                            for pos in range(dim):
                                if b_vec[pos] != 0:
                                    pivot_pos = pos
                                    break
                            if pivot_pos < 0:
                                continue
                            if v[pivot_pos] != 0:
                                factor = v[pivot_pos] / b_vec[pivot_pos]
                                for pos in range(pivot_pos, dim):
                                    v[pos] -= factor * b_vec[pos]
                        is_zero = True
                        for val in v:
                            if val != 0:
                                is_zero = False
                                break
                        if not is_zero:
                            basis.append((r, v))
                    
                    # Now check if each removed vector reduces to zero
                    all_dependent = True
                    for r in remove:
                        v = vectors[r][:]
                        dim = len(v)
                        for b_idx, b_vec in basis:
                            pivot_pos = -1
                            for pos in range(dim):
                                if b_vec[pos] != 0:
                                    pivot_pos = pos
                                    break
                            if pivot_pos < 0:
                                continue
                            if v[pivot_pos] != 0:
                                factor = v[pivot_pos] / b_vec[pivot_pos]
                                for pos in range(pivot_pos, dim):
                                    v[pos] -= factor * b_vec[pos]
                        for val in v:
                            if val != 0:
                                all_dependent = False
                                break
                        if not all_dependent:
                            break
                    
                    if all_dependent:
                        best_rank = 24
                        best_indices = keep
                        break
                
                if best_rank < 28:
                    break
            if best_rank < 28:
                break
        if best_rank < 28:
            break
    
    # --- Step 4: If no rank-24 found, use standard Gaussian elimination ---
    if best_rank == 28:
        independent_indices = []
        basis = []
        for r in range(rank_28):
            if time.time() - start_time > deadline:
                break
            v = vectors[r][:]
            dim = len(v)
            for b_idx, b_vec in basis:
                pivot_pos = -1
                for pos in range(dim):
                    if b_vec[pos] != 0:
                        pivot_pos = pos
                        break
                if pivot_pos < 0:
                    continue
                if v[pivot_pos] != 0:
                    factor = v[pivot_pos] / b_vec[pivot_pos]
                    for pos in range(pivot_pos, dim):
                        v[pos] -= factor * b_vec[pos]
            is_zero = True
            for val in v:
                if val != 0:
                    is_zero = False
                    break
            if not is_zero:
                independent_indices.append(r)
                basis.append((r, v))
        
        reduced_rank = len(independent_indices)
        if reduced_rank == 0:
            return {"U": U, "V": V, "W": W}
        
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
        
        if verify_decomposition(U_red, V_red, W_red, reduced_rank):
            return {"U": U_red, "V": V_red, "W": W_red}
        return {"U": U, "V": V, "W": W}
    
    # --- Step 5: Extract the rank-24 decomposition ---
    reduced_rank = best_rank
    U_red = [[0] * reduced_rank for _ in range(nm)]
    V_red = [[0] * reduced_rank for _ in range(mp)]
    W_red = [[0] * reduced_rank for _ in range(pn)]
    for new_r, old_r in enumerate(best_indices):
        for a in range(nm):
            U_red[a][new_r] = U[a][old_r]
        for b in range(mp):
            V_red[b][new_r] = V[b][old_r]
        for c in range(pn):
            W_red[c][new_r] = W[c][old_r]
    
    if verify_decomposition(U_red, V_red, W_red, reduced_rank):
        return {"U": U_red, "V": V_red, "W": W_red}
    
    # Fallback
    return {"U": U, "V": V, "W": W}
