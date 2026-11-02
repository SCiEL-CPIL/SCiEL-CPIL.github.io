"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49) with rank reduction."""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Strassen's 7 products for 2x2 matrices.
# For A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B:
# C00 = a*e + c*h
# C01 = b*h
# C10 = a*f + d*g
# C11 = b*g + d*h
#
# Strassen products:
# P1 = (a+d)(e+h)
# P2 = (c+d)e
# P3 = a(f-h)
# P4 = d(g-e)
# P5 = (a+b)h
# P6 = (c-a)(e+f)
# P7 = (b-d)(g+h)
#
# C00 = P1 + P4 - P5 + P7
# C01 = P3 + P5
# C10 = P2 + P4
# C11 = P1 - P2 + P3 + P6

# Each Strassen product is a linear combination of entries of A and B.
# We represent each product as:
#   (list of (ai, aj, coeff) for A-part, list of (bk, bl, coeff) for B-part)
# where ai,aj index into the 2x2 A block and bk,bl index into the 2x2 B block.

# Strassen products for a 2x2 block:
# A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
# B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h

_STRASSEN_2X2 = [
    # P1 = (a+d)(e+h)
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # P2 = (c+d)e
    ([(1,0,1),(1,1,1)], [(0,0,1)]),
    # P3 = a(f-h)
    ([(0,0,1)], [(0,1,1),(1,1,-1)]),
    # P4 = d(g-e)
    ([(1,1,1)], [(1,0,1),(0,0,-1)]),
    # P5 = (a+b)h
    ([(0,0,1),(0,1,1)], [(1,1,1)]),
    # P6 = (c-a)(e+f)
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
    # P7 = (b-d)(g+h)
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
]

# How each C entry is formed from the 7 Strassen products:
# C00 = P1 + P4 - P5 + P7
# C01 = P3 + P5
# C10 = P2 + P4
# C11 = P1 - P2 + P3 + P6

_STRASSEN_C_COMBINATION = [
    # (c_row, c_col, list of (product_idx, coeff))
    (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
    (0, 1, [(2, 1), (4, 1)]),
    (1, 0, [(1, 1), (3, 1)]),
    (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
]


def _strassen_4x4_decomposition():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication using
    recursive Strassen: apply Strassen at the 4x4 level (7 block products),
    then apply Strassen within each 2x2 sub-block product (7 scalar mults each).
    Total rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49

    # Initialize U, V, W as numpy arrays for faster construction, then convert to lists
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # The 7 block-level Strassen products for 4x4:
    # Each is (A_block_combo, B_block_combo) where combo is list of (br, bc, coeff)
    # br, bc are block row/col indices (0 or 1)
    block_products = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),       # M1
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),                   # M2
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),                  # M3
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),                  # M4
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),                   # M5
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),       # M6
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),       # M7
    ]

    # C block combination: C[cbr][cbc] = sum_i w_i * Mi
    # c_block_combination[cbr][cbc] = list of (product_idx, coeff)
    c_block_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    # Precompute: for each block product i, w_i(cbr,cbc) for each C-block
    # w_block[i][cbr][cbc] = coefficient
    w_block = [[[0]*4 for _ in range(2)] for _ in range(7)]
    for (cbr, cbc, c_list) in c_block_combination:
        for (pi, wi) in c_list:
            w_block[pi][cbr][cbc] = wi

    # Precompute: for each Strassen product s, v_s(lr,lc) for each C-local entry
    # v_local[s][lr][lc] = coefficient
    v_local = [[[0]*2 for _ in range(2)] for _ in range(7)]
    for (lr, lc, v_list) in _STRASSEN_C_COMBINATION:
        for (si, vs) in v_list:
            v_local[si][lr][lc] = vs

    # Build U, V, W
    for i in range(7):
        a_combo, b_combo = block_products[i]
        for s in range(7):
            r = i * 7 + s
            a_local, b_local = _STRASSEN_2X2[s]

            # U: for each (a_br, a_bc, a_coeff) in a_combo and (lr, lc, l_coeff) in a_local
            # U index: (global_row, global_col) where global_row = 2*a_br + lr, global_col = 2*a_bc + lc
            # U is indexed by [row * m + col]
            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col, r] += a_coeff * l_coeff

            # V: for each (b_br, b_bc, b_coeff) in b_combo and (lr, lc, l_coeff) in b_local
            # V index: [row * p + col]
            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col, r] += b_coeff * l_coeff

            # W: C[2*cbr+lr][2*cbc+lc] gets w_block[i][cbr][cbc] * v_local[s][lr][lc]
            # W index: [col * n + row]
            for cbr in range(2):
                for cbc in range(2):
                    wi = w_block[i][cbr][cbc]
                    if wi == 0:
                        continue
                    for lr in range(2):
                        for lc in range(2):
                            vs = v_local[s][lr][lc]
                            if vs == 0:
                                continue
                            c_row = 2 * cbr + lr
                            c_col = 2 * cbc + lc
                            W[c_col * n + c_row, r] += wi * vs

    # Convert back to lists of lists to match the expected interface format if strict
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy for speed."""
    U_arr = np.array(U, dtype=np.float64)  # (n*m, rank)
    V_arr = np.array(V, dtype=np.float64)  # (m*p, rank)
    W_arr = np.array(W, dtype=np.float64)  # (p*n, rank)
    
    # Reshape to 3D: U_3d[i,j,r], V_3d[j,k,r], W_3d[k,i,r]
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Contract over r: result[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # Note: W is indexed by [k*n+i], so W_3d[k,i,r] corresponds to W[k*n+i, r]
    result = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d)
    
    # All entries should be 1
    return bool(np.allclose(result, 1.0, atol=1e-10))


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _rank_reduce(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce rank by finding linearly dependent bilinear terms.
    Each term r contributes U[:,r] ⊗ V[:,r] ⊗ W[:,r] to the tensor.
    Two terms r1, r2 are linearly dependent if U[:,r1]⊗V[:,r1]⊗W[:,r1] is a scalar
    multiple of U[:,r2]⊗V[:,r2]⊗W[:,r2].
    
    More generally, we can try to find a set of terms where one is a linear combination
    of others. We use the fact that the tensor T = sum_r u_r ⊗ v_r ⊗ w_r.
    
    A simpler approach: check if any column of U (or V, or W) is a linear combination
    of others, which would indicate redundancy. But that's not quite right for tensor rank.
    
    The correct approach: the rank is the minimum number of terms. We can try to find
    linear dependencies among the "bilinear forms." 
    
    For 4x4x4, the tensor space has dimension 64. Each rank-1 term u⊗v⊗w spans a 1D
    subspace. We need to find if any of our 49 terms is in the span of the others.
    
    We represent each term as a 64D vector (flattened tensor). If one vector is in the
    span of the others, we can remove it.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)  # (n*m, rank)
    V_arr = np.array(V, dtype=np.float64)  # (m*p, rank)
    W_arr = np.array(W, dtype=np.float64)  # (p*n, rank)
    
    # Build the 64D vector for each term
    # Term r: U[:,r] (16D) ⊗ V[:,r] (16D) ⊗ W[:,r] (16D) flattened to 4096D? No.
    # Actually the tensor is 4x4x4 = 64D.
    # U[:,r] is indexed by (i,j), V[:,r] by (j,k), W[:,r] by (k,i)
    # The tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So each term r contributes to T[i,j,k]: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a rank-1 tensor in the 4x4x4 space.
    
    # To check linear dependence, we can try to express one term's contribution
    # as a linear combination of others. But that's expensive.
    
    # Alternative: use the fact that if we have rank r, the tensor T must equal
    # sum_r u_r ⊗ v_r ⊗ w_r. If we can find coefficients c_r such that
    # sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 and not all c_r are zero, then the terms
    # are linearly dependent and we can remove one.
    
    # This is equivalent to checking if the 49 vectors in the tensor product space
    # are linearly independent. The tensor product space has dimension 64 (for 4x4x4).
    # Wait, no. The space of all 4x4x4 tensors has dimension 64. Each rank-1 term
    # is a vector in this 64D space. So we have 49 vectors in a 64D space.
    # They can be linearly dependent!
    
    # Let's build the 64D vector for each term and check for linear dependence.
    n, m, p = 4, 4, 4
    tensor_dim = n * m * p  # 64
    
    # Build matrix M where column r is the 64D vector for term r
    M = np.zeros((tensor_dim, rank), dtype=np.float64)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    M[idx, r] = U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
    
    # Now find the rank of M. If rank(M) < 49, there are linear dependencies.
    # Use SVD to find the null space.
    try:
        U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
    except:
        return None
    
    # Find the numerical rank
    tol = 1e-10
    num_rank = np.sum(s > tol)
    
    if num_rank >= rank:
        # All terms are linearly independent, can't reduce
        return None
    
    # The null space is spanned by the last (rank - num_rank) rows of Vt_svd
    null_space = Vt_svd[num_rank:, :].T  # (rank, nullity)
    nullity = rank - num_rank
    
    # For each null vector, find a term with significant coefficient that we can remove
    # We want to find a set of terms to remove such that the remaining terms still span
    # the tensor.
    
    # Greedy approach: try to remove one term at a time
    # For each null vector c, if c[r] != 0 for some r, we can potentially remove term r
    # by adjusting the other terms.
    
    # Actually, a cleaner approach: find the maximum set of linearly independent columns
    # and remove the rest.
    
    # Use the SVD to identify which columns are in the null space
    # A column r is "redundant" if it can be expressed as a combination of others.
    
    # Let's try a greedy elimination:
    # Keep a set of "kept" terms. Start with all terms.
    # For each term, check if it's in the span of the kept terms. If yes, remove it.
    
    kept = list(range(rank))
    removed = []
    
    # Sort terms by some heuristic - maybe by norm
    norms = np.linalg.norm(M, axis=0)
    order = np.argsort(norms)  # smallest first
    
    # Try to remove small-norm terms first
    for r in order:
        if time.time() > deadline:
            break
        if r not in kept:
            continue
        # Check if column r is in the span of the other kept columns
        others = [k for k in kept if k != r]
        if not others:
            break
        # Solve M[:, others] * x = M[:, r]
        try:
            x, residuals, rank_m, s_vals = np.linalg.lstsq(M[:, others], M[:, r], rcond=None)
            residual = np.linalg.norm(M[:, r] - M[:, others] @ x)
            if residual < 1e-8:
                kept.remove(r)
                removed.append(r)
        except:
            pass
    
    if not removed:
        return None
    
    # Build reduced decomposition
    new_rank = len(kept)
    U_new = U_arr[:, kept].tolist()
    V_new = V_arr[:, kept].tolist()
    W_new = W_arr[:, kept].tolist()
    
    # Verify
    if _verify_decomposition(U_new, V_new, W_new, n, m, p, new_rank):
        return {"U": U_new, "V": V_new, "W": W_new}
    
    return None


def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS

    # Try the recursive Strassen rank-49 decomposition
    t0 = time.time()
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p, rank):
        # Fallback to schoolbook if verification fails (should not happen)
        return _schoolbook_decomposition(n, m, p)

    # Try to reduce rank
    if time.time() < deadline - 5.0:
        reduced = _rank_reduce(U, V, W, n, m, p, deadline - 3.0)
        if reduced is not None:
            return reduced

    return result

# EVOLVE-BLOCK-END
