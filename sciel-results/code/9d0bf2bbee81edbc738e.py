"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Each factor is (u_vec, v_vec, w_vec) where:
      u_vec: 4-element vector for A (row-major: A00, A01, A10, A11)
      v_vec: 4-element vector for B (row-major: B00, B01, B10, B11)
      w_vec: 4-element vector for C (row-major: C00, C01, C10, C11)
    """
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M3 - M4 + M6
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    factors = []

    # M1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]
    factors.append((u, v, w))

    # M2 = (A10+A11) * B00
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, -1]
    factors.append((u, v, w))

    # M3 = A00 * (B01-B11)
    u = [1, 0, 0, 0]
    v = [0, 1, 0, -1]
    w = [0, 1, 0, 1]
    factors.append((u, v, w))

    # M4 = A11 * (B10-B00)
    u = [0, 0, 0, 1]
    v = [-1, 0, 1, 0]
    w = [1, 0, 1, 0]
    factors.append((u, v, w))

    # M5 = (A00+A01) * B11
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 1, 0, 0]
    factors.append((u, v, w))

    # M6 = (A10-A00) * (B00+B01)
    u = [-1, 0, 1, 0]
    v = [1, 1, 0, 0]
    w = [0, 0, 0, 1]
    factors.append((u, v, w))

    # M7 = (A01-A11) * (B10+B11)
    u = [0, 1, 0, -1]
    v = [0, 0, 1, 1]
    w = [1, 0, 0, 0]
    factors.append((u, v, w))

    return factors

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    
    Strategy: Split the 4x8 matrix B into two 4x4 blocks B1 and B2.
    Compute C1 = A*B1 and C2 = A*B2, each using Strassen (rank 49).
    Total rank = 98.
    
    Each 4x4*4x4 Strassen decomposition uses 7 top-level Strassen products,
    each of which is a 2x2*2x2 multiplication, itself decomposed with 7 scalar
    Strassen products. Total: 7 * 7 = 49 per subproblem.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get the 2x2 Strassen factors
    s2_factors = _strassen_2x2_factors()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # (16, 98)
    V = [[0] * rank for _ in range(m * p)]  # (32, 98)
    W = [[0] * rank for _ in range(p * n)]  # (32, 98)
    
    # For each of the two subproblems (t=0 for B1, t=1 for B2):
    for t in range(2):
        # For each of the 7 top-level Strassen products (s=0..6):
        for s in range(7):
            # For each of the 7 scalar Strassen products (r=0..6):
            for r in range(7):
                col = t * 49 + s * 7 + r
                
                # The top-level Strassen product s involves certain A blocks and B blocks.
                # We need to determine the U and V coefficients at the block level,
                # then combine with the scalar-level coefficients.
                
                # Top-level Strassen for 4x4*4x4 (same structure as 2x2*2x2 but with 2x2 blocks):
                # The 7 products are:
                # s=0: (A11+A22)(B11+B22)
                # s=1: (A21+A22)B11
                # s=2: A11(B12-B22)
                # s=3: A22(B21-B11)
                # s=4: (A11+A12)B22
                # s=5: (A21-A11)(B11+B12)
                # s=6: (A12-A22)(B21+B22)
                
                # For product s, the A blocks involved and their coefficients:
                # Using the same pattern as the 2x2 Strassen U factors:
                s2_u_block = s2_factors[s][0]  # [coeff for A00, A01, A10, A11] in block terms
                s2_v_block = s2_factors[s][1]  # [coeff for B00, B01, B10, B11] in block terms
                
                # For the scalar Strassen product r:
                s2_u_scalar = s2_factors[r][0]  # [coeff for element (0,0), (0,1), (1,0), (1,1)]
                s2_v_scalar = s2_factors[r][1]
                s2_w_scalar = s2_factors[r][2]
                
                # U contributions:
                # A block (bi, bj) where bi=0,1 (row block), bj=0,1 (col block)
                # Block (bi, bj) corresponds to index bi*2+bj in the 4-element vector
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_u_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_u_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j][col] += c_block * c_scalar
                
                # V contributions:
                # B block (bi, bj) for subproblem t
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_v_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_v_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                j = 2 * bi + ei  # B row
                                k = 4 * t + 2 * bj + ej  # B col (offset by 4*t)
                                V[j * p + k][col] += c_block * c_scalar
                
                # W contributions:
                # The output C for subproblem t. The top-level Strassen product s
                # contributes to certain C blocks with certain coefficients.
                # The scalar Strassen product r determines which elements of those
                # C blocks are affected.
                
                # We need to determine which C blocks are affected by product s
                # and with what coefficients. This comes from the W vector of the
                # top-level Strassen factor s, but applied at the block level.
                
                # Actually, the W factor for the top-level Strassen product s tells us
                # which C blocks are affected. But we're using the same s2_factors for
                # both levels. The W vector of s2_factors[s] gives the block-level
                # C coefficients: [C00_coeff, C01_coeff, C10_coeff, C11_coeff].
                
                s2_w_block = s2_factors[s][2]  # [coeff for C00, C01, C10, C11] in block terms
                
                for ci in range(2):
                    for cj in range(2):
                        block_idx = ci * 2 + cj
                        c_block = s2_w_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_w_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                i = 2 * ci + ei  # C row
                                k = 4 * t + 2 * cj + ej  # C col (offset by 4*t)
                                W[k * n + i][col] += c_block * c_scalar
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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

def _reduce_rank(U, V, W, deadline=None):
    """
    Try to reduce the rank by greedily removing columns that are redundant.
    For each column, check if removing it still gives a valid decomposition.
    Also try to merge proportional columns.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if rank == 0:
        return U, V, W, 0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_rows = [list(U_arr[i]) for i in range(len(U_arr))]
    V_rows = [list(V_arr[i]) for i in range(len(V_arr))]
    W_rows = [list(W_arr[i]) for i in range(len(W_arr))]
    
    new_rank = rank
    
    # Pass 1: Remove individually redundant columns (iterate until no more can be removed)
    progress = True
    while progress:
        progress = False
        for r in range(new_rank):
            if deadline is not None and time.time() > deadline:
                break
            # Try removing column r
            test_U = [row[:r] + row[r+1:] for row in U_rows]
            test_V = [row[:r] + row[r+1:] for row in V_rows]
            test_W = [row[:r] + row[r+1:] for row in W_rows]
            
            if _verify(test_U, test_V, test_W):
                U_rows = test_U
                V_rows = test_V
                W_rows = test_W
                new_rank -= 1
                progress = True
                break  # Restart from beginning after removal
        if deadline is not None and time.time() > deadline:
            break
    
    # Pass 2: Try to merge proportional columns
    U_arr2 = np.array(U_rows, dtype=float)
    V_arr2 = np.array(V_rows, dtype=float)
    W_arr2 = np.array(W_rows, dtype=float)
    
    merged = True
    while merged:
        merged = False
        for r1 in range(new_rank):
            if deadline is not None and time.time() > deadline:
                break
            for r2 in range(r1+1, new_rank):
                if deadline is not None and time.time() > deadline:
                    break
                # Check if columns r1 and r2 are proportional in all three matrices
                # U[:,r1] = c * U[:,r2], V[:,r1] = c * V[:,r2], W[:,r1] = c * W[:,r2]
                # for some scalar c
                
                # Find c from U
                nonzero_u = np.abs(U_arr2[:, r2]) > 1e-15
                if not np.any(nonzero_u):
                    # U[:,r2] is zero, check if U[:,r1] is also zero
                    if np.allclose(U_arr2[:, r1], 0):
                        # Check V and W
                        nonzero_v = np.abs(V_arr2[:, r2]) > 1e-15
                        if np.any(nonzero_v):
                            c_v = V_arr2[nonzero_v, r1] / V_arr2[nonzero_v, r2]
                            if np.allclose(c_v, c_v[0], atol=1e-10):
                                c = c_v[0]
                                if np.allclose(V_arr2[:, r1], c * V_arr2[:, r2], atol=1e-10):
                                    nonzero_w = np.abs(W_arr2[:, r2]) > 1e-15
                                    if np.any(nonzero_w):
                                        c_w = W_arr2[nonzero_w, r1] / W_arr2[nonzero_w, r2]
                                        if np.allclose(c_w, c_w[0], atol=1e-10):
                                            if c_w[0] == c:
                                                if np.allclose(W_arr2[:, r1], c * W_arr2[:, r2], atol=1e-10):
                                                    # Merge: combine r1 and r2
                                                    # New column = U_r1 + U_r2, V_r1 + V_r2, W_r1 + W_r2
                                                    # But this is only valid if the sum is still rank-1
                                                    # Since they're proportional, sum is proportional to each
                                                    U_arr2[:, r1] = U_arr2[:, r1] + U_arr2[:, r2]
                                                    V_arr2[:, r1] = V_arr2[:, r1] + V_arr2[:, r2]
                                                    W_arr2[:, r1] = W_arr2[:, r1] + W_arr2[:, r2]
                                                    # Remove column r2
                                                    U_arr2 = np.delete(U_arr2, r2, axis=1)
                                                    V_arr2 = np.delete(V_arr2, r2, axis=1)
                                                    W_arr2 = np.delete(W_arr2, r2, axis=1)
                                                    U_rows = [list(row) for row in U_arr2]
                                                    V_rows = [list(row) for row in V_arr2]
                                                    W_rows = [list(row) for row in W_arr2]
                                                    new_rank -= 1
                                                    merged = True
                                                    break
                        else:
                            # V[:,r2] is zero
                            if np.allclose(V_arr2[:, r1], 0):
                                nonzero_w = np.abs(W_arr2[:, r2]) > 1e-15
                                if np.any(nonzero_w):
                                    c_w = W_arr2[nonzero_w, r1] / W_arr2[nonzero_w, r2]
                                    if np.allclose(c_w, c_w[0], atol=1e-10):
                                        c = c_w[0]
                                        if np.allclose(W_arr2[:, r1], c * W_arr2[:, r2], atol=1e-10):
                                            U_arr2[:, r1] = U_arr2[:, r1] + U_arr2[:, r2]
                                            V_arr2[:, r1] = V_arr2[:, r1] + V_arr2[:, r2]
                                            W_arr2[:, r1] = W_arr2[:, r1] + W_arr2[:, r2]
                                            U_arr2 = np.delete(U_arr2, r2, axis=1)
                                            V_arr2 = np.delete(V_arr2, r2, axis=1)
                                            W_arr2 = np.delete(W_arr2, r2, axis=1)
                                            U_rows = [list(row) for row in U_arr2]
                                            V_rows = [list(row) for row in V_arr2]
                                            W_rows = [list(row) for row in W_arr2]
                                            new_rank -= 1
                                            merged = True
                                            break
                    else:
                        # Not proportional
                        pass
                else:
                    c = U_arr2[nonzero_u, r1] / U_arr2[nonzero_u, r2]
                    if np.allclose(c, c[0], atol=1e-10):
                        c = c[0]
                        if np.allclose(U_arr2[:, r1], c * U_arr2[:, r2], atol=1e-10):
                            # Check V
                            nonzero_v = np.abs(V_arr2[:, r2]) > 1e-15
                            if np.any(nonzero_v):
                                c_v = V_arr2[nonzero_v, r1] / V_arr2[nonzero_v, r2]
                                if np.allclose(c_v, c_v[0], atol=1e-10):
                                    if c_v[0] == c:
                                        if np.allclose(V_arr2[:, r1], c * V_arr2[:, r2], atol=1e-10):
                                            # Check W
                                            nonzero_w = np.abs(W_arr2[:, r2]) > 1e-15
                                            if np.any(nonzero_w):
                                                c_w = W_arr2[nonzero_w, r1] / W_arr2[nonzero_w, r2]
                                                if np.allclose(c_w, c_w[0], atol=1e-10):
                                                    if c_w[0] == c:
                                                        if np.allclose(W_arr2[:, r1], c * W_arr2[:, r2], atol=1e-10):
                                                            # Merge
                                                            U_arr2[:, r1] = U_arr2[:, r1] + U_arr2[:, r2]
                                                            V_arr2[:, r1] = V_arr2[:, r1] + V_arr2[:, r2]
                                                            W_arr2[:, r1] = W_arr2[:, r1] + W_arr2[:, r2]
                                                            U_arr2 = np.delete(U_arr2, r2, axis=1)
                                                            V_arr2 = np.delete(V_arr2, r2, axis=1)
                                                            W_arr2 = np.delete(W_arr2, r2, axis=1)
                                                            U_rows = [list(row) for row in U_arr2]
                                                            V_rows = [list(row) for row in V_arr2]
                                                            W_rows = [list(row) for row in W_arr2]
                                                            new_rank -= 1
                                                            merged = True
                                                            break
                            else:
                                if np.allclose(V_arr2[:, r1], 0):
                                    nonzero_w = np.abs(W_arr2[:, r2]) > 1e-15
                                    if np.any(nonzero_w):
                                        c_w = W_arr2[nonzero_w, r1] / W_arr2[nonzero_w, r2]
                                        if np.allclose(c_w, c_w[0], atol=1e-10):
                                            if c_w[0] == c:
                                                if np.allclose(W_arr2[:, r1], c * W_arr2[:, r2], atol=1e-10):
                                                    U_arr2[:, r1] = U_arr2[:, r1] + U_arr2[:, r2]
                                                    V_arr2[:, r1] = V_arr2[:, r1] + V_arr2[:, r2]
                                                    W_arr2[:, r1] = W_arr2[:, r1] + W_arr2[:, r2]
                                                    U_arr2 = np.delete(U_arr2, r2, axis=1)
                                                    V_arr2 = np.delete(V_arr2, r2, axis=1)
                                                    W_arr2 = np.delete(W_arr2, r2, axis=1)
                                                    U_rows = [list(row) for row in U_arr2]
                                                    V_rows = [list(row) for row in V_arr2]
                                                    W_rows = [list(row) for row in W_arr2]
                                                    new_rank -= 1
                                                    merged = True
                                                    break
        if deadline is not None and time.time() > deadline:
            break
    
    if new_rank < rank:
        return U_rows, V_rows, W_rows, new_rank
    
    return U, V, W, rank

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Tries Strassen-based decomposition (rank 98) first, then tries to reduce rank,
    falls back to schoolbook (rank 128).
    """
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin for the 150s limit
    
    best_result = None
    best_rank = float('inf')
    
    # Try Strassen-based decomposition
    try:
        result = _try_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            best_result = result
            best_rank = len(result["U"][0])
            
            # Try to reduce rank
            if time.time() < deadline:
                U_red, V_red, W_red, new_rank = _reduce_rank(
                    result["U"], result["V"], result["W"], deadline=deadline
                )
                if new_rank < best_rank:
                    if _verify(U_red, V_red, W_red):
                        best_result = {"U": U_red, "V": V_red, "W": W_red}
                        best_rank = new_rank
    except Exception:
        pass
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
