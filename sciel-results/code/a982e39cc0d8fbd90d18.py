import time
import random
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify that the decomposition is correct using numpy for speed."""
    r = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float).reshape(n*m, r)
    V_np = np.array(V, dtype=float).reshape(m*p, r)
    W_np = np.array(W, dtype=float).reshape(p*n, r)
    
    for c in range(r):
        u_mat = U_np[:, c].reshape(n, m)
        v_mat = V_np[:, c].reshape(m, p)
        prod = u_mat @ v_mat
        w_mat = W_np[:, c].reshape(p, n)
        if not np.allclose(prod, w_mat.T, atol=1e-10):
            return False
    return True

def _verify_fast(U, V, W, n=3, m=5, p=7):
    """Fast vectorized verification."""
    r = len(U[0]) if U else 0
    if r == 0:
        return True
    U_np = np.array(U, dtype=float).reshape(n*m, r)
    V_np = np.array(V, dtype=float).reshape(m*p, r)
    W_np = np.array(W, dtype=float).reshape(p*n, r)
    
    # For each column c: reshape U[:,c] to (n,m), V[:,c] to (m,p), compute product, compare with W[:,c].T reshaped to (n,m)
    # Vectorize: U_mat[i,j,c] = U_np[i*m+j, c], V_mat[j,k,c] = V_np[j*p+k, c]
    # prod[i,k,c] = sum_j U_mat[i,j,c] * V_mat[j,k,c]
    # We need prod[i,k,c] == W_mat[k,i,c] for all i,k,c
    
    # Build 3D arrays
    U_mat = U_np.reshape(n, m, r)  # U_mat[i,j,c]
    V_mat = V_np.reshape(m, p, r)  # V_mat[j,k,c]
    W_mat = W_np.reshape(p, n, r)  # W_mat[k,i,c]
    
    # Compute tensor product: prod[i,k,c] = sum_j U_mat[i,j,c] * V_mat[j,k,c]
    prod = np.einsum('ijc,jkc->ikc', U_mat, V_mat)
    
    return np.allclose(prod, W_mat.T, atol=1e-10)

def _construct_schoolbook():
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

def _strassen_3x3x3(i0, j0, k0, n, m, p, U, V, W, col):
    """
    Strassen-like decomposition for 3x3x3 block.
    Uses 7 multiplications instead of 27.
    We embed a 2x2x2 Strassen in the 3x3x3 by using the first 2 rows/cols of each dimension.
    The 3rd row/col uses schoolbook for the remaining terms.
    
    Actually, let's use a proper 3x3x3 decomposition.
    For 3x3x3, the best known rank is 23 (Landsberg et al.), but let's use a simpler approach:
    Split 3 = 2+1, apply Strassen to the 2x2x2 part (rank 7) and schoolbook for the rest.
    """
    # Split into 2+1 in each dimension
    # 2x2x2 part: Strassen (rank 7)
    # 2x2x1, 2x1x2, 1x2x2, 2x1x1, 1x2x1, 1x1x2, 1x1x1 parts: schoolbook
    
    # 2x2x2 Strassen at (i0, j0, k0)
    a_coeffs = [
        [(0,1), (1,1)],
        [(1,0), (1,1)],
        [(0,0)],
        [(1,1)],
        [(0,0), (0,1)],
        [(1,0), (0,0)],
        [(0,1), (1,1)],
    ]
    a_signs = [
        [1, 1],
        [1, 1],
        [1],
        [1],
        [1, 1],
        [1, -1],
        [1, -1],
    ]
    b_coeffs = [
        [(0,0), (1,1)],
        [(0,0)],
        [(1,0), (1,1)],
        [(0,1), (0,0)],
        [(1,1)],
        [(0,0), (0,1)],
        [(1,0), (1,1)],
    ]
    b_signs = [
        [1, 1],
        [1],
        [1, -1],
        [1, -1],
        [1],
        [1, 1],
        [1, 1],
    ]
    c_contribs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]
    
    for t in range(7):
        c = col + t
        for (di, dj), coeff in zip(a_coeffs[t], a_signs[t]):
            row = (i0 + di) * m + (j0 + dj)
            U[row][c] += coeff
        for (dj, dk), coeff in zip(b_coeffs[t], b_signs[t]):
            row = (j0 + dj) * p + (k0 + dk)
            V[row][c] += coeff
        for (dk, di), coeff in c_contribs[t].items():
            row = (k0 + dk) * n + (i0 + di)
            W[row][c] += coeff
    col += 7
    
    # Schoolbook for the remaining parts involving the 3rd index (index 2)
    # (2, j, k) for j=0,1 and k=0,1: 4 terms
    # (i, 2, k) for i=0,1 and k=0,1: 4 terms
    # (i, j, 2) for i=0,1 and j=0,1: 4 terms
    # (2, 2, k) for k=0,1: 2 terms
    # (2, j, 2) for j=0,1: 2 terms
    # (i, 2, 2) for i=0,1: 2 terms
    # (2, 2, 2): 1 term
    # Total: 4+4+4+2+2+2+1 = 19 terms
    # But wait, some of these overlap with the Strassen part? No, Strassen only covers indices 0,1.
    
    remaining = [
        (2, 0, 0), (2, 1, 0), (2, 0, 1), (2, 1, 1),
        (0, 2, 0), (1, 2, 0), (0, 2, 1), (1, 2, 1),
        (0, 0, 2), (1, 0, 2), (0, 1, 2), (1, 1, 2),
        (2, 2, 0), (2, 2, 1),
        (2, 0, 2), (2, 1, 2),
        (0, 2, 2), (1, 2, 2),
        (2, 2, 2),
    ]
    
    for (di, dj, dk) in remaining:
        U[(i0 + di) * m + (j0 + dj)][col] = 1
        V[(j0 + dj) * p + (k0 + dk)][col] = 1
        W[(k0 + dk) * n + (i0 + di)][col] = 1
        col += 1
    
    return col

def _schoolbook_block(i0, j0, k0, hi, hj, hk, n, m, p, U, V, W, col):
    """Schoolbook for sub-block of size hi x hj x hk."""
    for di in range(hi):
        for dj in range(hj):
            for dk in range(hk):
                U[(i0 + di) * m + (j0 + dj)][col] += 1
                V[(j0 + dj) * p + (k0 + dk)][col] += 1
                W[(k0 + dk) * n + (i0 + di)][col] += 1
                col += 1
    return col

def _try_block_decomposition(splits_i, splits_j, splits_k, use_strassen_3x3x3=True):
    """
    Construct using block decomposition.
    splits_i, splits_j, splits_k are lists of block sizes that sum to n, m, p.
    For each block, if it's 3x3x3 and use_strassen_3x3x3, use Strassen; else schoolbook.
    """
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    i_offsets = [0]
    for s in splits_i[:-1]:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in splits_j[:-1]:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in splits_k[:-1]:
        k_offsets.append(k_offsets[-1] + s)
    
    for ii in range(len(splits_i)):
        for jj in range(len(splits_j)):
            for kk in range(len(splits_k)):
                hi = splits_i[ii]
                hj = splits_j[jj]
                hk = splits_k[kk]
                i0 = i_offsets[ii]
                j0 = j_offsets[jj]
                k0 = k_offsets[kk]
                
                if hi == 3 and hj == 3 and hk == 3 and use_strassen_3x3x3:
                    col = _strassen_3x3x3(i0, j0, k0, n, m, p, U, V, W, col)
                else:
                    col = _schoolbook_block(i0, j0, k0, hi, hj, hk, n, m, p, U, V, W, col)
    
    # Trim trailing zero columns
    last_col = 0
    for c in range(col - 1, -1, -1):
        nz = False
        for row in range(n * m):
            if U[row][c] != 0:
                nz = True
                break
        if not nz:
            for row in range(m * p):
                if V[row][c] != 0:
                    nz = True
                    break
        if not nz:
            for row in range(p * n):
                if W[row][c] != 0:
                    nz = True
                    break
        if nz:
            last_col = c
            break
    final_rank = last_col + 1
    
    U_trim = [row[:final_rank] for row in U]
    V_trim = [row[:final_rank] for row in V]
    W_trim = [row[:final_rank] for row in W]
    
    return {"U": U_trim, "V": V_trim, "W": W_trim, "_rank": final_rank}

def _all_compositions(total, min_part=1, max_parts=4):
    """Generate all compositions of total into parts >= min_part, up to max_parts parts."""
    results = []
    for num in range(1, max_parts + 1):
        for comp in itertools.product(range(min_part, total + 1), repeat=num):
            if sum(comp) == total:
                results.append(comp)
    return results

def _try_structured_search(deadline):
    """
    Try structured decompositions using 3x3x3 Strassen blocks.
    Key splits for 3x5x7:
    - i: [3] or [2,1] or [1,2] or [1,1,1]
    - j: [5] or [3,2] or [2,3] or [2,2,1] etc.
    - k: [7] or [3,4] or [4,3] or [3,2,2] etc.
    """
    n, m, p = DIMENSIONS
    best = None
    best_rank = n * m * p
    
    # Focus on splits that include a 3x3x3 block
    comp_i = [(3,), (2,1), (1,2), (1,1,1)]
    comp_j = [(5,), (3,2), (2,3), (2,2,1), (2,1,2), (1,2,2), (3,1,1), (1,3,1), (1,1,3)]
    comp_k = [(7,), (3,4), (4,3), (3,2,2), (2,3,2), (2,2,3), (4,2,1), (2,4,1), (1,4,2), (3,3,1), (3,1,3), (1,3,3)]
    
    for ci in comp_i:
        if time.time() > deadline - 20:
            break
        for cj in comp_j:
            if time.time() > deadline - 20:
                break
            for ck in comp_k:
                if time.time() > deadline - 20:
                    break
                try:
                    result = _try_block_decomposition(ci, cj, ck, use_strassen_3x3x3=True)
                    rank = result["_rank"]
                    if _verify_fast(result["U"], result["V"], result["W"]):
                        if rank < best_rank:
                            best_rank = rank
                            best = {"U": result["U"], "V": result["V"], "W": result["W"]}
                except Exception:
                    pass
    
    return best

def construct():
    deadline = time.time() + 110
    best_result = None
    best_rank = 105
    
    # 1. Schoolbook baseline
    schoolbook = _construct_schoolbook()
    if _verify_fast(schoolbook["U"], schoolbook["V"], schoolbook["W"]):
        best_result = schoolbook
        best_rank = 105
    
    # 2. Try structured search with 3x3x3 Strassen blocks
    if time.time() < deadline - 30:
        hybrid = _try_structured_search(deadline)
        if hybrid is not None and _verify_fast(hybrid["U"], hybrid["V"], hybrid["W"]):
            r = len(hybrid["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = hybrid
    
    if best_result is None:
        return schoolbook
    return best_result

# EVOLVE-BLOCK-END
