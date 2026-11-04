import time
import random
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify that the decomposition is correct."""
    r = len(U[0])
    for c in range(r):
        uc = np.array([U[i][c] for i in range(n*m)])
        vc = np.array([V[j][c] for j in range(m*p)])
        wc = np.array([W[k][c] for k in range(p*n)])
        u_mat = uc.reshape(n, m)
        v_mat = vc.reshape(m, p)
        prod = u_mat @ v_mat
        w_mat = wc.reshape(p, n)
        if not np.allclose(prod, w_mat.T, atol=1e-10):
            return False
    return True

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

def _construct_strassen_block(i0, j0, k0, n, m, p, U, V, W, col):
    """Apply Strassen to 2x2 block at row i0..i0+1, col j0..j0+1, producing k0..k0+1."""
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
    
    return col + 7

def _construct_schoolbook_block(i0, j0, k0, hi, hj, hk, U, V, W, col):
    """Schoolbook for sub-block of size hi x hj x hk."""
    for di in range(hi):
        for dj in range(hj):
            for dk in range(hk):
                U[(i0 + di) * m + (j0 + dj)][col] += 1
                V[(j0 + dj) * p + (k0 + dk)][col] += 1
                W[(k0 + dk) * n + (i0 + di)][col] += 1
                col += 1
    return col

def _construct_hybrid(splits_i, splits_j, splits_k, use_strassen_for_2x2):
    """Construct using block decomposition."""
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
                
                if hi == 2 and hj == 2 and hk == 2 and use_strassen_for_2x2:
                    col = _construct_strassen_block(i0, j0, k0, n, m, p, U, V, W, col)
                else:
                    col = _construct_schoolbook_block(i0, j0, k0, hi, hj, hk, U, V, W, col)
    
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
    
    return {"U": U_trim, "V": V_trim, "W": W_trim, "_rank": final_rank, "_col": col}

def _all_compositions(total, min_part=1, max_parts=4):
    """Generate all compositions of total into parts >= min_part, up to max_parts parts."""
    results = []
    for num in range(1, max_parts + 1):
        for comp in itertools.product(range(min_part, total + 1), repeat=num):
            if sum(comp) == total:
                results.append(comp)
    return results

def _try_strassen_3x3_block(i0, j0, k0, n, m, p, U, V, W, col):
    """Try to find a low-rank decomposition for a 3x3x3 block.
    Returns (col_after, success) or (col, False) if not found."""
    # For 3x3x3, schoolbook is 27. Let's try to find better.
    # One approach: use Strassen on 2x2x2 sub-blocks and schoolbook for the rest.
    # Or try random coefficient search.
    
    # Try a specific pattern: split 3x3x3 into:
    # - One 2x2x2 Strassen block (7 mults)
    # - Remaining: rows 2, cols 2, or k=2 (schoolbook)
    # This gives 7 + 3 + 3 + 3 - overlaps... let's just try random search
    
    n3, m3, p3 = 3, 3, 3
    schoolbook_rank = 27
    
    # Try random coefficient search with small integers
    random.seed(42)  # Fixed seed for reproducibility within this function
    for attempt in range(50):
        # Try rank from 20 to 26
        target_rank = 20 + attempt % 7
        if target_rank >= schoolbook_rank:
            continue
            
        U_local = [[0] * target_rank for _ in range(n3 * m3)]
        V_local = [[0] * target_rank for _ in range(m3 * p3)]
        W_local = [[0] * target_rank for _ in range(p3 * n3)]
        
        # Initialize with random small coefficients
        for c in range(target_rank):
            # Random sparse pattern
            for _ in range(random.randint(2, 4)):
                ri = random.randint(0, n3*m3-1)
                U_local[ri][c] = random.choice([-1, 1])
            for _ in range(random.randint(2, 4)):
                rj = random.randint(0, m3*p3-1)
                V_local[rj][c] = random.choice([-1, 1])
            for _ in range(random.randint(2, 4)):
                rk = random.randint(0, p3*n3-1)
                W_local[rk][c] = random.choice([-1, 1])
        
        # Verify locally
        valid = True
        for c in range(target_rank):
            uc = np.array([U_local[i][c] for i in range(n3*m3)])
            vc = np.array([V_local[j][c] for j in range(m3*p3)])
            wc = np.array([W_local[k][c] for k in range(p3*n3)])
            u_mat = uc.reshape(n3, m3)
            v_mat = vc.reshape(m3, p3)
            prod = u_mat @ v_mat
            w_mat = wc.reshape(p3, n3)
            if not np.allclose(prod, w_mat.T, atol=1e-10):
                valid = False
                break
        
        if valid:
            # Found a valid 3x3x3 decomposition!
            for i in range(n3):
                for j in range(m3):
                    for c in range(target_rank):
                        U[(i0+i)*m + (j0+j)][col+c] += U_local[i*m3+j][c]
            for j in range(m3):
                for k in range(p3):
                    for c in range(target_rank):
                        V[(j0+j)*p + (k0+k)][col+c] += V_local[j*p3+k][c]
            for k in range(p3):
                for i in range(n3):
                    for c in range(target_rank):
                        W[(k0+k)*n + (i0+i)][col+c] += W_local[k*n3+i][c]
            return col + target_rank, True
    
    return col, False

def _search_splits(deadline):
    n, m, p = DIMENSIONS
    best = None
    best_rank = n * m * p
    
    comp_i = _all_compositions(n)
    comp_j = _all_compositions(m)
    comp_k = _all_compositions(p)
    
    for ci in comp_i:
        if time.time() > deadline - 20:
            break
        for cj in comp_j:
            if time.time() > deadline - 20:
                break
            for ck in comp_k:
                if time.time() > deadline - 20:
                    break
                for use_str in [True, False]:
                    if time.time() > deadline - 20:
                        break
                    try:
                        result = _construct_hybrid(ci, cj, ck, use_str)
                        rank = result["_rank"]
                        if _verify(result["U"], result["V"], result["W"]):
                            if rank < best_rank:
                                best_rank = rank
                                best = {"U": result["U"], "V": result["V"], "W": result["W"]}
                    except Exception:
                        pass
    return best

def _try_direct_random_search(deadline, start_rank=90):
    """Try to find a low-rank decomposition via random search."""
    n, m, p = DIMENSIONS
    best = None
    best_rank = n * m * p
    
    random.seed(12345)
    
    for target_rank in range(start_rank, n*m*p):
        if time.time() > deadline - 30:
            break
        
        # Try a few random attempts per rank
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Initialize with sparse random coefficients
            for c in range(target_rank):
                # Each factor vector should be sparse
                num_u = random.randint(1, 3)
                indices_u = random.sample(range(n*m), min(num_u, n*m))
                for idx in indices_u:
                    U[idx][c] = random.choice([-1, 1])
                
                num_v = random.randint(1, 3)
                indices_v = random.sample(range(m*p), min(num_v, m*p))
                for idx in indices_v:
                    V[idx][c] = random.choice([-1, 1])
                
                num_w = random.randint(1, 3)
                indices_w = random.sample(range(p*n), min(num_w, p*n))
                for idx in indices_w:
                    W[idx][c] = random.choice([-1, 1])
            
            if _verify(U, V, W):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best = {"U": U, "V": V, "W": W}
                break  # Found valid for this rank, try lower
    
    return best

# EVOLVE-BLOCK-END

def construct():
    deadline = time.time() + 100
    best_result = None
    best_rank = 106
    
    # 1. Try schoolbook
    schoolbook = _construct_schoolbook()
    if _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"]):
        best_result = schoolbook
        best_rank = 105
    
    # 2. Try systematic hybrid search
    if time.time() < deadline - 30:
        hybrid = _search_splits(deadline)
        if hybrid is not None and _verify(hybrid["U"], hybrid["V"], hybrid["W"]):
            r = len(hybrid["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = hybrid
    
    # 3. Try random search if time remains
    if time.time() < deadline - 40 and best_rank > 100:
        random_result = _try_direct_random_search(deadline, start_rank=max(90, best_rank - 10))
        if random_result is not None and _verify(random_result["U"], random_result["V"], random_result["W"]):
            r = len(random_result["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = random_result
    
    if best_result is None:
        return schoolbook
    return best_result
