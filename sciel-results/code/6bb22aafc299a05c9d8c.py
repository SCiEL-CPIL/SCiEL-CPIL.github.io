"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2 x 2x2 -> 2x2 matrix multiplication.
    Each term: (a_coefs, b_coefs, c_coefs)
    a_coefs: {(i,j): val} for A[i,j]
    b_coefs: {(j,k): val} for B[j,k]
    c_coefs: {(k,i): val} for C[i,k]
    """
    # Strassen for 2x2:
    # m1=(A00+A11)(B00+B11)
    # m2=(A10+A11)(B00)
    # m3=(A00)(B10-B00)
    # m4=(A11)(B01-B10)
    # m5=(A00+A01)(B11)
    # m6=(A10-A00)(B00+B01)
    # m7=(A01-A11)(B10+B11)
    # C00=m1-m3-m4+m7
    # C01=m3+m5
    # C10=m2+m4
    # C11=m1-m2+m5-m6
    
    terms = [
        # m1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # m2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):-1, (1,0):1}),
        # m3
        ({(0,0):1}, {(1,0):1, (0,0):-1}, {(0,0):-1, (0,1):1}),
        # m4
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):-1, (1,0):1}),
        # m5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),
        # m6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):-1}),
        # m7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    return terms


def _compose_strassen_4x4():
    """
    Compose two levels of Strassen to get rank-49 decomposition for 4x4 x 4x4 -> 4x4.
    Returns list of 49 (a_coefs, b_coefs, c_coefs) dicts.
    """
    s2 = _strassen_2x2_terms()
    
    # Block-level Strassen for 4x4 (2x2 blocks of 2x2 each)
    # M1=(A00+A11)(B00+B11), C00+=M1, C11+=M1
    # M2=(A10+A11)(B00), C10+=M2, C11-=M2
    # M3=(A00)(B10-B00), C00-=M3, C01+=M3
    # M4=(A11)(B01-B10), C00-=M4, C10+=M4
    # M5=(A00+A01)(B11), C01+=M5, C11+=M5
    # M6=(A10-A00)(B00+B01), C11-=M6
    # M7=(A01-A11)(B10+B11), C00+=M7
    
    block_defs = [
        # (a_block_map, b_block_map, c_block_map)
        # a_block_map: {(block_r, block_c): sign} for A
        # b_block_map: {(block_r, block_c): sign} for B
        # c_block_map: {(block_r, block_c): sign} for C
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(1,0):1, (0,0):-1}, {(0,0):-1, (0,1):1}),
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):-1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):-1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    all_terms = []
    
    for bi, (a_bmap, b_bmap, c_bmap) in enumerate(block_defs):
        # For each of the 7 scalar Strassen terms within this block product
        for si, (sa, sb, sc) in enumerate(s2):
            # Build full 4x4 coefficient dicts
            a_full = {}
            for (br, bc), bsign in a_bmap.items():
                for (i, j), asgn in sa.items():
                    key = (br*2+i, bc*2+j)
                    a_full[key] = a_full.get(key, 0) + bsign * asgn
            a_full = {k: v for k, v in a_full.items() if v != 0}
            
            b_full = {}
            for (br, bc), bsign in b_bmap.items():
                for (j, k), bs in sb.items():
                    key = (br*2+j, bc*2+k)
                    b_full[key] = b_full.get(key, 0) + bsign * bs
            b_full = {k: v for k, v in b_full.items() if v != 0}
            
            c_full = {}
            for (br, bc), csign in c_bmap.items():
                for (k, i), cs in sc.items():
                    key = (bc*2+k, br*2+i)  # (k,i) in full 4x4
                    c_full[key] = c_full.get(key, 0) + csign * cs
            c_full = {k: v for k, v in c_full.items() if v != 0}
            
            if a_full and b_full and c_full:
                all_terms.append((a_full, b_full, c_full))
    
    return all_terms


def _verify_decomposition(U, V, W, n, m, p, seed=42):
    """Verify that the decomposition correctly computes matrix multiplication."""
    np.random.seed(seed)
    A = np.random.randint(-3, 4, size=(n, m)).astype(float)
    B = np.random.randint(-3, 4, size=(m, p)).astype(float)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    rank = U.shape[1]
    for c in range(rank):
        u_mat = U[:, c].reshape(n, m)
        v_mat = V[:, c].reshape(m, p)
        w_mat = W[:, c].reshape(p, n)
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += w_mat[k, i] * np.dot(u_mat[i, :], v_mat[:, k])
    
    return np.allclose(C_computed, C_expected, atol=1e-9)


def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    """
    Matrix multiplication 4x4 by 4x9.
    Returns U (16 x rank), V (36 x rank), W (36 x rank) with integer entries.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9
    DEADLINE = start_time + 100  # safety margin
    
    # Try rank-114 decomposition: 2x Strassen-49 + schoolbook for 4x4x1
    strassen_terms = _compose_strassen_4x4()
    
    if len(strassen_terms) == 49:
        total_rank = 49 + 49 + 16
        U = np.zeros((n*m, total_rank), dtype=int)
        V = np.zeros((m*p, total_rank), dtype=int)
        W = np.zeros((p*n, total_rank), dtype=int)
        
        # Part 1: A * B[:, 0:4]
        for c, (a_coefs, b_coefs, c_coefs) in enumerate(strassen_terms):
            for (i, j), val in a_coefs.items():
                U[i*m + j, c] += val
            for (j, k), val in b_coefs.items():
                V[j*p + k, c] += val
            for (k, i), val in c_coefs.items():
                W[k*n + i, c] += val
        
        # Part 2: A * B[:, 4:8]
        offset = 49
        for c, (a_coefs, b_coefs, c_coefs) in enumerate(strassen_terms):
            for (i, j), val in a_coefs.items():
                U[i*m + j, offset + c] += val
            for (j, k), val in b_coefs.items():
                V[j*p + (k + 4), offset + c] += val
            for (k, i), val in c_coefs.items():
                W[k*n + i, offset + c] += val
        
        # Part 3: A * B[:, 8:9] (4x4 x 4x1 schoolbook)
        offset2 = 98
        for i in range(n):
            for j in range(m):
                col = offset2 + i*m + j
                U[i*m + j, col] = 1
                V[j*p + 8, col] = 1
                W[8*n + i, col] = 1
        
        # Verify
        if time.time() < DEADLINE and _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    if time.time() < DEADLINE:
        U_fb, V_fb, W_fb = _schoolbook(n, m, p)
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Last resort: return empty (should not happen)
    return {"U": [], "V": [], "W": []}
# EVOLVE-BLOCK-END
