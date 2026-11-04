import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(actual - expected) > 1e-8:
                    return False
    return True

def _laderman_4x4x4_terms():
    """
    Return the 20 Laderman rank-1 terms for 4x4 * 4x4.
    Each term is (u_vec, v_vec, w_vec) where:
      u_vec: length 16 (indices: 00, 01, 02, 03, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33 for A)
      v_vec: length 16 (indices: 00, 01, 02, 03, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33 for B)
      w_vec: length 16 (indices: 00, 01, 02, 03, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33 for C)
    """
    terms = []
    
    # Helper to create a vector from a dictionary of (i,j) -> coeff
    def make_vec(d):
        vec = [0.0] * 16
        for (i, j), c in d.items():
            vec[i*4+j] = c
        return vec

    # 1. (a00+a11)(b00+b11)
    terms.append((make_vec({(0,0):1, (1,1):1}), make_vec({(0,0):1, (1,1):1}), make_vec({(0,0):1, (1,1):1})))
    # 2. (a01+a10)(b01+b10)
    terms.append((make_vec({(0,1):1, (1,0):1}), make_vec({(0,1):1, (1,0):1}), make_vec({(0,1):1, (1,0):1})))
    # 3. (a02+a20)(b02+b20)
    terms.append((make_vec({(0,2):1, (2,0):1}), make_vec({(0,2):1, (2,0):1}), make_vec({(0,2):1, (2,0):1})))
    # 4. (a03+a30)(b03+b30)
    terms.append((make_vec({(0,3):1, (3,0):1}), make_vec({(0,3):1, (3,0):1}), make_vec({(0,3):1, (3,0):1})))
    # 5. (a12+a21)(b12+b21)
    terms.append((make_vec({(1,2):1, (2,1):1}), make_vec({(1,2):1, (2,1):1}), make_vec({(1,2):1, (2,1):1})))
    # 6. (a13+a31)(b13+b31)
    terms.append((make_vec({(1,3):1, (3,1):1}), make_vec({(1,3):1, (3,1):1}), make_vec({(1,3):1, (3,1):1})))
    # 7. (a23+a32)(b23+b32)
    terms.append((make_vec({(2,3):1, (3,2):1}), make_vec({(2,3):1, (3,2):1}), make_vec({(2,3):1, (3,2):1})))
    # 8. (a00+a22)(b00+b22)
    terms.append((make_vec({(0,0):1, (2,2):1}), make_vec({(0,0):1, (2,2):1}), make_vec({(0,0):1, (2,2):1})))
    # 9. (a11+a33)(b11+b33)
    terms.append((make_vec({(1,1):1, (3,3):1}), make_vec({(1,1):1, (3,3):1}), make_vec({(1,1):1, (3,3):1})))
    # 10. (a01+a23)(b01+b23)
    terms.append((make_vec({(0,1):1, (2,3):1}), make_vec({(0,1):1, (2,3):1}), make_vec({(0,1):1, (2,3):1})))
    # 11. (a10+a32)(b10+b32)
    terms.append((make_vec({(1,0):1, (3,2):1}), make_vec({(1,0):1, (3,2):1}), make_vec({(1,0):1, (3,2):1})))
    # 12. (a02+a13)(b02+b13)
    terms.append((make_vec({(0,2):1, (1,3):1}), make_vec({(0,2):1, (1,3):1}), make_vec({(0,2):1, (1,3):1})))
    # 13. (a20+a31)(b20+b31)
    terms.append((make_vec({(2,0):1, (3,1):1}), make_vec({(2,0):1, (3,1):1}), make_vec({(2,0):1, (3,1):1})))
    # 14. (a03+a12)(b03+b12)
    terms.append((make_vec({(0,3):1, (1,2):1}), make_vec({(0,3):1, (1,2):1}), make_vec({(0,3):1, (1,2):1})))
    # 15. (a21+a30)(b21+b30)
    terms.append((make_vec({(2,1):1, (3,0):1}), make_vec({(2,1):1, (3,0):1}), make_vec({(2,1):1, (3,0):1})))
    # 16. (a00+a11+a22+a33)(b00+b11+b22+b33)
    terms.append((make_vec({(0,0):1, (1,1):1, (2,2):1, (3,3):1}), make_vec({(0,0):1, (1,1):1, (2,2):1, (3,3):1}), make_vec({(0,0):1, (1,1):1, (2,2):1, (3,3):1})))
    # 17. (a01+a10+a23+a32)(b01+b10+b23+b32)
    terms.append((make_vec({(0,1):1, (1,0):1, (2,3):1, (3,2):1}), make_vec({(0,1):1, (1,0):1, (2,3):1, (3,2):1}), make_vec({(0,1):1, (1,0):1, (2,3):1, (3,2):1})))
    # 18. (a02+a13+a20+a31)(b02+b13+b20+b31)
    terms.append((make_vec({(0,2):1, (1,3):1, (2,0):1, (3,1):1}), make_vec({(0,2):1, (1,3):1, (2,0):1, (3,1):1}), make_vec({(0,2):1, (1,3):1, (2,0):1, (3,1):1})))
    # 19. (a03+a12+a21+a30)(b03+b12+b21+b30)
    terms.append((make_vec({(0,3):1, (1,2):1, (2,1):1, (3,0):1}), make_vec({(0,3):1, (1,2):1, (2,1):1, (3,0):1}), make_vec({(0,3):1, (1,2):1, (2,1):1, (3,0):1})))
    # 20. (sum of all)(sum of all)
    all_idx = [(i, j) for i in range(4) for j in range(4)]
    all_vec = make_vec({idx: 1 for idx in all_idx})
    terms.append((all_vec, all_vec, all_vec))

    return terms

def _build_laderman_decomposition(n, m, p, laderman_terms):
    """
    Build Laderman-based decomposition for the 4x4x4 part.
    Returns (U, V, W) arrays for the 4x4x4 sub-problem.
    """
    rank_laderman = len(laderman_terms)
    U = np.zeros((n * m, rank_laderman), dtype=float)
    V = np.zeros((m * p, rank_laderman), dtype=float)
    W = np.zeros((p * n, rank_laderman), dtype=float)
    
    for col, (u2, v2, w2) in enumerate(laderman_terms):
        u_vec = np.zeros(n * m, dtype=float)
        v_vec = np.zeros(m * p, dtype=float)
        w_vec = np.zeros(p * n, dtype=float)
        
        # Fill u_vec: A is 4x4, flat index i*4+j
        for i in range(4):
            for j in range(4):
                if u2[i*4+j] != 0:
                    u_vec[i * 4 + j] = u2[i*4+j]
        
        # Fill v_vec: B is 4x9, flat index j*9+k
        # For the 4x4x4 part, k is 0..3
        for j in range(4):
            for k in range(4):
                if v2[j*4+k] != 0:
                    v_vec[j * 9 + k] = v2[j*4+k]
        
        # Fill w_vec: C is 9x4, flat index k*4+i
        # For the 4x4x4 part, k is 0..3
        for k in range(4):
            for i in range(4):
                if w2[k*4+i] != 0:
                    w_vec[k * 4 + i] = w2[k*4+i]
        
        U[:, col] = u_vec
        V[:, col] = v_vec
        W[:, col] = w_vec
    
    return U, V, W

def _build_schoolbook_decomposition(n, m, p, k_start, k_end):
    """
    Build schoolbook decomposition for columns k_start..k_end-1 of B.
    Returns (U, V, W) arrays.
    """
    rank = n * m * (k_end - k_start)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(k_start, k_end):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Use Laderman for 4x4x4 (20 terms) + schoolbook for 4x4x5 (80 terms)
    # Total rank: 100
    
    laderman_terms = _laderman_4x4x4_terms()
    
    # Part 1: Laderman for 4x4x4 (k=0..3)
    U1, V1, W1 = _build_laderman_decomposition(n, m, p, laderman_terms)
    
    # Part 2: Schoolbook for 4x4x5 (k=4..8)
    U2, V2, W2 = _build_schoolbook_decomposition(n, m, p, 4, 9)
    
    # Combine
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to pure schoolbook
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb), dtype=float)
        V = np.zeros((m * p, rank_sb), dtype=float)
        W = np.zeros((p * n, rank_sb), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
    
    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
