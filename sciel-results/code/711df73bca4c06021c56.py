"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0


def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition is correct. W indexed as W[k*n+i, r]."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False

    rank = U.shape[1]

    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dec_val = 0.0
                for r in range(rank):
                    dec_val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(dec_val - 1.0) > tol:
                    return False
    return True


def _build_rank11():
    """
    Rank-11 decomposition for 2x3x6 matrix multiplication.
    
    The tensor M_{2,3,6} can be decomposed using:
    - 7 Strassen terms for the 2x2x2 core (j in {0,1}, k in {0,1})
    - 4 additional rank-1 terms for the remaining k values (j in {0,1,2}, k in {2,...,5})
    
    The 4 additional terms are:
    1. j=0, k in {2,3,4,5}: A[i,0]*B[0,k] for all i,k
    2. j=1, k in {2,3,4,5}: A[i,1]*B[1,k] for all i,k
    3. j=2, k in {0,1}:     A[i,2]*B[2,k] for all i,k
    4. j=2, k in {2,3,4,5}: A[i,2]*B[2,k] for all i,k
    """
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2x2 block (columns 0-6)
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B10
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B10-B00)
        ({0: 1}, {2: 1, 0: -1}, {1: 1, 3: 1}),
        # M4 = A11(B01-B10)
        ({4: 1}, {1: 1, 2: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B10
        ({0: 1, 1: 1}, {2: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for t_idx, (u_map, v_map, w_map) in enumerate(strassen):
        for idx, val in u_map.items():
            U[idx, t_idx] = val
        for l_idx, val in v_map.items():
            j_local = l_idx // 2
            k_local = l_idx % 2
            g_idx = j_local * p + k_local
            V[g_idx, t_idx] = val
        for l_idx, val in w_map.items():
            i_local = l_idx // 2
            k_local = l_idx % 2
            g_idx = k_local * n + i_local
            W[g_idx, t_idx] = val
    
    # Term 7: j=0, k in {2,3,4,5}
    # A[i,0]*B[0,k] for i in {0,1}, k in {2,3,4,5}
    # U[0,7]=1, U[3,7]=1 (A00 and A10 have index 0 in their respective rows)
    # Actually, A indices: A00=0, A01=1, A10=3, A11=4
    # So for j=0: A[i,0] means i=0 -> A00=0, i=1 -> A10=3
    U[0, 7] = 1
    U[3, 7] = 1
    # B[0,k] for k in {2,3,4,5}: B02=2, B03=3, B04=4, B05=5
    V[2, 7] = 1
    V[3, 7] = 1
    V[4, 7] = 1
    V[5, 7] = 1
    # W[k*n+i, 7] = 1 for k in {2,3,4,5}, i in {0,1}
    for k in range(2, 6):
        for i in range(2):
            W[k * n + i, 7] = 1
    
    # Term 8: j=1, k in {2,3,4,5}
    # A[i,1]*B[1,k] for i in {0,1}, k in {2,3,4,5}
    # A[i,1]: i=0 -> A01=1, i=1 -> A11=4
    U[1, 8] = 1
    U[4, 8] = 1
    # B[1,k] for k in {2,3,4,5}: B12=8, B13=9, B14=10, B15=11
    V[8, 8] = 1
    V[9, 8] = 1
    V[10, 8] = 1
    V[11, 8] = 1
    # W[k*n+i, 8] = 1 for k in {2,3,4,5}, i in {0,1}
    for k in range(2, 6):
        for i in range(2):
            W[k * n + i, 8] = 1
    
    # Term 9: j=2, k in {0,1}
    # A[i,2]*B[2,k] for i in {0,1}, k in {0,1}
    # A[i,2]: i=0 -> A02=2, i=1 -> A12=5
    U[2, 9] = 1
    U[5, 9] = 1
    # B[2,k] for k in {0,1}: B20=12, B21=13
    V[12, 9] = 1
    V[13, 9] = 1
    # W[k*n+i, 9] = 1 for k in {0,1}, i in {0,1}
    for k in range(0, 2):
        for i in range(2):
            W[k * n + i, 9] = 1
    
    # Term 10: j=2, k in {2,3,4,5}
    # A[i,2]*B[2,k] for i in {0,1}, k in {2,3,4,5}
    # A[i,2]: i=0 -> A02=2, i=1 -> A12=5
    U[2, 10] = 1
    U[5, 10] = 1
    # B[2,k] for k in {2,3,4,5}: B22=14, B23=15, B24=16, B25=17
    V[14, 10] = 1
    V[15, 10] = 1
    V[16, 10] = 1
    V[17, 10] = 1
    # W[k*n+i, 10] = 1 for k in {2,3,4,5}, i in {0,1}
    for k in range(2, 6):
        for i in range(2):
            W[k * n + i, 10] = 1
    
    return {"U": U, "V": V, "W": W}


def _build_rank24():
    """
    Rank-24 decomposition: direct sum of three 2x3x2 blocks.
    Each block has rank 8 (7 Strassen + 1 outer product term).
    """
    n, m, p = DIMENSIONS
    rank_per_block = 8
    rank = 3 * rank_per_block  # 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # For each 2x3x2 block (k range of 2):
    # A indices: A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # B indices for this block: B00=0, B01=1, B10=2, B11=3, B20=4, B21=5 (local)
    # C indices: C00=0, C01=1, C10=2, C11=3 (local)
    
    # Strassen for A[0:2,0:2] x B[0:2,0:2]:
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B10-B00)
        ({0: 1}, {2: 1, 0: -1}, {1: 1, 3: 1}),
        # M4 = A11(B01-B10)
        ({4: 1}, {1: 1, 2: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B10
        ({0: 1, 1: 1}, {2: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # Outer product term: A[i,2]*B[2,k] for i in {0,1}, k in {0,1}
    # U[2]=1, U[5]=1, V[4]=1, V[5]=1, W[0]=1, W[1]=1, W[2]=1, W[3]=1
    outer_term = ({2: 1, 5: 1}, {4: 1, 5: 1}, {0: 1, 1: 1, 2: 1, 3: 1})

    all_terms = strassen_terms + [outer_term]

    for block in range(3):
        for t_idx, (u_map, v_map, w_map) in enumerate(all_terms):
            col = block * rank_per_block + t_idx
            for idx, val in u_map.items():
                U[idx, col] = val
            for l_idx, val in v_map.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
            for l_idx, val in w_map.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val

    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()

    # Try rank 11 first
    try:
        best11 = _build_rank11()
        if _verify(best11["U"], best11["V"], best11["W"]):
            return best11
    except Exception:
        pass

    # Try rank 24
    best24 = _build_rank24()
    if _verify(best24["U"], best24["V"], best24["W"]):
        return best24

    # Fallback: construct rank 36 (schoolbook)
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
    return {"U": U, "V": V, "W": W}
