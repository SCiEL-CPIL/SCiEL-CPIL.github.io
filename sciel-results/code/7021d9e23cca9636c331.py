"""Combined program: Strassen blocks + targeted integer search for 2x2x3 rank-10."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140  # 10s safety margin before 150s limit

    # Build rank-53 baseline (7 Strassen blocks + 1 schoolbook)
    U53, V53, W53 = _build_strassen_53(n, m, p)

    if not _verify_int(U53, V53, W53, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    best_rank = 53
    best_U, best_V, best_W = U53, V53, W53

    # Try to find rank-10 decomposition for 2x2x3 (columns 12,13,14)
    if time.time() < deadline - 5:
        U10, V10, W10 = _find_2x2x3_rank10(deadline)
        if U10 is not None:
            U52, V52, W52 = _build_combined_52(n, m, p, U10, V10, W10)
            if _verify_int(U52, V52, W52, n, m, p):
                best_rank = 52
                best_U, best_V, best_W = U52, V52, W52

    return {"U": best_U, "V": best_V, "W": best_W}


def _build_strassen_53(n, m, p):
    """Build rank-53: 7 Strassen blocks (rank 7 each) + 1 schoolbook (rank 4)."""
    n, m, p = 2, 2, 15
    blocks = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,)]

    all_U, all_V, all_W = [], [], []

    for block in blocks:
        if len(block) == 2:
            k0, k1 = block
            terms = _strassen_terms(k0, k1)
        else:
            k = block[0]
            terms = _schoolbook_terms(k)

        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i*m+j, c) for (i,j), c in u_dict.items()]
            v_entries = [(j*p+k_idx, c) for (j,k_idx), c in v_dict.items()]
            w_entries = [(k_idx*n+i, c) for (k_idx,i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    rank = len(all_U)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val

    return U, V, W


def _strassen_terms(k0, k1):
    """Strassen rank-7 terms for 2x2x2 block with k indices k0, k1."""
    return [
        # M1: (A00+A11)(B0k0+B1k1)
        ({(0,0):1, (1,1):1}, {(0,k0):1, (1,k1):1}, {(k0,0):1, (k1,1):1}),
        # M2: (A10+A11)B0k0
        ({(1,0):1, (1,1):1}, {(0,k0):1}, {(k0,1):1, (k1,1):-1}),
        # M3: A00(B0k1-B1k1)
        ({(0,0):1}, {(0,k1):1, (1,k1):-1}, {(k1,0):1, (k1,1):1}),
        # M4: A11(B1k0-B0k0)
        ({(1,1):1}, {(1,k0):1, (0,k0):-1}, {(k0,0):1, (k0,1):1}),
        # M5: (A00+A01)B1k1
        ({(0,0):1, (0,1):1}, {(1,k1):1}, {(k0,0):-1, (k1,0):1}),
        # M6: (A10-A00)(B0k0+B0k1)
        ({(1,0):1, (0,0):-1}, {(0,k0):1, (0,k1):1}, {(k1,1):1}),
        # M7: (A01-A11)(B1k0+B1k1)
        ({(0,1):1, (1,1):-1}, {(1,k0):1, (1,k1):1}, {(k0,0):1}),
    ]


def _schoolbook_terms(k):
    """Schoolbook rank-4 terms for 2x2x1 block."""
    return [
        ({(0,0):1}, {(0,k):1}, {(k,0):1}),
        ({(0,1):1}, {(1,k):1}, {(k,0):1}),
        ({(1,0):1}, {(0,k):1}, {(k,1):1}),
        ({(1,1):1}, {(1,k):1}, {(k,1):1}),
    ]


def _build_schoolbook(n, m, p):
    """Build schoolbook rank-60 decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _verify_int(U, V, W, n, m, p):
    """Verify exact integer decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i*m + j
                v_row = j*p + k
                w_row = k*n + i
                val = 0
                for c in range(rank):
                    val += U[u_row][c] * V[v_row][c] * W[w_row][c]
                if val != 1:
                    return False
    return True


def _build_combined_52(n, m, p, U10, V10, W10):
    """Build rank-52: 6 Strassen blocks + rank-10 2x2x3 block for columns 12,13,14."""
    n, m, p = 2, 2, 15
    blocks = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]

    all_U, all_V, all_W = [], [], []

    for block in blocks:
        k0, k1 = block
        terms = _strassen_terms(k0, k1)
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i*m+j, c) for (i,j), c in u_dict.items()]
            v_entries = [(j*p+k_idx, c) for (j,k_idx), c in v_dict.items()]
            w_entries = [(k_idx*n+i, c) for (k_idx,i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    # Add rank-10 terms for columns 12, 13, 14
    for r in range(10):
        u_entries = []
        v_entries = []
        w_entries = []

        for i in range(n):
            for j in range(m):
                val = U10[i*m+j, r]
                if val != 0:
                    u_entries.append((i*m+j, val))

        for j in range(m):
            for k_local in range(3):
                val = V10[j*3 + k_local, r]
                if val != 0:
                    k_global = 12 + k_local
                    v_entries.append((j*p + k_global, val))

        for k_local in range(3):
            for i in range(n):
                val = W10[k_local*n + i, r]
                if val != 0:
                    k_global = 12 + k_local
                    w_entries.append((k_global*n + i, val))

        all_U.append(u_entries)
        all_V.append(v_entries)
        all_W.append(w_entries)

    rank = len(all_U)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val

    return U, V, W


def _find_2x2x3_rank10(deadline):
    """
    Search for rank-10 decomposition of 2x2x3 matrix multiplication.
    Uses systematic search over U and V patterns, solving for W via least squares.
    """
    n, m, p = 2, 2, 3
    rank = 10
    b_vec = np.ones(12)

    # Generate candidate U patterns: each term has a subset of {0,1,2,3} with values in {-1,0,1}
    # Focus on patterns with at most 2 non-zero entries (sparse)
    u_candidates = []
    for i in range(4):
        for j in range(i+1, 4):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    pat = [0,0,0,0]
                    pat[i] = si
                    pat[j] = sj
                    u_candidates.append(pat)
    # Also single-entry patterns
    for i in range(4):
        for s in [-1, 1]:
            pat = [0,0,0,0]
            pat[i] = s
            u_candidates.append(pat)

    # Generate candidate V patterns: each term has entries in {0..5} (j*p+k)
    # Focus on patterns with at most 2 non-zero entries
    v_candidates = []
    for i in range(6):
        for j in range(i+1, 6):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    pat = [0]*6
                    pat[i] = si
                    pat[j] = sj
                    v_candidates.append(pat)
    for i in range(6):
        for s in [-1, 1]:
            pat = [0]*6
            pat[i] = s
            v_candidates.append(pat)

    # Strategy: For each combination of 10 U patterns and 10 V patterns,
    # solve for W and check if integer solution exists.
    # This is too large, so use a smarter approach:
    # Fix the first 7 U patterns to Strassen, and search over the last 3 U and all V.

    # Strassen U patterns (7 terms)
    strassen_U = [
        [1,0,0,1],   # A00+A11
        [0,0,1,1],   # A10+A11
        [1,0,0,0],   # A00
        [0,0,0,1],   # A11
        [1,1,0,0],   # A00+A01
        [-1,0,1,0],  # A10-A00
        [0,1,0,-1],  # A01-A11
    ]

    # For the remaining 3 U patterns, try from u_candidates
    # For V, try patterns from v_candidates

    # To make this tractable, use a greedy approach:
    # 1. Fix U to Strassen + 3 extra
    # 2. For each V pattern combination, solve for W
    # 3. Check if W is integer

    # Precompute: for a given U (4x10) and V (6x10), build the 12x60 system for W
    def build_W_system(U_mat, V_mat):
        A = np.zeros((12, 60))
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        u_val = U_mat[i, r]
                        v_val = V_mat[j * p + k, r]
                        if u_val != 0 and v_val != 0:
                            w_idx = (k * n + i) * rank + r
                            A[eq_idx, w_idx] = u_val * v_val
                    eq_idx += 1
        return A

    def solve_and_check(U_mat, V_mat):
        A = build_W_system(U_mat, V_mat)
        try:
            W_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
        except:
            return None
        W_sol = W_sol.reshape((6, 10))
        W_rounded = np.round(W_sol)
        # Verify
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_mat[i, r] * V_mat[j*p+k, r] * W_rounded[k*n+i, r]
                    if abs(val - 1.0) > 1e-6:
                        error += abs(val - 1.0)
        if error < 1e-6:
            return W_rounded.astype(int)
        return None

    # Try different combinations of extra U patterns
    # Use a limited set of extra U patterns to keep search tractable
    extra_U_options = [
        [1,0,0,0],   # A00
        [0,1,0,0],   # A01
        [0,0,1,0],   # A10
        [0,0,0,1],   # A11
        [1,0,0,1],   # A00+A11
        [0,1,0,1],   # A01+A11
        [1,1,0,0],   # A00+A01
        [0,0,1,1],   # A10+A11
        [1,0,0,-1],  # A00-A11
        [0,1,0,-1],  # A01-A11
        [-1,0,1,0],  # A10-A00
        [1,1,0,0],   # A00+A01 (dup, ok)
    ]

    # Try different V pattern sets for the 3 extra terms
    # V patterns for extra terms: focus on k=2 positions (indices 4,5) and some k=0,1
    extra_V_options = [
        [0,0,0,0,1,0],   # B02
        [0,0,0,0,0,1],   # B12
        [0,0,0,0,1,1],   # B02+B12
        [0,0,0,0,1,-1],  # B02-B12
        [1,0,0,0,1,0],   # B00+B02
        [0,0,1,0,0,1],   # B10+B12
        [0,0,0,1,1,0],   # B11+B02
        [1,1,0,0,0,0],   # B00+B01
        [0,0,1,1,0,0],   # B10+B11
        [0,1,0,1,0,0],   # B01+B11
        [1,0,0,1,0,0],   # B00+B11
        [0,0,0,0,-1,1],  # B12-B02
    ]

    # Search: for each combination of 3 extra U patterns and 3 extra V patterns
    # This is C(12,3)^2 which is too large. Use random sampling.
    import random
    random.seed(42)

    num_trials = 500
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break

        # Pick 3 extra U patterns
        extra_U = [list(extra_U_options[random.randint(0, len(extra_U_options)-1)]) for _ in range(3)]
        U_mat = np.zeros((4, rank))
        for r in range(7):
            for i in range(4):
                U_mat[i, r] = strassen_U[r][i]
        for r in range(3):
            for i in range(4):
                U_mat[i, 7+r] = extra_U[r][i]

        # Pick 3 extra V patterns
        extra_V = [list(extra_V_options[random.randint(0, len(extra_V_options)-1)]) for _ in range(3)]
        V_mat = np.zeros((6, rank))
        # Standard Strassen V for first 7 terms
        strassen_V = [
            [1,0,0,1,0,0],   # B00+B11
            [1,0,0,0,0,0],   # B00
            [0,1,0,-1,0,0],  # B01-B11
            [-1,0,1,0,0,0],  # B10-B00
            [0,0,0,1,0,0],   # B11
            [1,1,0,0,0,0],   # B00+B01
            [0,0,1,1,0,0],   # B10+B11
        ]
        for r in range(7):
            for jk in range(6):
                V_mat[jk, r] = strassen_V[r][jk]
        for r in range(3):
            for jk in range(6):
                V_mat[jk, 7+r] = extra_V[r][jk]

        W_sol = solve_and_check(U_mat, V_mat)
        if W_sol is not None:
            return U_mat.astype(int), V_mat.astype(int), W_sol

    # If random search fails, try a more structured approach:
    # Use the linearity in W more directly.
    # For fixed U and V, the problem is: find W such that A*W_flat = b
    # where A is 12x60. We want W to be integer.
    # Use integer programming via exhaustive search on a subset of W variables.

    # Alternative: try a specific known construction from literature.
    # Based on the paper "The rank of the 2x2x3 matrix multiplication tensor"
    # One known rank-10 construction:

    # Let me try a specific ansatz:
    U_mat = np.zeros((4, rank))
    V_mat = np.zeros((6, rank))

    # Term 0: (A00+A11)(B00+B11+B02)
    U_mat[0,0] = 1; U_mat[3,0] = 1
    V_mat[0,0] = 1; V_mat[3,0] = 1; V_mat[4,0] = 1

    # Term 1: (A10+A11)(B00+B12)
    U_mat[2,1] = 1; U_mat[3,1] = 1
    V_mat[0,1] = 1; V_mat[5,1] = 1

    # Term 2: A00(B01-B11)
    U_mat[0,2] = 1
    V_mat[1,2] = 1; V_mat[3,2] = -1

    # Term 3: A11(B10-B00)
    U_mat[3,3] = 1
    V_mat[2,3] = 1; V_mat[0,3] = -1

    # Term 4: (A00+A01)(B11+B12)
    U_mat[0,4] = 1; U_mat[1,4] = 1
    V_mat[3,4] = 1; V_mat[5,4] = 1

    # Term 5: (A10-A00)(B00+B01)
    U_mat[2,5] = 1; U_mat[0,5] = -1
    V_mat[0,5] = 1; V_mat[1,5] = 1

    # Term 6: (A01-A11)(B10+B11)
    U_mat[1,6] = 1; U_mat[3,6] = -1
    V_mat[2,6] = 1; V_mat[3,6] = 1

    # Term 7: A00*B02
    U_mat[0,7] = 1
    V_mat[4,7] = 1

    # Term 8: A01*B12
    U_mat[1,8] = 1
    V_mat[5,8] = 1

    # Term 9: (A10+A11)(B02+B12)
    U_mat[2,9] = 1; U_mat[3,9] = 1
    V_mat[4,9] = 1; V_mat[5,9] = 1

    W_sol = solve_and_check(U_mat, V_mat)
    if W_sol is not None:
        return U_mat.astype(int), V_mat.astype(int), W_sol

    return None, None, None


def _verify_2x2x3(U, V, W, tol=1e-9):
    """Verify 2x2x3 decomposition."""
    n, m, p = 2, 2, 3
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(actual - 1.0) > tol:
                    return False
    return True


def _find_2x2x4_rank14(deadline):
    """
    Search for rank-14 decomposition of 2x2x4 matrix multiplication.
    2x2x4 = 2x2x2 + 2x2x2, so rank <= 14 (two Strassen blocks).
    Check if rank 14 is achievable (it is, trivially).
    """
    n, m, p = 2, 2, 4
    rank = 14
    
    # Just use two Strassen blocks for k=(0,1) and k=(2,3).
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))  # m*p = 2*4 = 8
    W = np.zeros((8, rank))  # p*n = 4*2 = 8
    
    # Block 1: k=0,1 (Strassen)
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V indices: 0=B00, 1=B01, 2=B10, 3=B11, 4=B02, 5=B03, 6=B12, 7=B13
    # W indices: 0=C00, 1=C10, 2=C01, 3=C11, 4=C02, 5=C12, 6=C03, 7=C13
    
    # p1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # p2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    
    # p3 = A00(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    
    # p4 = A11(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    # p5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    # p6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # p7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    # Block 2: k=2,3 (Strassen)
    # p8 = (A00+A11)(B02+B13)
    U[0,7] = 1; U[3,7] = 1
    V[4,7] = 1; V[7,7] = 1
    W[4,7] = 1; W[7,7] = 1
    
    # p9 = (A10+A11)B02
    U[2,8] = 1; U[3,8] = 1
    V[4,8] = 1
    W[5,8] = 1; W[7,8] = -1
    
    # p10 = A00(B03-B13)
    U[0,9] = 1
    V[5,9] = 1; V[7,9] = -1
    W[6,9] = 1; W[7,9] = 1
    
    # p11 = A11(B12-B02)
    U[3,10] = 1
    V[6,10] = 1; V[4,10] = -1
    W[4,10] = 1; W[5,10] = 1
    
    # p12 = (A00+A01)B13
    U[0,11] = 1; U[1,11] = 1
    V[7,11] = 1
    W[4,11] = -1; W[6,11] = 1
    
    # p13 = (A10-A00)(B02+B03)
    U[2,12] = 1; U[0,12] = -1
    V[4,12] = 1; V[5,12] = 1
    W[7,12] = 1
    
    # p14 = (A01-A11)(B12+B13)
    U[1,13] = 1; U[3,13] = -1
    V[6,13] = 1; V[7,13] = 1
    W[4,13] = 1
    
    # Verify
    if _verify_2x2x4(U, V, W):
        return U, V, W
    return None, None, None


def _verify_2x2x4(U, V, W, tol=1e-9):
    """Verify 2x2x4 decomposition."""
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(actual - 1.0) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
