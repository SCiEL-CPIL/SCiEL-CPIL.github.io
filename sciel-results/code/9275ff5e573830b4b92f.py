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

    # Try to find rank-9 decomposition for 2x2x3 (would give rank-51 total)
    if time.time() < deadline - 3 and best_rank == 52:
        U9, V9, W9 = _find_2x2x3_rank9(deadline)
        if U9 is not None:
            U51, V51, W51 = _build_combined_51(n, m, p, U9, V9, W9)
            if _verify_int(U51, V51, W51, n, m, p):
                best_rank = 51
                best_U, best_V, best_W = U51, V51, W51

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
    Uses structured search with linear algebra to find valid integer solutions.
    """
    n, m, p = 2, 2, 3
    rank = 10

    # Strategy: Use 7 Strassen terms for k=0,1 and 3 additional terms for k=2.
    # The 3 additional terms must handle: C02 = A00*B02 + A01*B12, C12 = A10*B02 + A11*B12
    # while not disturbing the k=0,1 results.
    
    # First, set up the 7 Strassen terms (fixed)
    U_base = np.zeros((4, 7), dtype=int)
    V_base = np.zeros((6, 7), dtype=int)
    W_base = np.zeros((6, 7), dtype=int)
    
    # p1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    U_base[0,0] = 1; U_base[3,0] = 1
    V_base[0,0] = 1; V_base[3,0] = 1
    W_base[0,0] = 1; W_base[3,0] = 1
    
    # p2 = (A10+A11)B00 -> contributes to C10, C11
    U_base[2,1] = 1; U_base[3,1] = 1
    V_base[0,1] = 1
    W_base[1,1] = 1; W_base[3,1] = -1
    
    # p3 = A00(B01-B11) -> contributes to C01, C11
    U_base[0,2] = 1
    V_base[1,2] = 1; V_base[3,2] = -1
    W_base[2,2] = 1; W_base[3,2] = 1
    
    # p4 = A11(B10-B00) -> contributes to C00, C10
    U_base[3,3] = 1
    V_base[2,3] = 1; V_base[0,3] = -1
    W_base[0,3] = 1; W_base[1,3] = 1
    
    # p5 = (A00+A01)B11 -> contributes to C00, C01
    U_base[0,4] = 1; U_base[1,4] = 1
    V_base[3,4] = 1
    W_base[0,4] = -1; W_base[2,4] = 1
    
    # p6 = (A10-A00)(B00+B01) -> contributes to C11
    U_base[2,5] = 1; U_base[0,5] = -1
    V_base[0,5] = 1; V_base[1,5] = 1
    W_base[3,5] = 1
    
    # p7 = (A01-A11)(B10+B11) -> contributes to C00
    U_base[1,6] = 1; U_base[3,6] = -1
    V_base[2,6] = 1; V_base[3,6] = 1
    W_base[0,6] = 1
    
    # Verify base gives correct k=0,1 results
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for r in range(7):
                    val += U_base[i*2+j, r] * V_base[j*3+k, r] * W_base[k*2+i, r]
                if val != 1:
                    # Base is wrong, try different approach
                    return _find_2x2x3_rank10_alternative(deadline)
    
    # Now find 3 additional terms for k=2.
    # We need: for each (i,j,k=2), sum of contributions = 1
    # The 3 new terms (r=7,8,9) have U, V, W entries.
    # V entries for k=2 are V[4,r] and V[5,r] (B02 and B12)
    # W entries for k=2 are W[4,r] and W[5,r] (C02 and C12)
    
    # Try small integer values for the 3 additional terms.
    # Each term has 4 U values, 2 V(k=2) values, 2 W(k=2) values = 8 params.
    # But we also need V and W entries for k=0,1 to be 0 (to not disturb base).
    
    # Simplification: assume the 3 new terms only have V entries at k=2 and W entries at k=2.
    # Then we need: for each (i,j), sum_{r=7}^{9} U[i,j,r] * V[j,2,r] * W[2,i,r] = 1
    
    # This is a 4x3 matrix equation (4 (i,j) pairs, 3 terms) with 12 unknowns.
    # Let's try to find a solution with small integers.
    
    # Try specific patterns:
    # Pattern 1: 
    # Term 7: U=[1,0,0,0], V=[1,0], W=[1,0] -> A00*B02*C02
    # Term 8: U=[0,1,0,0], V=[0,1], W=[1,0] -> A01*B12*C02
    # Term 9: U=[0,0,1,1], V=[1,1], W=[0,1] -> (A10+A11)*(B02+B12)*C12
    
    # Check: C02 = 1*1*1 + 1*1*1 + 0 = 2 (wrong, need 1)
    # C12 = 0 + 0 + 1*1*1 + 1*1*1 = 2 (wrong, need 1)
    
    # Pattern 2:
    # Term 7: U=[1,0,0,0], V=[1,0], W=[1,0]
    # Term 8: U=[0,1,0,0], V=[0,1], W=[1,0]  
    # Term 9: U=[0,0,1,1], V=[1/2,1/2], W=[0,1] -> but we need integers
    
    # Let me try a different approach: use the fact that we can have negative coefficients.
    # Try to solve the system:
    # Let a_r = U[0,r], b_r = U[1,r], c_r = U[2,r], d_r = U[3,r]
    # Let e_r = V[4,r], f_r = V[5,r] (V entries for k=2)
    # Let g_r = W[4,r], h_r = W[5,r] (W entries for k=2)
    
    # Equations:
    # (i=0,j=0,k=2): a7*e7*g7 + a8*e8*g8 + a9*e9*g9 = 1
    # (i=0,j=1,k=2): b7*f7*g7 + b8*f8*g8 + b9*f9*g9 = 1
    # (i=1,j=0,k=2): c7*e7*h7 + c8*e8*h8 + c9*e9*h9 = 1
    # (i=1,j=1,k=2): d7*f7*h7 + d8*f8*h8 + d9*f9*h9 = 1
    
    # Try a simple solution:
    # Set a7=1, e7=1, g7=1, others 0 in term 7
    # Set b8=1, f8=1, g8=1, others 0 in term 8
    # Set c9=1, e9=1, h9=1, d9=1, f9=1, h9=1, others 0 in term 9
    # But h9 appears in both equations for i=1
    
    # Let me try:
    # Term 7: U=[1,0,0,0], V=[1,0], W=[1,0]
    # Term 8: U=[0,1,0,0], V=[0,1], W=[1,0]
    # Term 9: U=[0,0,1,1], V=[1,1], W=[0,1]
    # This gives C02=2, C12=2. Scale by 1/2? But we need integers.
    
    # Alternative: Use 4 terms instead of 3? No, we're limited to 10 total.
    
    # Let me try a different base structure. Maybe the 7 Strassen terms can be modified.
    
    # Actually, let's try a completely different approach: search over all possible
    # 10-term decompositions with entries in {-2,-1,0,1,2}.
    
    # This is too large. Let me try a more targeted search.
    
    # Known result: rank of M_{2,2,3} is 10. Let me try to find a specific construction.
    
    # Try: Use 4 terms for k=0, 3 for k=1, 3 for k=2.
    # Or: 3 for k=0, 4 for k=1, 3 for k=2.
    
    # Let me try a systematic search with a time limit.
    search_start = time.time()
    max_search_time = min(30, max(5, deadline - time.time() - 10))
    
    # Try random initializations and local search
    np.random.seed(42)
    
    for trial in range(1000):
        if time.time() - search_start > max_search_time:
            break
        
        # Random initialization with small integers
        U = np.random.randint(-2, 3, size=(4, 10))
        V = np.random.randint(-2, 3, size=(6, 10))
        W = np.random.randint(-2, 3, size=(6, 10))
        
        # Check if this gives the correct tensor
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    val = 0
                    for r in range(10):
                        val += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W
    
    # If random search fails, try a more structured approach.
    # Use the fact that we can have half-integer entries.
    # Try entries in {-1, -1/2, 0, 1/2, 1}
    
    search_start = time.time()
    max_search_time = min(20, max(3, deadline - time.time() - 10))
    
    for trial in range(500):
        if time.time() - search_start > max_search_time:
            break
        
        # Random initialization with half-integers
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(4, 10))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(6, 10))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(6, 10))
        
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    val = 0
                    for r in range(10):
                        val += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to integers by scaling
            U_int = (U * 2).astype(int)
            V_int = (V * 2).astype(int)
            W_int = (W * 2).astype(int)
            # But this changes the tensor! We need to be careful.
            # Actually, if we scale U by 2, we need to scale V or W by 1/2.
            # Let's just return the half-integer solution as is.
            return U, V, W
    
    return None, None, None


def _find_2x2x3_rank10_alternative(deadline):
    """Alternative approach for rank-10 decomposition."""
    # Try a different base structure
    n, m, p = 2, 2, 3
    rank = 10
    
    # Use 4 schoolbook terms for k=0, 3 for k=1, 3 for k=2
    U = np.zeros((4, 10), dtype=int)
    V = np.zeros((6, 10), dtype=int)
    W = np.zeros((6, 10), dtype=int)
    
    # k=0: 4 schoolbook terms
    U[0,0] = 1; V[0,0] = 1; W[0,0] = 1  # A00*B00 -> C00
    U[1,1] = 1; V[3,1] = 1; W[2,1] = 1  # A01*B10 -> C01
    U[2,2] = 1; V[0,2] = 1; W[1,2] = 1  # A10*B00 -> C10
    U[3,3] = 1; V[3,3] = 1; W[3,3] = 1  # A11*B10 -> C11
    
    # k=1: 3 terms (need to handle 4 products with 3 terms)
    # Try: (A00+A11)*(B01+B11), A01*B11, A10*B01
    U[0,4] = 1; U[3,4] = 1
    V[1,4] = 1; V[3,4] = 1
    W[0,4] = 1; W[3,4] = 1
    
    U[1,5] = 1
    V[3,5] = 1
    W[2,5] = 1; W[3,5] = -1
    
    U[2,6] = 1
    V[1,6] = 1
    W[1,6] = 1; W[3,6] = -1
    
    # k=2: 3 terms
    # Try: (A00+A11)*(B02+B12), A01*B12, A10*B02
    U[0,7] = 1; U[3,7] = 1
    V[4,7] = 1; V[5,7] = 1
    W[4,7] = 1; W[5,7] = 1
    
    U[1,8] = 1
    V[5,8] = 1
    W[4,8] = 1; W[5,8] = -1
    
    U[2,9] = 1
    V[4,9] = 1
    W[5,9] = 1; W[5,9] = -1  # This is wrong, W[5,9] can't be both 1 and -1
    
    # Fix: use different W entries
    W[5,9] = 1
    
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(3):
                val = 0
                for r in range(10):
                    val += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
                if val != 1:
                    return None, None, None
    
    return U, V, W


def _find_2x2x3_rank9(deadline):
    """
    Search for rank-9 decomposition of 2x2x3 matrix multiplication.
    This is more challenging; may not exist.
    """
    n, m, p = 2, 2, 3
    rank = 9
    
    # Quick check: is rank 9 possible? The lower bound for M_{2,2,3} is 9.
    # Let's try to find it with a time-limited search.
    
    search_start = time.time()
    max_search_time = min(15, max(2, deadline - time.time() - 5))
    
    np.random.seed(123)
    
    for trial in range(2000):
        if time.time() - search_start > max_search_time:
            break
        
        U = np.random.randint(-2, 3, size=(4, 9))
        V = np.random.randint(-2, 3, size=(6, 9))
        W = np.random.randint(-2, 3, size=(6, 9))
        
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    val = 0
                    for r in range(9):
                        val += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W
    
    return None, None, None


def _build_combined_51(n, m, p, U9, V9, W9):
    """Build rank-51: 6 Strassen blocks + rank-9 2x2x3 block for columns 12,13,14."""
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

    # Add rank-9 terms for columns 12, 13, 14
    for r in range(9):
        u_entries = []
        v_entries = []
        w_entries = []

        for i in range(n):
            for j in range(m):
                val = U9[i*m+j, r]
                if val != 0:
                    u_entries.append((i*m+j, val))

        for j in range(m):
            for k_local in range(3):
                val = V9[j*3 + k_local, r]
                if val != 0:
                    k_global = 12 + k_local
                    v_entries.append((j*p + k_global, val))

        for k_local in range(3):
            for i in range(n):
                val = W9[k_local*n + i, r]
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
# EVOLVE-BLOCK-END
