"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE_S = 120.0

def _time_left():
    return _DEADLINE_S - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-7):
    """Verify decomposition using the trace convention W[k*n+i,r]."""
    n, m, p = 3, 3, 3
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_random(U, V, W, num_trials=5, tol=1e-6):
    """Verify with random integer matrices as a stronger check."""
    n, m, p = 3, 3, 3
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    for _ in range(num_trials):
        A = np.random.randint(-5, 6, (n, m), dtype=np.int64)
        B = np.random.randint(-5, 6, (m, p), dtype=np.int64)
        C_expected = A @ B
        
        A_flat = A.flatten()  # i*m+j
        B_flat = B.flatten()  # j*p+k
        
        U_proj = U_arr.T @ A_flat
        V_proj = V_arr.T @ B_flat
        
        C_computed = np.zeros((n, p), dtype=np.int64)
        for r in range(rank):
            p_r = U_proj[r] * V_proj[r]
            if p_r != 0:
                w_col = W_arr[:, r]
                w_matrix = w_col.reshape(p, n)  # row k, col i
                C_computed += p_r * w_matrix.T
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def _rank26_strassen():
    """
    Rank-26 decomposition: Strassen for 2x2 block + direct for cross terms.
    W convention: W[k*n+i] where C[i,k] maps to index k*3+i.
    C[0,0]->0, C[0,1]->3, C[0,2]->6
    C[1,0]->1, C[1,1]->4, C[1,2]->7
    C[2,0]->2, C[2,1]->5, C[2,2]->8
    """
    terms = []
    
    def add(u, v, w):
        terms.append((u, v, w))
    
    # --- Strassen for top-left 2x2 block (indices 0,1,3,4 in A; 0,1,3,4 in B) ---
    # A = [[a, b], [c, d]] where a=A[0,0]=idx0, b=A[0,1]=idx1, c=A[1,0]=idx3, d=A[1,1]=idx4
    # B = [[e, f], [g, h]] where e=B[0,0]=idx0, f=B[0,1]=idx1, g=B[1,0]=idx3, h=B[1,1]=idx4
    # C00->W[0], C01->W[3], C10->W[1], C11->W[4]
    
    # M1 = (a+d)(e+h): C00(+1), C11(+1)
    add({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})
    # M2 = (c+d)e: C10(+1)
    add({3:1, 4:1}, {0:1}, {1:1})
    # M3 = a(f-g): C01(+1)
    add({0:1}, {1:1, 3:-1}, {3:1})
    # M4 = d(h-f): C00(+1), C11(+1)
    add({4:1}, {4:1, 1:-1}, {0:1, 4:1})
    # M5 = (a+b)h: C00(-1), C01(+1)
    add({0:1, 1:1}, {4:1}, {0:-1, 3:1})
    # M6 = (c-a)(e+f): C11(+1)
    add({3:1, 0:-1}, {0:1, 1:1}, {4:1})
    # M7 = (b-d)(g+h): C00(-1), C11(+1)
    add({1:1, 4:-1}, {3:1, 4:1}, {0:-1, 4:1})
    
    # --- Bottom-right 1x1: A[2,2]*B[2,2] -> C[2,2] (W=8) ---
    add({8:1}, {8:1}, {8:1})
    
    # --- Cross terms ---
    # A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    add({2:1}, {6:1}, {0:1})   # A[0,2]*B[2,0] -> C[0,0]
    add({2:1}, {7:1}, {3:1})   # A[0,2]*B[2,1] -> C[0,1]
    add({5:1}, {6:1}, {1:1})   # A[1,2]*B[2,0] -> C[1,0]
    add({5:1}, {7:1}, {4:1})   # A[1,2]*B[2,1] -> C[1,1]
    
    # A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    add({0:1}, {2:1}, {6:1})   # A[0,0]*B[0,2] -> C[0,2]
    add({1:1}, {5:1}, {6:1})   # A[0,1]*B[1,2] -> C[0,2]
    add({3:1}, {2:1}, {7:1})   # A[1,0]*B[0,2] -> C[1,2]
    add({4:1}, {5:1}, {7:1})   # A[1,1]*B[1,2] -> C[1,2]
    
    # A[0:2, 2] * B[2, 2] -> C[0:2, 2]
    add({2:1}, {8:1}, {6:1})   # A[0,2]*B[2,2] -> C[0,2]
    add({5:1}, {8:1}, {7:1})   # A[1,2]*B[2,2] -> C[1,2]
    
    # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    add({6:1}, {0:1}, {2:1})   # A[2,0]*B[0,0] -> C[2,0]
    add({7:1}, {3:1}, {2:1})   # A[2,1]*B[1,0] -> C[2,0]
    add({6:1}, {1:1}, {5:1})   # A[2,0]*B[0,1] -> C[2,1]
    add({7:1}, {4:1}, {5:1})   # A[2,1]*B[1,1] -> C[2,1]
    
    # A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    add({8:1}, {6:1}, {2:1})   # A[2,2]*B[2,0] -> C[2,0]
    add({8:1}, {7:1}, {5:1})   # A[2,2]*B[2,1] -> C[2,1]
    
    # A[2, 0:2] * B[0:2, 2] -> C[2, 2]
    add({6:1}, {2:1}, {8:1})   # A[2,0]*B[0,2] -> C[2,2]
    add({7:1}, {5:1}, {8:1})   # A[2,1]*B[1,2] -> C[2,2]
    
    # Total: 7 + 1 + 4 + 4 + 2 + 4 + 2 + 2 = 26
    rank = len(terms)
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}

def _rank24_attempt():
    """
    Attempt rank-24 by combining cross-term pairs that share structure.
    Key observation: some cross terms can be merged using bilinear identities.
    Specifically, pairs like (A[i,2]*B[2,j] -> C[i,j]) for fixed i or j
    can sometimes be compressed.
    
    Strategy: Use Strassen (7) + 1x1 (1) + compressed cross terms (16) = 24
    """
    terms = []
    
    def add(u, v, w):
        terms.append((u, v, w))
    
    # Strassen 2x2 block (same as before)
    add({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})
    add({3:1, 4:1}, {0:1}, {1:1})
    add({0:1}, {1:1, 3:-1}, {3:1})
    add({4:1}, {4:1, 1:-1}, {0:1, 4:1})
    add({0:1, 1:1}, {4:1}, {0:-1, 3:1})
    add({3:1, 0:-1}, {0:1, 1:1}, {4:1})
    add({1:1, 4:-1}, {3:1, 4:1}, {0:-1, 4:1})
    
    # 1x1 corner
    add({8:1}, {8:1}, {8:1})
    
    # Cross terms - attempt compression
    # Group 1: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] (4 terms -> try 3)
    # M_a = (A[0,2]+A[1,2])*(B[2,0]+B[2,1]) = A02B20 + A02B21 + A12B20 + A12B21
    # This gives C[0,0]+C[0,1]+C[1,0]+C[1,1] all +1
    # But we need them separately. Use:
    # M_a = A[0,2]*B[2,0] -> C[0,0]
    # M_b = A[0,2]*B[2,1] -> C[0,1]  
    # M_c = A[1,2]*B[2,0] -> C[1,0]
    # M_d = A[1,2]*B[2,1] -> C[1,1]
    # Can we do better? (A02+A12)(B20+B21) - A12B20 - A02B21... no, that's still 4.
    # Actually: (A02+A12)*B20 -> C00+C10, (A02+A12)*B21 -> C01+C11
    # So: M1=(A02+A12)*B20 -> W{0:1,1:1}, M2=(A02+A12)*B21 -> W{3:1,4:1}
    # That's 2 terms for 4 outputs!
    add({2:1, 5:1}, {6:1}, {0:1, 1:1})
    add({2:1, 5:1}, {7:1}, {3:1, 4:1})
    
    # Group 2: A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2] (4 terms -> try 2)
    # (A00+A10)*B02 -> C02+C12, (A01+A11)*B12 -> C02+C12
    # Hmm, that gives C02 twice and C12 twice. Not right.
    # A00*B02 + A01*B12 -> C02, A10*B02 + A11*B12 -> C12
    # M1=(A00+A10)*B02 -> C02+C12 (wrong, gives both)
    # Actually: A00*B02 -> C02, A10*B02 -> C12: share B02
    # M1=A00*B02 -> W6, M2=A10*B02 -> W7, M3=A01*B12 -> W6, M4=A11*B12 -> W7
    # Can merge: (A00+A10)*B02 gives C02+C12, but we need them separately.
    # Use: M1=(A00+A10)*B02 -> W{6:1,7:1}? No, that adds to both.
    # Better: keep as 4 or use:
    # M1=A00*B02 -> W6, M2=A10*B02 -> W7
    # M3=A01*B12 -> W6, M4=A11*B12 -> W7
    # Alternative: (A00+B02-related)... Let's try:
    # M1=(A00+A11)*B02 -> C02+C12, M2=(A01+A10)*B12 -> C02+C12... no.
    # Stick with 4 for now, or try:
    # M1=A00*B02, M2=A10*B02, M3=A01*B12, M4=A11*B12 -> 4 terms
    # OR: M1=(A00+A10)*B02 -> contributes to C02 AND C12. Not separable.
    # Keep 4.
    add({0:1}, {2:1}, {6:1})
    add({3:1}, {2:1}, {7:1})
    add({1:1}, {5:1}, {6:1})
    add({4:1}, {5:1}, {7:1})
    
    # Group 3: A[0:2, 2] * B[2, 2] -> C[0:2, 2] (2 terms)
    # M1=(A02+A12)*B22 -> C02+C12
    add({2:1, 5:1}, {8:1}, {6:1, 7:1})
    
    # Group 4: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] (4 terms -> try 2)
    # A20*(B00+B01) -> C20+C21, A21*(B10+B11) -> C20+C21... no.
    # A20*B00 -> C20, A20*B01 -> C21, A21*B10 -> C20, A21*B11 -> C21
    # M1=A20*(B00+B01) -> C20+C21, M2=A21*(B10+B11) -> C20+C21
    # That gives 2*(C20+C21). Wrong.
    # M1=(A20+A21)*B00 -> C20+C20=2C20. Wrong.
    # Keep 4.
    add({6:1}, {0:1}, {2:1})
    add({6:1}, {1:1}, {5:1})
    add({7:1}, {3:1}, {2:1})
    add({7:1}, {4:1}, {5:1})
    
    # Group 5: A[2, 2] * B[2, 0:2] -> C[2, 0:2] (2 terms -> 1)
    # A22*(B20+B21) -> C20+C21
    add({8:1}, {6:1, 7:1}, {2:1, 5:1})
    
    # Group 6: A[2, 0:2] * B[0:2, 2] -> C[2, 2] (2 terms -> 1)
    # (A20*B02 + A21*B12) -> C22
    # M1=A20*B02, M2=A21*B12
    # Can we merge? (A20+A21)*(B02+B12) - A20*B12 - A21*B02... no.
    # Keep 2.
    add({6:1}, {2:1}, {8:1})
    add({7:1}, {5:1}, {8:1})
    
    # Count: 7 + 1 + 2 + 4 + 1 + 4 + 1 + 2 = 22
    # That's rank 22! Let's verify.
    rank = len(terms)
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find the minimal-rank exact tensor decomposition for 3x3 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' - factor matrices.
    """
    best = _schoolbook()
    best_rank = 27
    
    if _time_left() < 5.0:
        return best
    
    # Try rank-24 (or lower) attempt
    try:
        cand24 = _rank24_attempt()
        if _verify(cand24["U"], cand24["V"], cand24["W"]):
            if _verify_random(cand24["U"], cand24["V"], cand24["W"], num_trials=3):
                best = cand24
                best_rank = len(cand24["U"][0])
                return best
    except Exception:
        pass
    
    if _time_left() < 5.0:
        return best
    
    # Try rank-26 Strassen as fallback
    try:
        cand26 = _rank26_strassen()
        if _verify(cand26["U"], cand26["V"], cand26["W"]):
            if _verify_random(cand26["U"], cand26["V"], cand26["W"], num_trials=3):
                best = cand26
                best_rank = 26
                return best
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
