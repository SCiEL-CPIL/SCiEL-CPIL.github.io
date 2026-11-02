"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """Return the 7-term Strassen decomposition for 2x2 x 2x2 -> 2x2.
    Returns list of 7 triples (U_vec, V_vec, W_vec) where each vec is a list of 4 coeffs.
    U index: i*2+j for row i, col j of A (0=A00,1=A01,2=A10,3=A11)
    V index: j*2+k for row j, col k of B (0=B00,1=B01,2=B10,3=B11)
    W index: k*2+i for col k, row i of C (0=C00,1=C01,2=C10,3=C11)
    """
    # Standard Strassen decomposition
    # M1 = (A11 - A10)(B10 + B11)
    # M2 = (A00 + A10)B10
    # M3 = A00(B11 - B10)
    # M4 = A11(B00 + B11)
    # M5 = (A00 + A11)(B00 + B11)
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A11 - A10)(B01 + B11)
    # C00 = M5 + M4 - M2 - M6
    # C01 = M1 + M2
    # C10 = M3 + M5
    # C11 = M4 - M1 + M7
    
    terms = [
        # M1: U=[0,0,-1,1], V=[0,0,1,1], W=[0,1,0,-1]
        ([0, 0, -1, 1], [0, 0, 1, 1], [0, 1, 0, -1]),
        # M2: U=[1,0,1,0], V=[0,0,1,0], W=[-1,1,0,0]
        ([1, 0, 1, 0], [0, 0, 1, 0], [-1, 1, 0, 0]),
        # M3: U=[1,0,0,0], V=[0,0,-1,1], W=[0,0,1,0]
        ([1, 0, 0, 0], [0, 0, -1, 1], [0, 0, 1, 0]),
        # M4: U=[0,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
        ([0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M5: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,1,0]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0]),
        # M6: U=[-1,0,1,0], V=[1,0,1,0], W=[-1,0,0,0]
        ([-1, 0, 1, 0], [1, 0, 1, 0], [-1, 0, 0, 0]),
        # M7: U=[0,0,-1,1], V=[0,1,0,1], W=[0,0,0,1]
        ([0, 0, -1, 1], [0, 1, 0, 1], [0, 0, 0, 1]),
    ]
    return terms

def _verify_strassen():
    """Verify the Strassen decomposition is correct."""
    import numpy as np
    terms = _strassen_2x2x2()
    # Build 4x4x4 tensor
    T = np.zeros((4, 4, 4), dtype=float)
    for U, V, W in terms:
        T += np.einsum('i,j,k->ijk', np.array(U), np.array(V), np.array(W))
    # Target: T[i,j,k]=1 iff (i,j,k) is a valid (A,B,C) triple
    # A idx: i=0->A00, i=1->A01, i=2->A10, i=3->A11
    # B idx: j=0->B00, j=1->B01, j=2->B10, j=3->B11
    # C idx: k=0->C00, k=1->C01, k=2->C10, k=3->C11
    # C00 = A00*B00 + A01*B10 -> (0,0,0), (1,2,0)
    # C01 = A00*B01 + A01*B11 -> (0,1,1), (1,3,1)
    # C10 = A10*B00 + A11*B10 -> (2,0,2), (3,2,2)
    # C11 = A10*B01 + A11*B11 -> (2,1,3), (3,3,3)
    target = np.zeros((4, 4, 4), dtype=float)
    for (i, j, k) in [(0,0,0),(1,2,0),(0,1,1),(1,3,1),(2,0,2),(3,2,2),(2,1,3),(3,3,3)]:
        target[i, j, k] = 1
    return np.allclose(T, target, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Check if Strassen is valid
    use_strassen = False
    try:
        use_strassen = _verify_strassen()
    except Exception:
        use_strassen = False
    
    if not use_strassen:
        # Fall back to pure schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Use 4 independent 2x2x2 Strassen blocks + schoolbook for remaining
    # n=3: rows 0,1 use Strassen (2 rows), row 2 schoolbook
    # m=5: cols 0,1,2,3 can be split into two 2-col groups for Strassen, col 4 schoolbook
    # p=5: cols 0,1,2,3 can be split into two 2-col groups for Strassen, col 4 schoolbook
    
    # Strategy: 4 Strassen blocks of 2x2x2:
    # Block 1: n[0:2], m[0:2], p[0:2]
    # Block 2: n[0:2], m[0:2], p[2:4]
    # Block 3: n[0:2], m[2:4], p[0:2]
    # Block 4: n[0:2], m[2:4], p[2:4]
    # Schoolbook for all remaining (n=2 or m=4 or p=4)
    
    strassen_blocks = [
        # (n_start, m_start, p_start)
        (0, 0, 0),
        (0, 0, 2),
        (0, 2, 0),
        (0, 2, 2),
    ]
    
    U_entries = []
    V_entries = []
    W_entries = []
    
    # Track which (i,j,k) triples are covered by Strassen
    covered = set()
    
    # Add Strassen blocks
    for (n_s, m_s, p_s) in strassen_blocks:
        strassen_terms = _strassen_2x2x2()
        for U_vec, V_vec, W_vec in strassen_terms:
            U_entries.append([])
            V_entries.append([])
            W_entries.append([])
            col = len(U_entries) - 1
            
            for u_rel in range(4):
                if U_vec[u_rel] != 0:
                    u_row = n_s + u_rel // 2
                    u_col = m_s + u_rel % 2
                    u_idx = u_row * m + u_col
                    U_entries[col].append((u_idx, U_vec[u_rel]))
            
            for v_rel in range(4):
                if V_vec[v_rel] != 0:
                    v_row = m_s + v_rel // 2
                    v_col = p_s + v_rel % 2
                    v_idx = v_row * p + v_col
                    V_entries[col].append((v_idx, V_vec[v_rel]))
            
            for w_rel in range(4):
                if W_vec[w_rel] != 0:
                    w_row = p_s + w_rel // 2
                    w_col = n_s + w_rel % 2
                    w_idx = w_row * n + w_col
                    W_entries[col].append((w_idx, W_vec[w_rel]))
        
        # Mark covered triples
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    covered.add((n_s + i, m_s + j, p_s + k))
    
    # Add schoolbook for all uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U_entries.append([(u_idx, 1)])
                    V_entries.append([(v_idx, 1)])
                    W_entries.append([(w_idx, 1)])
    
    rank = len(U_entries)
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for u_idx, coeff in U_entries[col]:
            U[u_idx][col] = coeff
        for v_idx, coeff in V_entries[col]:
            V[v_idx][col] = coeff
        for w_idx, coeff in W_entries[col]:
            W[w_idx][col] = coeff
    
    # Verify before returning
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1
        
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        if not np.allclose(T_decomp, target, atol=1e-10):
            rank_school = n * m * p
            U_s = [[0] * rank_school for _ in range(n * m)]
            V_s = [[0] * rank_school for _ in range(m * p)]
            W_s = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j][col] = 1
                        V_s[j * p + k][col] = 1
                        W_s[k * n + i][col] = 1
                        col += 1
            return {"U": U_s, "V": V_s, "W": W_s}
    except Exception:
        pass
    
    if time.time() > deadline:
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
