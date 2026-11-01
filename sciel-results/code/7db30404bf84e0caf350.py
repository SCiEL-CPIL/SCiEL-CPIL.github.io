"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140

    U, V, W = _build_optimal(n, m, p)

    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

def _build_optimal(n, m, p):
    """
    Build the rank-53 decomposition:
    - 7 Strassen blocks for columns (0,1), (2,3), ..., (12,13)
    - 1 schoolbook block for column 14
    Total rank = 7*7 + 4 = 53
    """
    n, m, p = 2, 2, 15
    
    # Strassen rank-7 terms for a 2x2x2 block
    # Each term is (U_dict, V_dict, W_dict) where:
    #   U_dict: {(i,j): coeff} for A[i,j]
    #   V_dict: {(j,k): coeff} for B[j,k]
    #   W_dict: {(k,i): coeff} for C[k,i] (trace convention W[k*n+i])
    strassen_terms = [
        # 1: (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # 2: (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1}),
        # 3: A00(B10-B11)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(1,0): 1}),
        # 4: A11(B01-B00)
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # 5: (A00+A01)(B11)
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        # 6: (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # 7: (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Schoolbook terms for a 2x2x1 block
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            schoolbook_terms.append((
                {(i, j): 1},
                {(j, 0): 1},
                {(0, i): 1}
            ))
    
    all_U = []
    all_V = []
    all_W = []
    
    # 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for block_idx in range(7):
        k0 = 2 * block_idx
        k1 = 2 * block_idx + 1
        for (u_dict, v_dict, w_dict) in strassen_terms:
            u_entries = [(i * m + j, c) for (i, j), c in u_dict.items()]
            v_entries = [(j * p + k, c) for (j, k), c in v_dict.items()]
            # Map local k (0 or 1) to global k
            w_entries = []
            for (k_local, i), c in w_dict.items():
                k_global = k0 + k_local
                w_entries.append((k_global * n + i, c))
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)
    
    # 1 schoolbook block for column 14
    k = 14
    for (u_dict, v_dict, w_dict) in schoolbook_terms:
        u_entries = [(i * m + j, c) for (i, j), c in u_dict.items()]
        v_entries = [(j * p + k, c) for (j, k_local), c in v_dict.items()]
        w_entries = [(k * n + i, c) for (k_local, i), c in w_dict.items()]
        all_U.append(u_entries)
        all_V.append(v_entries)
        all_W.append(w_entries)
    
    rank = len(all_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (row, val) in all_U[c]:
            U[row][c] = val
        for (row, val) in all_V[c]:
            V[row][c] = val
        for (row, val) in all_W[c]:
            W[row][c] = val
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Build the schoolbook rank-n*m*p decomposition."""
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition using exact arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = 0
                for c in range(rank):
                    val += U[u_row][c] * V[v_row][c] * W[w_row][c]
                if abs(val - 1) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
