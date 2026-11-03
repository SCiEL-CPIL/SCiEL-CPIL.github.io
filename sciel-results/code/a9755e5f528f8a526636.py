"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Verify a decomposition
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True

    # Build Strassen + schoolbook decomposition
    # Use 6 disjoint 2x2x2 blocks for maximum savings
    # Block positions: (i0, j0, k0) where block covers [i0,i0+1] x [j0,j0+1] x [k0,k0+1]
    
    # Optimal placement to cover as much as possible:
    # i: use (0,1) or (1,2) - pick (0,1) to leave i=2 for schoolbook
    # j: use (0,1) and (2,3) - leaves j=4 for schoolbook  
    # k: use (0,1), (2,3), (4,5) - leaves k=6 for schoolbook
    blocks = [
        (0, 0, 0),  # i:{0,1}, j:{0,1}, k:{0,1}
        (0, 2, 0),  # i:{0,1}, j:{2,3}, k:{0,1}
        (0, 0, 2),  # i:{0,1}, j:{0,1}, k:{2,3}
        (0, 2, 2),  # i:{0,1}, j:{2,3}, k:{2,3}
        (0, 0, 4),  # i:{0,1}, j:{0,1}, k:{4,5}
        (0, 2, 4),  # i:{0,1}, j:{2,3}, k:{4,5}
    ]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i0, j0, k0):
        """Add 7 Strassen terms for a 2x2x2 block at offset (i0, j0, k0)."""
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        U = {(i0, j0): 1, (i0+1, j0+1): 1}
        V = {(j0, k0): 1, (j0+1, k0+1): 1}
        W = {(k0, i0): 1, (k0+1, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M2: (A10+A11)B00 -> C10 - C11
        U = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        V = {(j0, k0): 1}
        W = {(k0, i0+1): 1, (k0+1, i0+1): -1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M3: A00(B01-B11) -> C01 + C11
        U = {(i0, j0): 1}
        V = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        W = {(k0+1, i0): 1, (k0+1, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M4: A11(B10-B00) -> C00 + C10
        U = {(i0+1, j0+1): 1}
        V = {(j0+1, k0): 1, (j0, k0): -1}
        W = {(k0, i0): 1, (k0, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M5: (A00+A01)B11 -> -C00 + C01
        U = {(i0, j0): 1, (i0, j0+1): 1}
        V = {(j0+1, k0+1): 1}
        W = {(k0, i0): -1, (k0+1, i0): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M6: (A10-A00)(B00+B01) -> C11
        U = {(i0+1, j0): 1, (i0, j0): -1}
        V = {(j0, k0): 1, (j0, k0+1): 1}
        W = {(k0+1, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M7: (A01-A11)(B10+B11) -> C00
        U = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        V = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        W = {(k0, i0): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

    # Add all Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0)
    
    # Identify covered (i,j,k) triples
    covered = set()
    for i0, j0, k0 in blocks:
        for i in [i0, i0+1]:
            for j in [j0, j0+1]:
                for k in [k0, k0+1]:
                    covered.add((i, j, k))
    
    # Add schoolbook terms for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U = {(i, j): 1}
                    V = {(j, k): 1}
                    W = {(k, i): 1}
                    U_cols.append(U); V_cols.append(V); W_cols.append(W)
    
    # Convert sparse columns to dense matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (i, j), val in U_cols[c].items():
            U[i * m + j][c] = val
        for (j, k), val in V_cols[c].items():
            V[j * p + k][c] = val
        for (k, i), val in W_cols[c].items():
            W[k * n + i][c] = val
    
    # Verify the decomposition
    U_mat = np.array(U)
    V_mat = np.array(V)
    W_mat = np.array(W)
    
    if verify(U_mat, V_mat, W_mat):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
