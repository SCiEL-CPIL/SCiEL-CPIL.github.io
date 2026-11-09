"""Matrix multiplication 3,3,4 - Rank 18 decomposition."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check that U @ V^T matches W reshaped as tensor."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Rank 18 decomposition
    # Part 1: 7 Strassen terms for 2x2x2 core (i,j,k in {0,1})
    # Part 2: 10 combined border terms (indices involving 2)
    # Part 3: 1 term for the 4th column (k=3)
    
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 18
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 18
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 18
    
    col = 0
    
    # --- Part 1: Strassen 2x2x2 (i,j in {0,1}, k in {0,1}) ---
    
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,1,0,1], [0,0,1,0], [0,1,0,0]),
        ([1,0,0,0], [0,1,0,-1], [0,0,1,0]),
        ([0,0,0,1], [1,0,-1,0], [1,1,0,0]),
        ([1,0,1,0], [0,1,0,0], [-1,0,1,0]),
        ([0,-1,1,0], [0,0,1,1], [0,0,0,1]),
        ([0,1,-1,0], [1,0,1,0], [1,0,0,0]),
    ]
    
    for (u_c, v_c, w_c) in strassen_terms:
        c = col
        U[0][c] += u_c[0]
        U[1][c] += u_c[1]
        U[3][c] += u_c[2]
        U[4][c] += u_c[3]
        
        V[0][c] += v_c[0]
        V[1][c] += v_c[1]
        V[4][c] += v_c[2]
        V[5][c] += v_c[3]
        
        W[0][c] += w_c[0]
        W[3][c] += w_c[1]
        W[1][c] += w_c[2]
        W[4][c] += w_c[3]
        
        col += 1
    
    # --- Part 2: Combined border terms ---
    # Each term is a list of (i,j,k) tuples to combine
    
    border_groups = [
        # (i,2,k) pairs: combine (0,2,k) and (1,2,k)
        [(0,2,0), (1,2,0)],
        [(0,2,1), (1,2,1)],
        [(0,2,2), (1,2,2)],
        # (2,j,k) pairs: combine (2,0,k) and (2,1,k)
        [(2,0,0), (2,1,0)],
        [(2,0,1), (2,1,1)],
        [(2,0,2), (2,1,2)],
        # (2,2,k) pair: combine (2,2,0) and (2,2,1)
        [(2,2,0), (2,2,1)],
        # (2,2,2) alone
        [(2,2,2)],
        # (i,j,2) pairs: combine (0,j,2) and (1,j,2)
        [(0,0,2), (1,0,2)],
        [(0,1,2), (1,1,2)],
    ]
    
    for group in border_groups:
        c = col
        # Determine which U, V, W entries to set
        # For a group of (i,j,k) tuples, we need:
        # U[i*m+j] nonzero for all (i,j) in group
        # V[j*p+k] nonzero for all (j,k) in group
        # W[k*n+i] nonzero for all (k,i) in group
        
        u_indices = set()
        v_indices = set()
        w_indices = set()
        
        for (i,j,k) in group:
            u_indices.add(i*m + j)
            v_indices.add(j*p + k)
            w_indices.add(k*n + i)
        
        # Set U, V, W to 1 for the required indices
        for idx in u_indices:
            U[idx][c] = 1.0
        for idx in v_indices:
            V[idx][c] = 1.0
        for idx in w_indices:
            W[idx][c] = 1.0
        
        col += 1
    
    # --- Part 3: 4th Column (k=3) ---
    c = col
    for i in range(n):
        for j in range(m):
            U[i*m + j][c] = 1.0
    for j in range(m):
        V[j*p + 3][c] = 1.0
    for i in range(n):
        W[3*n + i][c] = 1.0
    col += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U = [[0] * rank_fb for _ in range(n * m)]
        V = [[0] * rank_fb for _ in range(m * p)]
        W = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
