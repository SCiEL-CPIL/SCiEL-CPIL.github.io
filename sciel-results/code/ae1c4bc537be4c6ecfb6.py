"""Search for low-rank decomposition of 4x5x5 matrix multiplication tensor."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify_numpy(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy einsum."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)

def _strassen_decomposition():
    """
    Returns U, V, W for a 2x2x2 block using Strassen's algorithm (rank 7).
    Indices:
    U: (a, b) where a,b in {0,1}
    V: (b, c) where b,c in {0,1}
    W: (c, a) where c,a in {0,1}
    """
    # U entries for A_ab
    U = [
        [1, 0, 1, 0, 1, -1, 0], # a=0, b=0
        [0, 0, 0, 0, 1, 0, 1],  # a=0, b=1
        [0, 1, 0, 0, 0, 1, 0],  # a=1, b=0
        [1, 1, 0, 1, 0, 0, -1]  # a=1, b=1
    ]
    
    # V entries for B_bc
    V = [
        [1, 1, 0, 0, 0, 1, 0], # b=0, c=0
        [0, 0, -1, 1, 0, 1, 0],# b=0, c=1
        [0, 0, 1, -1, 0, 0, 1],# b=1, c=0
        [1, 0, 0, 0, 1, 0, 1]  # b=1, c=1
    ]
    
    # W entries for C_ca
    W = [
        [1, 0, 0, 1, -1, 0, -1], # c=0, a=0
        [0, 0, 1, -1, 1, 0, 0],  # c=0, a=1
        [0, 1, 0, 1, 0, 1, 0],   # c=1, a=0
        [1, -1, 1, 0, 0, 0, 1]   # c=1, a=1
    ]
    
    return U, V, W

def _construct_block_decomposition():
    """
    Constructs a rank 96 decomposition using 4 Strassen blocks and schoolbook for the rest.
    """
    n, m, p = 4, 5, 5
    rank = 96
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_s, V_s, W_s = _strassen_decomposition()
    
    # Define the 4 blocks
    # (i_base, j_base, k_base)
    blocks = [
        (0, 0, 0),
        (0, 2, 2),
        (2, 0, 0),
        (2, 2, 2)
    ]
    
    col_idx = 0
    used_triples = set()
    
    for i_base, j_base, k_base in blocks:
        # Assign 7 columns for this block
        for r in range(7):
            # U: i_base+a, j_base+b
            for a in range(2):
                for b in range(2):
                    i = i_base + a
                    j = j_base + b
                    U[i * m + j, col_idx] = U_s[a][b][r] if False else U_s[a * 2 + b][r] # U_s is list of lists, index by a,b
                    # Wait, U_s is defined as list of 4 rows (a*2+b).
                    # U_s[0] is a=0,b=0. U_s[1] is a=0,b=1. etc.
            
            # Correct indexing for U_s
            # U_s[a][b] is not valid if U_s is flat.
            # Let's re-access U_s correctly.
            # U_s is a list of 4 lists.
            # U_s[a*2+b] is the row for (a,b).
            
            # Reset and do it properly
            pass
        
        # Let's restart the inner loop logic cleanly
        for r in range(7):
            for a in range(2):
                for b in range(2):
                    i = i_base + a
                    j = j_base + b
                    U[i * m + j, col_idx] = U_s[a * 2 + b][r]
                    
            for b in range(2):
                for c in range(2):
                    j = j_base + b
                    k = k_base + c
                    V[j * p + k, col_idx] = V_s[b * 2 + c][r]
                    
            for c in range(2):
                for a in range(2):
                    k = k_base + c
                    i = i_base + a
                    W[k * n + i, col_idx] = W_s[c * 2 + a][r]
                    
            col_idx += 1
            
            # Mark triples as used
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        used_triples.add((i_base + a, j_base + b, k_base + c))
                        
    # Fill in schoolbook for remaining triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used_triples:
                    U[i * m + j, col_idx] = 1
                    V[j * p + k, col_idx] = 1
                    W[k * n + i, col_idx] = 1
                    col_idx += 1
                    
    # Verify the construction
    if _verify_numpy(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try the explicit block decomposition (Rank 96)
    result = _construct_block_decomposition()
    if result is not None:
        return result
        
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
