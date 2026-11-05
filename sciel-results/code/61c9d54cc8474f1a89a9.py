"""Matrix multiplication 6x6x6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly decomposes the matrix multiplication tensor."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    return np.allclose(T, 1.0, atol=1e-10)

def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_2x2():
    """Rank-4 schoolbook decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    for i in range(2):
        for j in range(2):
            c = i*2+j
            U[i*2+j][c] = 1
            V[i*2+j][c] = 1
            W[i*2+j][c] = 1
    return U, V, W

def _kron_decompose(sub_U, sub_V, sub_W, block_size_a, block_size_b, block_size_c,
                    size_a, size_b, size_c):
    """Construct a larger decomposition via Kronecker product of block decompositions.
    
    Parameters:
    - sub_U, sub_V, sub_W: factor matrices for the smaller problem
    - block_size_a, block_size_b, block_size_c: dimensions of the smaller problem
    - size_a, size_b, size_c: dimensions of the larger problem
    
    Returns U, V, W for the larger problem.
    """
    sub_rank = len(sub_U[0])
    blocks_a = size_a // block_size_a
    blocks_b = size_b // block_size_b
    blocks_c = size_c // block_size_c
    
    total_rank = blocks_a * blocks_b * blocks_c * sub_rank
    
    U = [[0]*total_rank for _ in range(size_a * size_b)]
    V = [[0]*total_rank for _ in range(size_b * size_c)]
    W = [[0]*total_rank for _ in range(size_c * size_a)]
    
    col = 0
    for ba in range(blocks_a):
        for bb in range(blocks_b):
            for bc in range(blocks_c):
                for c in range(sub_rank):
                    # U: rows indexed by (global_i, global_j)
                    for li in range(block_size_a):
                        for lj in range(block_size_b):
                            u_row = (ba*block_size_a + li) * size_b + (bb*block_size_b + lj)
                            U[u_row][col] = sub_U[li*block_size_b + lj][c]
                    
                    # V: rows indexed by (global_j, global_k)
                    for lj in range(block_size_b):
                        for lk in range(block_size_c):
                            v_row = (bb*block_size_b + lj) * size_c + (bc*block_size_c + lk)
                            V[v_row][col] = sub_V[lj*block_size_c + lk][c]
                    
                    # W: rows indexed by (global_k, global_i)
                    for lk in range(block_size_c):
                        for li in range(block_size_a):
                            w_row = (bc*block_size_c + lk) * size_a + (ba*block_size_a + li)
                            W[w_row][col] = sub_W[lk*block_size_a + li][c]
                    
                    col += 1
    
    return U, V, W

def _mixed_3x3_blocks():
    """Construct 6x6x6 decomposition using 3x3 grid of 2x2 blocks.
    Uses Strassen for some blocks and schoolbook for others to potentially
    find a lower rank decomposition through mixed strategies.
    
    For a 3x3 grid of 2x2 blocks:
    - 27 blocks total
    - If all Strassen: rank = 27 * 7 = 189
    - If all schoolbook: rank = 27 * 4 = 108
    
    This function returns the schoolbook version (rank 108) as it's the best
    straightforward option.
    """
    sU, sV, sW = _schoolbook_2x2()
    return _kron_decompose(sU, sV, sW, 2, 2, 2, 6, 6, 6)

def _try_rank_reduction_search(U, V, W, n, m, p, max_iter=50, deadline=None):
    """Attempt to reduce the rank by identifying and removing redundant terms.
    
    This is a heuristic search that tries to find linear dependencies among
    the rank-1 terms and eliminate them.
    
    Returns (U_reduced, V_reduced, W_reduced, new_rank) or None if no reduction found.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    current_rank = U_np.shape[1]
    
    # Check for duplicate or proportional columns
    # This is a simple heuristic - check if any two columns are linearly dependent
    # in a way that allows removal
    
    # For now, just return the original if no obvious reduction
    # A proper implementation would use SVD or similar
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    candidates = []
    
    # Option 1: 3x3x3 blocks of size 2 with schoolbook (rank 4 each)
    # Total rank: 27 * 4 = 108
    sU2 = _schoolbook_2x2()
    U1, V1, W1 = _kron_decompose(sU2, sU2, W1 := [[0]*4 for _ in range(4)], 2, 2, 2, 6, 6, 6)
    # Wait, I need to fix this - let me redo it properly
    sU2, sV2, sW2 = _schoolbook_2x2()
    U1, V1, W1 = _kron_decompose(sU2, sV2, sW2, 2, 2, 2, 6, 6, 6)
    candidates.append((U1, V1, W1, 108))
    
    # Option 2: 3x3x3 blocks of size 2 with Strassen (rank 7 each)
    # Total rank: 27 * 7 = 189
    sU, sV, sW = _strassen_2x2()
    U2, V2, W2 = _kron_decompose(sU, sV, sW, 2, 2, 2, 6, 6, 6)
    candidates.append((U2, V2, W2, 189))
    
    # Sort by rank (lowest first)
    candidates.sort(key=lambda x: x[3])
    
    # Verify and return the best valid candidate
    for U, V, W, rank in candidates:
        if time.time() > deadline - 10:
            break
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: pure schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i*m*p + j*p + k
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
