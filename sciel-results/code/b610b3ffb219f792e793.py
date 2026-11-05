"""Matrix multiplication 5x5x5 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U, V, W are lists of 7 vectors, each of length 4.
    Indexing: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
    """
    # Standard Strassen 2x2 decomposition
    U = [
        [1, 0, 0, 0],   # A00
        [0, 1, 0, 0],   # A01
        [1, 0, 0, 1],   # A00+A11
        [0, 0, 1, 1],   # A10+A11
        [1, 0, 1, 0],   # A00+A10
        [0, 1, 0, 0],   # A01
        [0, 0, 1, 0],   # A10
    ]
    V = [
        [0, 0, 0, 1],   # B11
        [0, 1, 0, 1],   # B01+B11
        [0, 0, 0, 1],   # B11
        [0, 0, 0, 1],   # B11
        [0, 0, 1, 0],   # B10
        [0, 0, 1, 1],   # B10+B11
        [1, 0, 0, 0],   # B00
    ]
    W = [
        [1, 0, 0, 1],   # C00+C11
        [0, 1, 0, 1],   # C01+C11
        [1, 0, 0, 1],   # C00+C11
        [0, 0, 0, 1],   # C11
        [1, 0, 1, 0],   # C00+C10
        [0, 1, 1, 0],   # C01+C10
        [0, 0, 0, 1],   # C11
    ]
    return U, V, W

def _verify_float(n, m, p, U, V, W, tol=1e-9):
    """Verify using floating point."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = len(U_np[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dot = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(dot - 1.0) > tol:
                    return False
    return True

def _verify_exact(n, m, p, U, V, W):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != Fraction(1):
                    return False
    return True

def _build_block_partition_5x5():
    """
    Build 5x5x5 decomposition using 2+2+1 block partition with Strassen for 2x2x2 blocks.
    
    Partition: split each dimension into blocks of size [2, 2, 1].
    For 2x2x2 sub-blocks: use Strassen (rank 7)
    For other sub-blocks: use schoolbook (rank = product of dimensions)
    
    Total rank: 8*7 + 12*4 + 12*4 + 12*4 + 2*2 + 2*2 + 2*2 + 1 = 56 + 48 + 48 + 48 + 12 = 212
    Wait, let me recount:
    - 8 blocks of 2x2x2: 8 * 7 = 56
    - 12 blocks of 1x2x2 (or permutations): 12 * 4 = 48
    - 12 blocks of 2x1x2: 12 * 4 = 48
    - 12 blocks of 2x2x1: 12 * 4 = 48
    - 2 blocks of 1x1x2: 2 * 2 = 4
    - 2 blocks of 1x2x1: 2 * 2 = 4
    - 2 blocks of 2x1x1: 2 * 2 = 4
    - 1 block of 1x1x1: 1 * 1 = 1
    Total: 56 + 48 + 48 + 48 + 4 + 4 + 4 + 1 = 213? That's not right.
    
    Let me recount more carefully:
    Block sizes: s = [2, 2, 1] for each dimension.
    Number of blocks: 3 * 3 * 3 = 27 total.
    
    For each combination (i,j,k) in {0,1,2}^3:
    - Block size: s[i] x s[j] x s[k]
    - If all sizes are 2: rank 7 (Strassen)
    - Otherwise: rank = s[i] * s[j] * s[k] (schoolbook)
    
    Count:
    - (0,0,0): 2x2x2 -> 7
    - (0,0,1): 2x2x1 -> 4
    - (0,1,0): 2x2x2 -> 7
    - (0,1,1): 2x2x1 -> 4
    - (1,0,0): 2x2x2 -> 7
    - (1,0,1): 2x2x1 -> 4
    - (1,1,0): 2x2x2 -> 7
    - (1,1,1): 2x2x1 -> 4
    - (2,0,0): 1x2x2 -> 4
    - (2,0,1): 1x2x1 -> 2
    - (2,1,0): 1x2x2 -> 4
    - (2,1,1): 1x2x1 -> 2
    - (0,2,0): 2x1x2 -> 4
    - (0,2,1): 2x1x1 -> 2
    - (1,2,0): 2x1x2 -> 4
    - (1,2,1): 2x1x1 -> 2
    - (2,2,0): 1x1x2 -> 2
    - (2,2,1): 1x1x1 -> 1
    - (0,0,2): 2x2x1 -> 4
    - (0,1,2): 2x2x1 -> 4
    - (1,0,2): 2x2x1 -> 4
    - (1,1,2): 2x2x1 -> 4
    - (2,0,2): 1x2x1 -> 2
    - (2,1,2): 1x2x1 -> 2
    - (0,2,2): 2x1x1 -> 2
    - (1,2,2): 2x1x1 -> 2
    - (2,2,2): 1x1x1 -> 1
    
    Wait, I'm confusing myself. Let me use a different approach.
    
    The three blocks in each dimension have sizes [2, 2, 1].
    For block (I, J, K) where I,J,K in {0,1,2}:
    - Size: s[I] x s[J] x s[K] where s = [2, 2, 1]
    
    Let me just compute it directly in code.
    """
    sizes = [2, 2, 1]  # Block sizes for each dimension
    
    U = []
    V = []
    W = []
    
    col = 0
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                si, sj, sk = sizes[I], sizes[J], sizes[K]
                
                # Starting indices
                i_start = sum(sizes[:I])
                j_start = sum(sizes[:J])
                k_start = sum(sizes[:K])
                
                if si == 2 and sj == 2 and sk == 2:
                    # Use Strassen for 2x2x2
                    U2, V2, W2 = _strassen_2x2()
                    for r in range(7):
                        u = [0] * 25
                        v = [0] * 25
                        w = [0] * 25
                        
                        # A indices: i in [i_start, i_start+2), j in [j_start, j_start+2)
                        for ii in range(2):
                            for jj in range(2):
                                a_idx = (i_start + ii) * 5 + (j_start + jj)
                                u[a_idx] = U2[r][ii * 2 + jj]
                        
                        # B indices: j in [j_start, j_start+2), k in [k_start, k_start+2)
                        for jj in range(2):
                            for kk in range(2):
                                b_idx = (j_start + jj) * 5 + (k_start + kk)
                                v[b_idx] = V2[r][jj * 2 + kk]
                        
                        # C indices: k in [k_start, k_start+2), i in [i_start, i_start+2)
                        for kk in range(2):
                            for ii in range(2):
                                c_idx = (k_start + kk) * 5 + (i_start + ii)
                                w[c_idx] = W2[r][ii * 2 + kk]
                        
                        U.append(u)
                        V.append(v)
                        W.append(w)
                        col += 1
                else:
                    # Schoolbook for this block
                    for ii in range(si):
                        for jj in range(sj):
                            for kk in range(sk):
                                u = [0] * 25
                                v = [0] * 25
                                w = [0] * 25
                                
                                a_idx = (i_start + ii) * 5 + (j_start + jj)
                                b_idx = (j_start + jj) * 5 + (k_start + kk)
                                c_idx = (k_start + kk) * 5 + (i_start + ii)
                                
                                u[a_idx] = 1
                                v[b_idx] = 1
                                w[c_idx] = 1
                                
                                U.append(u)
                                V.append(v)
                                W.append(w)
                                col += 1
    
    return U, V, W, len(U)

def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Fallback: Schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Strategy: 2+2+1 block partition with Strassen for 2x2x2 sub-blocks
    try:
        U5, V5, W5, total_rank = _build_block_partition_5x5()
        
        # Verify
        if _verify_float(n, m, p, U5, V5, W5):
            if _verify_exact(n, m, p, U5, V5, W5):
                if total_rank < best_rank:
                    best = {"U": U5, "V": V5, "W": W5}
                    best_rank = total_rank
    except Exception as e:
        pass
    
    return best

# EVOLVE-BLOCK-END
