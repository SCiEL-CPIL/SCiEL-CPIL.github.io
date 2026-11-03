"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j], V[i*2+j], W[i*2+j]
    """
    # Strassen coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Map (i,j) to index
    def idx(i, j):
        return i * 2 + j

    # M1
    U[idx(0,0), 0] += 1
    U[idx(1,1), 0] += 1
    V[idx(0,0), 0] += 1
    V[idx(1,1), 0] += 1
    W[idx(0,0), 0] += 1
    W[idx(1,1), 0] += 1
    W[idx(0,1), 0] -= 1
    W[idx(1,0), 0] -= 1
    W[idx(1,1), 0] += 1 # Wait, C11 = M1 + M4 - M5 + M7.
    # Let's be precise.
    
    # Reset W
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11

    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11

    # M3: A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11

    # M4: A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10

    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01

    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11

    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00
    W[3, 6] = 1  # C11
    
    # Convert to lists
    return U.tolist(), V.tolist(), W.tolist()

def _trivial_block(n, m, p):
    """
    Trivial decomposition for n x m times m x p.
    Returns U (nm x rank), V (mp x rank), W (pn x rank).
    """
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

def _combine_blocks(blocks, n, m, p):
    """
    Combine a list of blocks.
    Each block is a dict with 'U', 'V', 'W', 'offsets' (i, j, k)
    """
    total_rank = sum(b['rank'] for b in blocks)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for b in blocks:
        bi, bj, bk = b['offsets']
        bn, bm, bp = b['dims']
        bU, bV, bW = b['U'], b['V'], b['W']
        
        for i in range(bn):
            for j in range(bm):
                for r in range(b['rank']):
                    U[(bi + i) * m + (bj + j)][col_offset + r] = bU[i * bm + j][r]
                    
        for j in range(bm):
            for k in range(bp):
                for r in range(b['rank']):
                    V[(bj + j) * p + (bk + k)][col_offset + r] = bV[j * bp + k][r]
                    
        for k in range(bp):
            for i in range(bn):
                for r in range(b['rank']):
                    W[(bk + k) * n + (bi + i)][col_offset + r] = bW[k * bn + i][r]
                    
        col_offset += b['rank']
        
    return U, V, W

def _construct_improved():
    n, m, p = 4, 5, 5
    
    # Split n=4 into 2+2
    # Split m=5 into 2+2+1
    # Split p=5 into 2+2+1
    
    # Blocks for A (4x5):
    # A0: 2x2, A1: 2x2, A2: 2x1
    # A3: 2x2, A4: 2x2, A5: 2x1
    
    # Blocks for B (5x5):
    # B00: 2x2, B01: 2x2, B02: 2x1
    # B10: 2x2, B11: 2x2, B12: 2x1
    # B20: 1x2, B21: 1x2, B22: 1x1
    
    # We compute C = A B
    # C is 4x5.
    # C0 (2x5) = A0 B0 + A1 B1 + A2 B2
    # C1 (2x5) = A3 B0 + A4 B1 + A5 B2
    
    # We need to decompose each product.
    # Product of (2x2) * (2x2) -> 2x2. Rank 7 (Strassen).
    # Product of (2x2) * (2x1) -> 2x1. Rank 4 (Trivial).
    # Product of (2x1) * (1x2) -> 2x2. Rank 4 (Trivial).
    # Product of (2x1) * (1x1) -> 2x1. Rank 2 (Trivial).
    
    # We will build the decomposition for C0 and C1 separately.
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    blocks_C0 = []
    blocks_C1 = []
    
    # C0 = A0 B0 + A1 B1 + A2 B2
    # A0 B0: 2x2 * 2x2. Offset (0,0,0).
    blocks_C0.append({
        'U': strassen_U, 'V': strassen_V, 'W': strassen_W,
        'dims': (2, 2, 2), 'offsets': (0, 0, 0), 'rank': 7
    })
    # A1 B1: 2x2 * 2x2. Offset (0,2,2).
    blocks_C0.append({
        'U': strassen_U, 'V': strassen_V, 'W': strassen_W,
        'dims': (2, 2, 2), 'offsets': (0, 2, 2), 'rank': 7
    })
    # A2 B2: 2x1 * 1x2. Offset (0,4,4).
    U_triv, V_triv, W_triv = _trivial_block(2, 1, 2)
    blocks_C0.append({
        'U': U_triv, 'V': V_triv, 'W': W_triv,
        'dims': (2, 1, 2), 'offsets': (0, 4, 4), 'rank': 4
    })
    
    # C1 = A3 B0 + A4 B1 + A5 B2
    # A3 B0: 2x2 * 2x2. Offset (2,0,0).
    blocks_C1.append({
        'U': strassen_U, 'V': strassen_V, 'W': strassen_W,
        'dims': (2, 2, 2), 'offsets': (2, 0, 0), 'rank': 7
    })
    # A4 B1: 2x2 * 2x2. Offset (2,2,2).
    blocks_C1.append({
        'U': strassen_U, 'V': strassen_V, 'W': strassen_W,
        'dims': (2, 2, 2), 'offsets': (2, 2, 2), 'rank': 7
    })
    # A5 B2: 2x1 * 1x2. Offset (2,4,4).
    U_triv, V_triv, W_triv = _trivial_block(2, 1, 2)
    blocks_C1.append({
        'U': U_triv, 'V': V_triv, 'W': W_triv,
        'dims': (2, 1, 2), 'offsets': (2, 4, 4), 'rank': 4
    })
    
    # Now we need to handle the other columns of C.
    # C0 is 2x5. We handled cols 0,1 (from B0), 2,3 (from B1), 4 (from B2).
    # Wait, B0 is 2x2 (cols 0,1). B1 is 2x2 (cols 2,3). B2 is 2x1 (col 4).
    # So C0 covers all 5 columns.
    # C1 covers all 5 columns.
    
    # We combine C0 and C1 blocks.
    all_blocks = blocks_C0 + blocks_C1
    
    U, V, W = _combine_blocks(all_blocks, 4, 5, 5)
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    try:
        result = _construct_improved()
    except Exception:
        # Fallback to trivial
        n, m, p = 4, 5, 5
        total_rank = n * m * p
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}
        
    return result

# EVOLVE-BLOCK-END
