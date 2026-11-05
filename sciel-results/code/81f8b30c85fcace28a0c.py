"""Block Strassen decomposition for 3x5x6 matrix multiplication.

Strategy: partition the (i,j,k) index space into 2x2x2 blocks and apply
the rank-7 Strassen decomposition to each block. Blocks along the boundary
that are smaller than 2x2x2 use the schoolbook rank-8 product.

For a 3x5x6 matrix: 
  - i (rows of A): 3 -> 1 block of 2, 1 block of 1
  - j (cols of A): 5 -> 2 blocks of 2, 1 block of 1
  - k (cols of B): 6 -> 3 blocks of 2

Total: 2 (i-blocks) * 3 (j-blocks) * 3 (k-blocks) = 18 blocks
Full 2x2x2 blocks: 1*2*3 = 6 blocks * 7 = 42
Partially small blocks: 12 blocks
  - 1x2x2 blocks: 4 blocks * 4 = 16
  - 1x1x2 blocks: 4 blocks * 2 = 8
  - 2x1x2 blocks: 2 blocks * 4 = 8
  - 2x2x1 blocks: 1 block * 4 = 4
Total rank = 42 + 16 + 8 + 8 + 4 = 78
"""
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

# Strassen 7-term decomposition for 2x2x2 blocks.
# The 7 rank-1 tensors (u, v, w) where u is 2x2 (i,j), v is 2x2 (j,k), w is 2x2 (k,i).
# These satisfy: sum_r u_r * v_r * w_r = matmul_tensor
#
# Using the standard Strassen formulas adapted to the (i,j,k) tensor structure.
# The matmul tensor T[i,j,k] = 1 if i==k, else 0.
#
# Strassen 7 products:
# M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
# M2 = (A10+A11)B00       -> contributes to C10, C11
# M3 = A00(B01-B11)       -> contributes to C01
# M4 = A11(B10-B00)       -> contributes to C10
# M5 = (A00+A01)B10       -> contributes to C00, C10
# M6 = (A10-A00)(B00+B01) -> contributes to C11
# M7 = (A01-A11)(B10+B11) -> contributes to C01, C11

# Each term r has U (i,j), V (j,k), W (k,i) components (all 2x2 matrices).
# T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]

_STRASSEN_222 = [
    # r=0: M1 = (A00+A11)(B00+B11), W: C00+=M1, C11+=M1
    # U: i,j -> (0,0)=1, (1,1)=1
    # V: j,k -> (0,0)=1, (1,1)=1
    # W: k,i -> (0,0)=1, (1,1)=1
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # r=1: M2 = (A10+A11)B00, W: C10+=M2, C11-=M2
    # U: i,j -> (1,0)=1, (1,1)=1
    # V: j,k -> (0,0)=1
    # W: k,i -> (0,1)=1, (1,1)=-1
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
    # r=2: M3 = A00(B01-B11), W: C01+=M3
    # U: i,j -> (0,0)=1
    # V: j,k -> (0,1)=1, (1,1)=-1
    # W: k,i -> (1,0)=1
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1)]),
    # r=3: M4 = A11(B10-B00), W: C10+=M4
    # U: i,j -> (1,1)=1
    # V: j,k -> (1,0)=1, (0,0)=-1
    # W: k,i -> (0,1)=1
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,1,1)]),
    # r=4: M5 = (A00+A01)B10, W: C00+=M5, C10-=M5
    # U: i,j -> (0,0)=1, (0,1)=1
    # V: j,k -> (1,0)=1
    # W: k,i -> (0,0)=1, (0,1)=-1
    ([(0,0,1),(0,1,1)], [(1,0,1)], [(0,0,1),(0,1,-1)]),
    # r=5: M6 = (A10-A00)(B00+B01), W: C11+=M6
    # U: i,j -> (1,0)=1, (0,0)=-1
    # V: j,k -> (0,0)=1, (0,1)=1
    # W: k,i -> (1,1)=1
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
    # r=6: M7 = (A01-A11)(B10+B11), W: C01+=M7, C11-=M7
    # U: i,j -> (0,1)=1, (1,1)=-1
    # V: j,k -> (1,0)=1, (1,1)=1
    # W: k,i -> (1,0)=1, (1,1)=-1
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(1,0,1),(1,1,-1)]),
]


def _verify_block_strassen():
    """Verify the Strassen 2x2x2 decomposition is correct."""
    import numpy as np
    n = m = p = 2
    # Build the 7 rank-1 tensors
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    for r, (u_entries, v_entries, w_entries) in enumerate(_STRASSEN_222):
        for (i, j, val) in u_entries:
            U[i*m+j, r] = val
        for (j, k, val) in v_entries:
            V[j*p+k, r] = val
        for (k, i, val) in w_entries:
            W[k*n+i, r] = val
    
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 if i==k else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(7))
                expected = 1 if i == k else 0
                if abs(val - expected) > 1e-10:
                    return False
    return True


def _schoolbook_block(n, m, p, i0, j0, k0):
    """Generate rank-1 terms for a small block using schoolbook."""
    terms = []
    for di in range(n):
        for dj in range(m):
            for dk in range(p):
                if di == dk:  # Only terms where local i == local k
                    # U has 1 at (i0+di, j0+dj)
                    # V has 1 at (j0+dj, k0+dk)
                    # W has 1 at (k0+dk, i0+di)
                    u = [(i0+di, j0+dj, 1)]
                    v = [(j0+dj, k0+dk, 1)]
                    w = [(k0+dk, i0+di, 1)]
                    terms.append((u, v, w))
    return terms


def _strassen_block(n, m, p, i0, j0, k0):
    """Generate rank-1 terms for a 2x2x2 block using Strassen."""
    terms = []
    for r, (u_entries, v_entries, w_entries) in enumerate(_STRASSEN_222):
        u = [(i0+i, j0+j, val) for (i, j, val) in u_entries]
        v = [(j0+j, k0+k, val) for (j, k, val) in v_entries]
        w = [(k0+k, i0+i, val) for (k, i, val) in w_entries]
        terms.append((u, v, w))
    return terms


def _generate_decomposition():
    """Generate the full block decomposition."""
    n, m, p = 3, 5, 6
    all_terms = []
    
    # Iterate over all blocks
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                bn = min(2, n - i0)
                bm = min(2, m - j0)
                bp = min(2, p - k0)
                
                if bn == 2 and bm == 2 and bp == 2:
                    all_terms.extend(_strassen_block(2, 2, 2, i0, j0, k0))
                else:
                    all_terms.extend(_schoolbook_block(bn, bm, bp, i0, j0, k0))
    
    return all_terms


def _verify_decomposition(all_terms, n=3, m=5, p=6):
    """Verify the decomposition is correct."""
    import numpy as np
    rank = len(all_terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r, (u_entries, v_entries, w_entries) in enumerate(all_terms):
        for (i, j, val) in u_entries:
            U[i*m+j, r] = val
        for (j, k, val) in v_entries:
            V[j*p+k, r] = val
        for (k, i, val) in w_entries:
            W[k*n+i, r] = val
    
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 if i==k else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                expected = 1 if i == k else 0
                if abs(val - expected) > 1e-10:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Verify Strassen block is correct
    if not _verify_block_strassen():
        # Fall back to pure schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Generate block decomposition
    all_terms = _generate_decomposition()
    
    # Verify
    if not _verify_decomposition(all_terms):
        # Fall back to pure schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists of lists
    n, m, p = DIMENSIONS
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(all_terms):
        for (i, j, val) in u_entries:
            U[i * m + j][r] = val
        for (j, k, val) in v_entries:
            V[j * p + k][r] = val
        for (k, i, val) in w_entries:
            W[k * n + i][r] = val
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
