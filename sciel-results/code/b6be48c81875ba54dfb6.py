"""Matrix multiplication 3x4x8 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    Order of A elements (rows of U): a00, a01, a10, a11
    Order of B elements (rows of V): b00, b01, b10, b11
    Order of C elements (rows of W): c00, c01, c10, c11
    
    Strassen products:
    m1 = (a00+a11)(b00+b11)
    m2 = (a10+a11)b00
    m3 = a00(b01-b11)
    m4 = a11(b10-b00)
    m5 = (a00+a01)b11
    m6 = (a10-a00)(b00+b01)
    m7 = (a01-a11)(b10+b11)
    
    c00 = m1 + m4 - m5 + m7
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m2 + m3 + m6
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1],    # a01
        [0, 1, 0, 0, 0, 1, 0],    # a10
        [1, 1, 0, 1, 0, 0, -1]    # a11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # b00
        [0, 0, 1, 0, 0, 1, 0],    # b01
        [0, 0, 0, 1, 0, 0, 1],    # b10
        [1, 0, -1, 0, 1, 0, 1]    # b11
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 1, 0, 0, 0],    # c10
        [1, -1, 1, 0, 0, 1, 0]    # c11
    ]
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4).
    
    A is 1x2 (a0, a1), B is 2x2 (b00,b01,b10,b11), C is 1x2 (c0, c1)
    c0 = a0*b00 + a1*b10
    c1 = a0*b01 + a1*b11
    
    U rows (A): a0, a1
    V rows (B): b00, b01, b10, b11
    W rows (C): c0, c1
    """
    U = [
        [1, 0, 1, 0],   # a0: contributes to c0 (via b00) and c1 (via b01)
        [0, 1, 0, 1]    # a1: contributes to c0 (via b10) and c1 (via b11)
    ]
    V = [
        [1, 0, 0, 0],   # b00
        [0, 1, 0, 0],   # b01
        [0, 0, 1, 0],   # b10
        [0, 0, 0, 1]    # b11
    ]
    W = [
        [1, 0, 1, 0],   # c0: gets a0*b00 + a1*b10
        [0, 1, 0, 1]    # c1: gets a0*b01 + a1*b11
    ]
    return U, V, W


def _build_decomposition(tiling):
    """Build a decomposition given a tiling specification.
    
    Each block is (btype, i, js, je, ks, ke) where:
      btype: 'strassen2' or 'school1'
      i: the output row index (0, 1, or 2)
      js, je: j range [js, je) for the 2x2 block in j-dimension
      ks, ke: k range [ks, ke) for the 2x2 block in k-dimension
    """
    n, m, p = 3, 4, 8
    total_rank = 0
    blocks = []
    
    for (btype, i, js, je, ks, ke) in tiling:
        if btype == 'strassen2':
            r = 7
        elif btype == 'school1':
            r = 4
        else:
            raise ValueError("Unknown block type: %s" % btype)
        blocks.append((btype, i, js, je, ks, ke, r))
        total_rank += r
    
    U = [[0] * total_rank for _ in range(n * m)]   # 12 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]   # 32 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]   # 24 x total_rank
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    col_offset = 0
    for (btype, i, js, je, ks, ke, r) in blocks:
        j0 = js
        k0 = ks
        
        if btype == 'strassen2':
            # U2 has 4 rows: a00, a01, a10, a11 -> A[i, j0], A[i, j0+1], A[i+1, j0], A[i+1, j0+1]
            # But wait - in our problem, A is n x m = 3 x 4, so A[i,j] with i in 0..2, j in 0..3
            # The Strassen block computes C[i, k] for a 2x2 subproblem
            # Actually, the Strassen 2x2x2 computes:
            #   C[0,0] = sum_j A[0,j] * B[j,0]
            #   C[0,1] = sum_j A[0,j] * B[j,1]
            #   C[1,0] = sum_j A[1,j] * B[j,0]
            #   C[1,1] = sum_j A[1,j] * B[j,1]
            # where A is 2x2 and B is 2x2
            #
            # In our full problem:
            #   C[i,k] = sum_j A[i,j] * B[j,k]
            # A is 3x4 (i=0..2, j=0..3), B is 4x8 (j=0..3, k=0..7), C is 3x8 (i=0..2, k=0..7)
            #
            # For a Strassen block with fixed i_pair (two consecutive i values) and 
            # j_pair (two consecutive j values) and k_pair (two consecutive k values):
            # It computes C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
            # using A[i0,j0], A[i0,j1], A[i1,j0], A[i1,j1] and B[j0,k0], B[j0,k1], B[j1,k0], B[j1,k1]
            #
            # U rows correspond to A entries: A[i0,j0], A[i0,j1], A[i1,j0], A[i1,j1]
            # In our U matrix, row index is i*m + j
            # So U rows: i0*m+j0, i0*m+j1, i1*m+j0, i1*m+j1
            #
            # V rows correspond to B entries: B[j0,k0], B[j0,k1], B[j1,k0], B[j1,k1]
            # In our V matrix, row index is j*p + k
            # So V rows: j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1
            #
            # W rows correspond to C entries: C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
            # In our W matrix, row index is k*n + i
            # So W rows: k0*n+i0, k1*n+i0, k0*n+i1, k1*n+i1
            
            i0 = i
            i1 = i + 1
            j1 = j0 + 1
            k1 = k0 + 1
            
            u_rows = [i0 * m + j0, i0 * m + j1, i1 * m + j0, i1 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + i0, k1 * n + i0, k0 * n + i1, k1 * n + i1]
            
            for ri in range(4):
                for c in range(7):
                    U[u_rows[ri]][col_offset + c] += U2[ri][c]
                    V[v_rows[ri]][col_offset + c] += V2[ri][c]
                    W[w_rows[ri]][col_offset + c] += W2[ri][c]
            col_offset += 7
            
        elif btype == 'school1':
            # For i=2 (single row), we can't use Strassen (needs 2x2 A block)
            # Use schoolbook: C[2,k] = sum_j A[2,j] * B[j,k]
            # For a 2x2 sub-block in j and k:
            # C[2,k0] += A[2,j0]*B[j0,k0] + A[2,j1]*B[j1,k0]
            # C[2,k1] += A[2,j0]*B[j0,k1] + A[2,j1]*B[j1,k1]
            #
            # U rows: A[2,j0], A[2,j1] -> rows 2*m+j0, 2*m+j1
            # V rows: B[j0,k0], B[j0,k1], B[j1,k0], B[j1,k1] -> j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1
            # W rows: C[2,k0], C[2,k1] -> k0*n+2, k1*n+2
            
            j1 = j0 + 1
            k1 = k0 + 1
            
            u_rows = [2 * m + j0, 2 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + 2, k1 * n + 2]
            
            for ri in range(2):
                for c in range(4):
                    U[u_rows[ri]][col_offset + c] += U1[ri][c]
                    W[w_rows[ri]][col_offset + c] += W1[ri][c]
            for ri in range(4):
                for c in range(4):
                    V[v_rows[ri]][col_offset + c] += V1[ri][c]
            col_offset += 4
    
    return {"U": U, "V": V, "W": W}


def _default_tiling():
    """Default tiling: 
    For i=0,1: use Strassen 2x2x2 blocks over all (j_pair, k_pair) combinations
      j pairs: (0,1), (2,3) -> 2 options
      k pairs: (0,1), (2,3), (4,5), (6,7) -> 4 options
      Total: 2 * 2 * 4 = 16 Strassen blocks? No wait...
      
    Actually, for Strassen we need pairs of i rows. So:
    - i pair (0,1): 1 pair
    - For each i pair, we tile j (4 elements -> 2 pairs) and k (8 elements -> 4 pairs)
    - Total Strassen blocks: 1 * 2 * 4 = 8, each rank 7 = 56
    
    For i=2: use schoolbook 1x2x2 blocks
    - j pairs: (0,1), (2,3) -> 2 options  
    - k pairs: (0,1), (2,3), (4,5), (6,7) -> 4 options
    - Total: 2 * 4 = 8 schoolbook blocks, each rank 4 = 32
    
    Total rank: 56 + 32 = 88
    """
    tiling = []
    
    # Strassen blocks for i pair (0,1)
    for jp in range(2):  # j pairs: (0,1), (2,3)
        for kp in range(4):  # k pairs: (0,1), (2,3), (4,5), (6,7)
            js = 2 * jp
            ks = 2 * kp
            tiling.append(('strassen2', 0, js, js + 2, ks, ks + 2))
    
    # Schoolbook blocks for i=2
    for jp in range(2):
        for kp in range(4):
            js = 2 * jp
            ks = 2 * kp
            tiling.append(('school1', 2, js, js + 2, ks, ks + 2))
    
    return tiling


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    
    For all (i, j, k): sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] should equal 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True


def _schoolbook_full():
    """Full schoolbook decomposition as fallback."""
    n, m, p = 3, 4, 8
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
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Build the Strassen-based decomposition (rank 88)
    tiling = _default_tiling()
    result = _build_decomposition(tiling)
    
    # Verify correctness
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
        result = _schoolbook_full()
    
    return result
