"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x2 matrix by a 2x9 matrix.
    
    Uses Strassen's algorithm for the 2x2x2 blocks and trivial decomposition
    for the 2x2x1 blocks.
    
    The multiplication can be viewed as:
    - 4 blocks of size 2x2 multiplied by 2x2 blocks (for columns 0-3, 4-7)
    - Wait, let me reconsider.
    
    A is 2x2, B is 2x9. C is 2x9.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can group the 9 columns of B:
    - Columns 0,1,2,3: first 4 columns -> two 2-column blocks
    - Columns 4,5,6,7: next 4 columns -> two 2-column blocks  
    - Columns 8: last column -> one 1-column block
    
    Actually, let me think more carefully. The standard approach:
    
    For each pair of output columns (k, k+1), we can use Strassen if we treat
    the 2x2 submatrix of B.
    
    B is 2x9. We can partition into:
    - B[:, 0:2] (2x2)
    - B[:, 2:4] (2x2)
    - B[:, 4:6] (2x2)
    - B[:, 6:8] (2x2)
    - B[:, 8:9] (2x1)
    
    For each 2x2 block, we multiply 2x2 by 2x2 using Strassen (rank 7)
    For the 2x1 block, we multiply 2x2 by 2x1 using trivial (rank 4)
    
    Total rank: 4 * 7 + 1 * 4 = 28 + 4 = 32
    
    But wait - the output C is 2x9, and we need W[k*n+i, r] = W[k*2+i, r].
    
    Let me build this carefully.
    
    Strassen for 2x2 * 2x2:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen's 7 products:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For each block, the U, V, W matrices have 7 columns.
    
    Let me define the index conventions:
    - U has shape (n*m, rank) = (4, rank), indexed by (i,j) -> i*2+j
    - V has shape (m*p, rank) = (18, rank), indexed by (j,k) -> j*9+k
    - W has shape (p*n, rank) = (18, rank), indexed by (k,i) -> k*2+i
    
    For Strassen block covering columns k0, k0+1:
    A elements: A[0,0], A[0,1], A[1,0], A[1,1]
    B elements: B[0,k0], B[0,k0+1], B[1,k0], B[1,k0+1]
    
    a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
    e = B[0,k0], f = B[0,k0+1], g = B[1,k0], h = B[1,k0+1]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C[0,k0]   = M1 + M4 - M5 + M7
    C[0,k0+1] = M3 + M5
    C[1,k0]   = M2 + M4
    C[1,k0+1] = M1 - M2 + M3 + M6
    
    Now let me encode each M as a rank-1 term.
    
    M1 = (a+d)(e+h) = (A[0,0]+A[1,1]) * (B[0,k0]+B[0,k0+1])
    U: A[0,0]=1, A[1,1]=1
    V: B[0,k0]=1, B[0,k0+1]=1
    W: contributes to C[0,k0] with +1, C[0,k0+1] with +1, C[1,k0] with 0, C[1,k0+1] with +1
    Wait, let me re-check:
    C[0,k0] = M1 + M4 - M5 + M7
    C[0,k0+1] = M3 + M5
    C[1,k0] = M2 + M4
    C[1,k0+1] = M1 - M2 + M3 + M6
    
    So M1 appears in:
    C[0,k0] with coeff +1
    C[1,k0+1] with coeff +1
    
    M2 = (c+d)e = (A[1,0]+A[1,1]) * B[0,k0]
    Appears in:
    C[1,k0] with coeff +1
    C[1,k0+1] with coeff -1
    
    M3 = a(f-h) = A[0,0] * (B[0,k0+1]-B[1,k0+1])
    Appears in:
    C[0,k0+1] with coeff +1
    C[1,k0+1] with coeff +1
    
    M4 = d(g-e) = A[1,1] * (B[1,k0]-B[0,k0])
    Appears in:
    C[0,k0] with coeff +1
    C[1,k0] with coeff +1
    
    M5 = (a+b)h = (A[0,0]+A[0,1]) * B[1,k0+1]
    Appears in:
    C[0,k0] with coeff -1
    C[0,k0+1] with coeff +1
    
    M6 = (c-a)(e+f) = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k0+1])
    Appears in:
    C[1,k0+1] with coeff +1
    
    M7 = (b-d)(g+h) = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k0+1])
    Appears in:
    C[0,k0] with coeff +1
    """
    
    n, m, p = DIMENSIONS  # 2, 2, 9
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 18 x 32
    W = [[0] * rank for _ in range(p * n)]  # 18 x 32
    
    col = 0
    
    # Process 4 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7)
    for block in range(4):
        k0 = block * 2
        k1 = k0 + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[0,k1])
        U[0][col] = 1  # A[0,0]
        U[3][col] = 1  # A[1,1]
        V[0 * p + k0][col] = 1  # B[0,k0]
        V[0 * p + k1][col] = 1  # B[0,k1]
        W[k0 * n + 0][col] = 1  # C[0,k0]
        W[k1 * n + 1][col] = 1  # C[1,k1]
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[2][col] = 1  # A[1,0]
        U[3][col] = 1  # A[1,1]
        V[0 * p + k0][col] = 1  # B[0,k0]
        W[k0 * n + 1][col] = 1  # C[1,k0]
        W[k1 * n + 1][col] = -1  # C[1,k1]
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0][col] = 1  # A[0,0]
        V[0 * p + k1][col] = 1  # B[0,k1]
        V[1 * p + k1][col] = -1  # B[1,k1]
        W[k1 * n + 0][col] = 1  # C[0,k1]
        W[k1 * n + 1][col] = 1  # C[1,k1]
        col += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[3][col] = 1  # A[1,1]
        V[1 * p + k0][col] = 1  # B[1,k0]
        V[0 * p + k0][col] = -1  # B[0,k0]
        W[k0 * n + 0][col] = 1  # C[0,k0]
        W[k0 * n + 1][col] = 1  # C[1,k0]
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0][col] = 1  # A[0,0]
        U[1][col] = 1  # A[0,1]
        V[1 * p + k1][col] = 1  # B[1,k1]
        W[k0 * n + 0][col] = -1  # C[0,k0]
        W[k1 * n + 0][col] = 1  # C[0,k1]
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[2][col] = 1  # A[1,0]
        U[0][col] = -1  # A[0,0]
        V[0 * p + k0][col] = 1  # B[0,k0]
        V[0 * p + k1][col] = 1  # B[0,k1]
        W[k1 * n + 1][col] = 1  # C[1,k1]
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1][col] = 1  # A[0,1]
        U[3][col] = -1  # A[1,1]
        V[1 * p + k0][col] = 1  # B[1,k0]
        V[1 * p + k1][col] = 1  # B[1,k1]
        W[k0 * n + 0][col] = 1  # C[0,k0]
        col += 1
    
    # Process the last single column (k=8) with trivial 2x2 * 2x1
    # C[0,8] = A[0,0]*B[0,8] + A[0,1]*B[1,8]
    # C[1,8] = A[1,0]*B[0,8] + A[1,1]*B[1,8]
    
    k8 = 8
    
    # Term 1: A[0,0] * B[0,8] -> C[0,8]
    U[0][col] = 1
    V[0 * p + k8][col] = 1
    W[k8 * n + 0][col] = 1
    col += 1
    
    # Term 2: A[0,1] * B[1,8] -> C[0,8]
    U[1][col] = 1
    V[1 * p + k8][col] = 1
    W[k8 * n + 0][col] = 1
    col += 1
    
    # Term 3: A[1,0] * B[0,8] -> C[1,8]
    U[2][col] = 1
    V[0 * p + k8][col] = 1
    W[k8 * n + 1][col] = 1
    col += 1
    
    # Term 4: A[1,1] * B[1,8] -> C[1,8]
    U[3][col] = 1
    V[1 * p + k8][col] = 1
    W[k8 * n + 1][col] = 1
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
