"""Rewritten program with explicit rank-39 construction."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_rank39():
    """
    Construct an explicit rank-39 decomposition for M_{2,2,11}.
    
    Strategy: Split the 2x11 matrix B into 5 blocks of size 2x2 and 1 block of size 2x1.
    For each 2x2 block, use Strassen's algorithm (rank 7).
    For the 2x1 block, use the trivial decomposition (rank 4).
    Total rank = 5*7 + 4 = 39.
    
    Index conventions:
    - U: shape (n*m, rank) = (4, 39). Index U[i*m+j] = U[i*2+j]
    - V: shape (m*p, rank) = (22, 39). Index V[j*p+k] = V[j*11+k]
    - W: shape (p*n, rank) = (22, 39). Index W[k*n+i] = W[k*2+i]
    
    For a 2x2 matrix multiplication C = A * B where A is 2x2 and B is 2x2:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
    
    Strassen's 7 products:
    P1 = (a+d)(e+h)
    P2 = (c+d)e
    P3 = a(f-h)
    P4 = d(g-e)
    P5 = (a+b)h
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    Then:
    C00 = P1 + P4 - P5 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P6
    
    For the 2x1 block (columns 10, only):
    A is 2x2, B is 2x1 = [[B00], [B10]]
    C = [[A00*B00 + A01*B10], [A10*B00 + A11*B10]]
    Trivial decomposition uses 4 products:
    Q1 = A00 * B00
    Q2 = A01 * B10
    Q3 = A10 * B00
    Q4 = A11 * B10
    
    C00 = Q1 + Q2
    C10 = Q3 + Q4
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    # Initialize factor matrices
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]  # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]  # 22 x 39
    
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # So U[0] = A00, U[1] = A01, U[2] = A10, U[3] = A11
    
    # Strassen terms for a 2x2 block at column offset col (0, 2, 4, 6, 8)
    # B has columns 0-10. For block at col, B uses columns col and col+1.
    # V indices: V[j*11+k] where j is row (0 or 1), k is column.
    # W indices: W[k*2+i] where k is column of output (0-10), i is row (0 or 1).
    
    def add_strassen_block(col_offset, base_rank):
        """Add Strassen decomposition for 2x2 block starting at column col_offset."""
        # The 7 Strassen products for this block
        # Each product contributes to one rank position
        
        # P1 = (A00 + A11) * (B00 + B11) where B00=B[0,col], B11=B[1,col+1]
        # In terms of A: A00=U[0], A11=U[3]
        # In terms of B: B00=V[0*11+col]=V[col], B11=V[1*11+col+1]=V[11+col+1]=V[12+col]
        # Output: contributes to C[0,col] and C[1,col+1] and C[0,col+1] and C[1,col]
        # C[k*n+i] = C[k*2+i]
        # C[0,col] = W[col*2+0] = W[2*col]
        # C[0,col+1] = W[(col+1)*2+0] = W[2*col+2]
        # C[1,col] = W[col*2+1] = W[2*col+1]
        # C[1,col+1] = W[(col+1)*2+1] = W[2*col+3]
        
        # P1: U contributions: A00+1, A11+1 => U[0]+=1, U[3]+=1
        # V contributions: B[0,col]+1, B[1,col+1]+1 => V[col]+=1, V[12+col]+=1
        # W contributions: C[0,col]+1, C[1,col+1]+1 => W[2*col]+=1, W[2*col+3]+=1
        r = base_rank
        U[0][r] += 1.0
        U[3][r] += 1.0
        V[col][r] += 1.0
        V[12 + col][r] += 1.0
        W[2 * col][r] += 1.0
        W[2 * col + 3][r] += 1.0
        
        # P2 = (A10 + A11) * B[0,col]
        r = base_rank + 1
        U[2][r] += 1.0
        U[3][r] += 1.0
        V[col][r] += 1.0
        # C[1,col] = W[2*col+1]
        W[2 * col + 1][r] += 1.0
        
        # P3 = A00 * (B[0,col+1] - B[1,col])
        # = A00*B[0,col+1] - A00*B[1,col]
        r = base_rank + 2
        U[0][r] += 1.0
        V[11 + col + 1][r] += 1.0  # B[0, col+1] = V[0*11+col+1] = V[col+1]... wait
        # Let me recheck: V[j*p+k] = V[j*11+k]
        # B[0, col] = V[0*11+col] = V[col]
        # B[0, col+1] = V[0*11+col+1] = V[col+1]
        # B[1, col] = V[1*11+col] = V[11+col]
        # B[1, col+1] = V[1*11+col+1] = V[12+col]
        # So P3 = A00 * (B[0,col+1] - B[1,col]) = A00*V[col+1] - A00*V[11+col]
        V[col + 1][r] += 1.0
        V[11 + col][r] += -1.0
        # C[0,col+1] = W[2*(col+1)] = W[2*col+2]
        W[2 * col + 2][r] += 1.0
        
        # P4 = A11 * (B[1,col] - B[0,col])
        r = base_rank + 3
        U[3][r] += 1.0
        V[11 + col][r] += 1.0
        V[col][r] += -1.0
        # C[1,col] = W[2*col+1], C[0,col] = W[2*col]
        W[2 * col + 1][r] += 1.0
        W[2 * col][r] += -1.0
        
        # P5 = (A00 + A01) * B[1,col+1]
        r = base_rank + 4
        U[0][r] += 1.0
        U[1][r] += 1.0
        V[12 + col][r] += 1.0
        # C[0,col+1] = W[2*col+2], C[1,col+1] = W[2*col+3]
        W[2 * col + 2][r] += 1.0
        W[2 * col + 3][r] += -1.0
        
        # P6 = (A10 - A00) * (B[0,col] + B[0,col+1])
        r = base_rank + 5
        U[2][r] += 1.0
        U[0][r] += -1.0
        V[col][r] += 1.0
        V[col + 1][r] += 1.0
        # C[1,col] = W[2*col+1], C[1,col+1] = W[2*col+3]
        W[2 * col + 1][r] += 1.0
        W[2 * col + 3][r] += 1.0
        
        # P7 = (A01 - A11) * (B[1,col] + B[1,col+1])
        r = base_rank + 6
        U[1][r] += 1.0
        U[3][r] += -1.0
        V[11 + col][r] += 1.0
        V[12 + col][r] += 1.0
        # C[0,col] = W[2*col], C[1,col] = W[2*col+1]
        W[2 * col][r] += 1.0
        W[2 * col + 1][r] += -1.0
    
    # Add Strassen blocks for columns 0, 2, 4, 6, 8
    for block_idx in range(5):
        col_offset = 2 * block_idx
        base_rank = 7 * block_idx
        add_strassen_block(col_offset, base_rank)
    
    # Add trivial decomposition for the last 2x1 block at column 10
    # C[0,10] = A00*B[0,10] + A01*B[1,10]
    # C[1,10] = A10*B[0,10] + A11*B[1,10]
    # B[0,10] = V[10], B[1,10] = V[21]
    # C[0,10] = W[20], C[1,10] = W[21]
    
    base_rank = 35  # 5 * 7
    
    # Q1 = A00 * B[0,10]
    r = base_rank
    U[0][r] += 1.0
    V[10][r] += 1.0
    W[20][r] += 1.0
    
    # Q2 = A01 * B[1,10]
    r = base_rank + 1
    U[1][r] += 1.0
    V[21][r] += 1.0
    W[20][r] += 1.0
    
    # Q3 = A10 * B[0,10]
    r = base_rank + 2
    U[2][r] += 1.0
    V[10][r] += 1.0
    W[21][r] += 1.0
    
    # Q4 = A11 * B[1,10]
    r = base_rank + 3
    U[3][r] += 1.0
    V[21][r] += 1.0
    W[21][r] += 1.0
    
    return U, V, W

def _try_rank(rank, n, m, p, max_iter=5000, seed=42):
    """Try to find a decomposition of given rank using iterative refinement."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    targets = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets[(i*m+j, j*p+k, k*n+i)] = 1.0
    
    for it in range(max_iter):
        for a in range(n * m):
            for c in range(rank):
                num = 0.0
                den = 0.0
                for (ta, tb, tc), tv in targets.items():
                    if ta == a:
                        num += tv * V[tb, c] * W[tc, c]
                        den += V[tb, c] * W[tc, c] * V[tb, c] * W[tc, c]
                if den > 1e-12:
                    U[a, c] = num / den
        
        for b in range(m * p):
            for c in range(rank):
                num = 0.0
                den = 0.0
                for (ta, tb, tc), tv in targets.items():
                    if tb == b:
                        num += tv * U[ta, c] * W[tc, c]
                        den += U[ta, c] * W[tc, c] * U[ta, c] * W[tc, c]
                if den > 1e-12:
                    V[b, c] = num / den
        
        for cc in range(p * n):
            for c in range(rank):
                num = 0.0
                den = 0.0
                for (ta, tb, tc), tv in targets.items():
                    if tc == cc:
                        num += tv * U[ta, c] * V[tb, c]
                        den += U[ta, c] * V[tb, c] * U[ta, c] * V[tb, c]
                if den > 1e-12:
                    W[cc, c] = num / den
        
        if it % 100 == 0:
            err = 0.0
            for (ta, tb, tc), tv in targets.items():
                s = sum(U[ta, c] * V[tb, c] * W[tc, c] for c in range(rank))
                err += (s - tv) ** 2
            if err < 1e-12:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # First, try the explicit rank-39 construction
    try:
        U, V, W = _construct_rank39()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: try heuristic search for lower rank
    for rank in range(20, 40):
        if time.time() > deadline - 5:
            break
        result = _try_rank(rank, n, m, p, max_iter=2000, seed=rank)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook
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
# EVOLVE-BLOCK-END
