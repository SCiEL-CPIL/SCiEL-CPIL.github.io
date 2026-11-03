import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W.
    Strategy:
    1. Split middle dimension m=3 into m1=2 and m2=1.
    2. Part A (m1=2): A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8]
       Split B into 4 blocks of width 2.
       Each block is a 2x2 * 2x2 multiplication.
       Use Strassen's algorithm (rank 7) for each.
       Total rank for Part A: 4 * 7 = 28.
    3. Part B (m2=1): A[0:2, 2] * B[2, 0:8] -> C[0:2, 0:8]
       This is 2x1 * 1x8.
       Requires 16 scalar multiplications (schoolbook).
       Total rank for Part B: 16.
    4. Total Rank: 28 + 16 = 44.
    """
    start_time = time.time()
    deadline = start_time + 110 # 10 second safety margin

    n, m, p = DIMENSIONS
    rank = 44
    
    # Initialize matrices as numpy arrays for easier construction, then convert to lists
    # Shapes: U(n*m, rank), V(m*p, rank), W(p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to check deadline
    def check_deadline():
        if time.time() > deadline:
            return True
        return False

    # Part 1: 4 blocks of Strassen for A[0:2, 0:2] * B[0:2, :]
    # A indices:
    # A[0,0]=0, A[0,1]=1
    # A[1,0]=3, A[1,1]=4
    # B indices for block k (columns 2k, 2k+1):
    # B[0, 2k]=2k, B[0, 2k+1]=2k+1
    # B[1, 2k]=8+2k, B[1, 2k+1]=8+2k+1
    # W indices for C[0:2, 2k:2k+2]:
    # C[i, j] -> W[j*n + i] = W[2j + i]
    # C[0, 2k]=2(2k), C[0, 2k+1]=2(2k+1)
    # C[1, 2k]=2(2k)+1, C[1, 2k+1]=2(2k+1)+1

    for blk in range(4):
        if check_deadline():
            # Fallback to schoolbook if time runs out (should not happen for rank 44 construction)
            return _schoolbook_fallback(n, m, p)
            
        bo = 2 * blk
        # V indices
        ve = bo       # B[0, bo]
        vf = bo + 1   # B[0, bo+1]
        vg = 8 + bo   # B[1, bo]
        vh = 8 + bo + 1 # B[1, bo+1]
        
        # W indices
        w00 = 2 * bo       # C[0, bo]
        w01 = 2 * bo + 2   # C[0, bo+1]
        w10 = 2 * bo + 1   # C[1, bo]
        w11 = 2 * bo + 3   # C[1, bo+1]
        
        # Strassen Formulas
        # P1 = (A00 + A11) * (B00 + B11)
        # Contributes to C00 (+1), C11 (+1)
        # Note: Standard Strassen for C = AB:
        # C00 = P1 - P5 - P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # P1
        U[0, col] += 1
        U[4, col] += 1
        V[ve, col] += 1
        V[vh, col] += 1
        W[w00, col] += 1
        W[w11, col] += 1
        col += 1
        
        # P2 = (A10 + A11) * B00
        # Contributes to C10 (+1), C11 (-1)
        U[3, col] += 1
        U[4, col] += 1
        V[ve, col] += 1
        W[w10, col] += 1
        W[w11, col] -= 1
        col += 1
        
        # P3 = A00 * (B01 - B11)
        # Contributes to C01 (+1), C11 (+1)
        U[0, col] += 1
        V[vf, col] += 1
        V[vh, col] -= 1
        W[w01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # P4 = A11 * (B10 - B00)
        # Contributes to C00 (+1), C10 (+1)
        U[4, col] += 1
        V[vg, col] += 1
        V[ve, col] -= 1
        W[w00, col] += 1
        W[w10, col] += 1
        col += 1
        
        # P5 = (A00 + A01) * B11
        # Contributes to C00 (-1), C01 (+1)
        U[0, col] += 1
        U[1, col] += 1
        V[vh, col] += 1
        W[w00, col] -= 1
        W[w01, col] += 1
        col += 1
        
        # P6 = (A10 - A00) * (B00 + B01)
        # Contributes to C11 (+1)
        U[3, col] += 1
        U[0, col] -= 1
        V[ve, col] += 1
        V[vf, col] += 1
        W[w11, col] += 1
        col += 1
        
        # P7 = (A01 - A11) * (B10 + B11)
        # Contributes to C00 (+1)
        U[1, col] += 1
        U[4, col] -= 1
        V[vg, col] += 1
        V[vh, col] += 1
        W[w00, col] += 1
        col += 1

    # Part 2: A[0:2, 2] * B[2, 0:8]
    # A[0,2]=2, A[1,2]=5
    # B[2, k] = 16+k
    # C[i, k] = W[2k+i]
    for k in range(8):
        if check_deadline():
            return _schoolbook_fallback(n, m, p)
        for i in range(2):
            U[i * 3 + 2, col] += 1
            V[16 + k, col] += 1
            W[2 * k + i, col] += 1
            col += 1

    if col != rank:
        # Should not happen
        return _schoolbook_fallback(n, m, p)

    # Convert numpy arrays to lists of lists as required by the interface convention
    # The seed program returned lists.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Optional: Verify correctness on a small random instance if time permits
    # For safety and speed, we skip heavy verification in the final return,
    # as the construction is deterministic and verified in development.
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook_fallback(n, m, p):
    """Returns the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
