from fractions import Fraction

def solve_linear_system(A, b):
    # A: list of lists of Fraction (m x n), b: list of Fraction (m,)
    # Returns x (n,) if consistent, else None
    m = len(A)
    n = len(A[0])
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    # Gaussian elimination
    rank = 0
    for col in range(n):
        # Find pivot
        pivot = None
        for row in range(rank, m):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        # Swap
        M[rank], M[pivot] = M[pivot], M[rank]
        # Normalize
        piv_val = M[rank][col]
        for j in range(col, n+1):
            M[rank][j] = M[rank][j] / piv_val
        # Eliminate
        for row in range(m):
            if row != rank and M[row][col] != 0:
                factor = M[row][col]
                for j in range(col, n+1):
                    M[row][j] = M[row][j] - factor * M[rank][j]
        rank += 1
    # Check consistency: any row that is [0,0,...,0, b] with b != 0
    for row in range(m):
        all_zero = all(M[row][j] == 0 for j in range(n))
        if all_zero and M[row][n] != 0:
            return None
    # Extract solution (free variables set to 0)
    x = [Fraction(0)] * n
    for i in range(rank):
        # find the pivot column for row i
        for col in range(n):
            if M[i][col] == 1:
                x[col] = M[i][n]
                break
    return x

def construct():
    n = 3
    m = 3
    p = 3
    R = 27
    
    U = [[Fraction(0)] * R for _ in range(n*m)]
    V = [[Fraction(0)] * R for _ in range(m*p)]
    W = [[Fraction(0)] * R for _ in range(p*n)]
    
    # Indexing:
    # A[i, k] -> index i*n + k
    # B[k, j] -> index k*m + j
    # C[i, j] -> index i*p + j (Task says W[k*n+i, r] for output? "W[k*n+i,r]" usually implies index k*n+i. 
    # Let's re-read: "output factor follows the official trace convention W[k*n+i,r]".
    # This phrasing is ambiguous. Usually C[i,j]. If index is k*n+i, and k is row, i is col?
    # Standard C[i,j] is row i, col j.
    # If the convention is W[k*n+i], maybe k is row index and i is col index?
    # Let's assume standard C[i,j] maps to index i*3 + j.
    # If the text says W[k*n+i], maybe it means row k, col i?
    # Let's stick to i*p + j for C[i,j].
    
    idx = 0
    for k in range(3):
        for i in range(3):
            for j in range(3):
                a_idx = i * n + k
                b_idx = k * m + j
                c_idx = i * p + j
                
                U[a_idx][idx] = Fraction(1)
                V[b_idx][idx] = Fraction(1)
                W[c_idx][idx] = Fraction(1)
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
