"""Matrix multiplication tensor decomposition search for 4x4 * 4x5."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def _tensor_entry(i, j, k, n, m, p):
    """The matrix multiplication tensor entry: 1 if (i,j,k) is a valid triple, else 0.
    For A(n x m) * B(m x p) = C(n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    So T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor for the multiplication)
    """
    # The tensor T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is because C[i,k] = sum_j A[i,j]*B[j,k]
    return 1

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    # For each (i, j, k), check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, rank):
    """Faster verification using lists of Fractions."""
    # Pre-convert to Fraction lists for speed
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            ui = Uf[i*m+j]
            for k in range(p):
                vj = Vf[j*p+k]
                wk = Wf[k*n+i]
                total = Fraction(0)
                for c in range(rank):
                    total += ui[c] * vj[c] * wk[c]
                if total != 1:
                    return False
    return True

def _strassen_2x2_base():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2 (rank 7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen formulas for 2x2:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # c11 = M1 + M4 - M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M7
    
    # U rows: [a11, a12, a21, a22] (i*2+j)
    # V rows: [b11, b12, b21, b22] (j*2+k)  
    # W rows: [c11, c12, c21, c22] (k*2+i)
    
    # M1: (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1  # a11, a22
    V[0][0] = 1; V[3][0] = 1  # b11, b22
    W[0][0] = 1; W[3][0] = 1  # c11, c22
    
    # M2: (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1  # a21, a22
    V[0][1] = 1               # b11
    W[2][1] = 1; W[3][1] = -1 # c21, -c22
    
    # M3: a11*(b12-b22)
    U[0][2] = 1               # a11
    V[1][2] = 1; V[3][2] = -1 # b12, -b22
    W[1][2] = 1; W[3][2] = 1  # c12, c22
    
    # M4: a22*(b21-b11)
    U[3][3] = 1               # a22
    V[2][3] = 1; V[0][3] = -1 # b21, -b11
    W[2][3] = 1; W[0][3] = 1  # c21, c11
    
    # M5: (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1  # a11, a12
    V[3][4] = 1               # b22
    W[0][4] = -1; W[3][4] = 1 # -c11, c22
    
    # M6: (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1 # a21, -a11
    V[0][5] = 1; V[1][5] = 1  # b11, b12
    W[2][5] = 1; W[3][5] = 1  # c21, c22
    
    # M7: (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1 # a12, -a22
    V[2][6] = 1; V[3][6] = 1  # b21, b22
    W[0][6] = -1; W[3][6] = 1 # -c11, c22
    
    return U, V, W

def _random_candidate(rank, n, m, p, seed=42):
    """Generate a random candidate with small integer/half-integer entries."""
    rng = random.Random(seed)
    values = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    U = [[rng.choice(values) for _ in range(rank)] for _ in range(n*m)]
    V = [[rng.choice(values) for _ in range(rank)] for _ in range(m*p)]
    W = [[rng.choice(values) for _ in range(rank)] for _ in range(p*n)]
    
    return U, V, W

def _try_rank(rank, n, m, p, max_attempts=500, start_seed=0):
    """Try to find a valid decomposition at the given rank."""
    for attempt in range(max_attempts):
        seed = start_seed + attempt
        U, V, W = _random_candidate(rank, n, m, p, seed)
        if _verify_fast(U, V, W, n, m, p, rank):
            return U, V, W
    return None

def _structured_search(rank, n, m, p, max_time=10):
    """Structured search using Strassen-like patterns and local modifications."""
    start = time.time()
    rng = random.Random(rank * 1000 + 7)
    values = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    best_U = best_V = best_W = None
    
    for attempt in range(200):
        if time.time() - start > max_time:
            break
        
        # Start from a random sparse structure
        density = 0.15
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # Place some non-zero entries
        for _ in range(int(rank * density * 3)):
            u_row = rng.randint(0, n*m-1)
            v_row = rng.randint(0, m*p-1)
            w_row = rng.randint(0, p*n-1)
            col = rng.randint(0, rank-1)
            U[u_row][col] = rng.choice([1, -1])
            V[v_row][col] = rng.choice([1, -1])
            W[w_row][col] = rng.choice([1, -1])
        
        if _verify_fast(U, V, W, n, m, p, rank):
            return U, V, W
        
        # Local search: modify one entry at a time
        for _ in range(20):
            if time.time() - start > max_time:
                break
            choice = rng.randint(0, 2)
            if choice == 0:
                row = rng.randint(0, n*m-1)
                col = rng.randint(0, rank-1)
                U[row][col] = rng.choice(values)
            elif choice == 1:
                row = rng.randint(0, m*p-1)
                col = rng.randint(0, rank-1)
                V[row][col] = rng.choice(values)
            else:
                row = rng.randint(0, p*n-1)
                col = rng.randint(0, rank-1)
                W[row][col] = rng.choice(values)
            
            if _verify_fast(U, V, W, n, m, p, rank):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Known lower bounds: for 4x4*4x5, the rank is at least...
    # Let's try ranks from a reasonable lower bound up to schoolbook (80)
    # Theoretical lower bound is at least max(n*m, m*p, p*n) = max(16,20,20) = 20
    # But for matrix multiplication tensors, better lower bounds exist.
    # Let's try from 40 up to 80, skipping some.
    
    # First, try a structured approach at various ranks
    # Try ranks: 49, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80
    
    ranks_to_try = [49, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Try structured search
        result = _structured_search(rank, n, m, p, max_time=5)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Try random search with limited attempts
        result = _try_rank(rank, n, m, p, max_attempts=100, start_seed=rank*100)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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
