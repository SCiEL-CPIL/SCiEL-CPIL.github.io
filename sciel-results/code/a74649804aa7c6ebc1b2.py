"""
Rank-minimal tensor decomposition for 2x2x9 matrix multiplication.
Implements the Strassen-based rank-32 construction and returns it.
"""
import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[0])):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank32():
    """
    Build the known rank-32 decomposition:
    - 4 Strassen blocks (cols 0-27) covering B columns 0-7
    - 1 schoolbook block (cols 28-31) covering B column 8
    Returns (U, V, W) as lists of lists of floats.
    """
    n, m, p = 2, 2, 9
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]  # 4 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 18 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 18 rows
    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # Standard Strassen for 2x2x2 with B columns k0, k1
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]] where b00 = b_{0,k0}, b01 = b_{0,k1}, b10 = b_{1,k0}, b11 = b_{1,k1}
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0; W[k1*n + 1][col] = -1.0
        col += 1
        
        # M3 = a00 * (b01 - b11)
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        
        # M4 = a11 * (b10 - b00)
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0; V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0; W[k0*n + 1][col] = 1.0
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0; W[k1*n + 0][col] = 1.0
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p + k0][col] = 1.0; V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1

    def add_schoolbook(k):
        nonlocal col
        # For column k, compute all 4 products directly
        # a00 * b_{0,k} -> C_{k,0}
        U[0][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        # a01 * b_{1,k} -> C_{k,0}
        U[1][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        # a10 * b_{0,k} -> C_{k,1}
        U[2][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1
        # a11 * b_{1,k} -> C_{k,1}
        U[3][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1

    # Process B in 2-column Strassen blocks, then the last column with schoolbook
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook(8)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with the known rank-32 solution
    U, V, W = _build_rank32()
    
    # Verify the solution
    if not _verify(U, V, W, 2, 2, 9):
        # Fallback to full schoolbook rank 36
        rank_fb = 36
        U_fb = [[0.0] * rank_fb for _ in range(4)]
        V_fb = [[0.0] * rank_fb for _ in range(18)]
        W_fb = [[0.0] * rank_fb for _ in range(18)]
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    U_fb[i*2+j][c] = 1.0
                    V_fb[j*9+k][c] = 1.0
                    W_fb[k*2+i][c] = 1.0
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
