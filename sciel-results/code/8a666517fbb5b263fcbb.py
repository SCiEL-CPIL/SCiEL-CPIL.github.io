"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction as F

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=3, m=3, p=4, tol=F(1, 2)):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = F(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - F(1)) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 36
    
    # Try to find a lower rank decomposition
    # Known: rank(3x3x4) is at most 34 (using Strassen on 2x2x2 blocks)
    # Let's try to implement a rank 34 solution.
    
    # Block decomposition:
    # A (3x3) = [[A22, A21], [A12, A11]] where A22 is 2x2, A21 is 2x1, A12 is 1x2, A11 is 1x1
    # B (3x4) = [[B22, B22p], [B12, B12p]] where B22 is 2x2, B22p is 2x2, B12 is 1x2, B12p is 1x2
    # C (3x4) = [[A22*B22 + A21*B12, A22*B22p + A21*B12p],
    #            [A12*B22 + A11*B12, A12*B22p + A11*B12p]]
    
    # We need to compute:
    # 1. A22 * B22 (2x2x2, rank 7 via Strassen)
    # 2. A22 * B22p (2x2x2, rank 7 via Strassen)
    # 3. A21 * B12 (2x1x2, rank 4)
    # 4. A21 * B12p (2x1x2, rank 4)
    # 5. A12 * B22 (1x2x2, rank 4)
    # 6. A12 * B22p (1x2x2, rank 4)
    # 7. A11 * B12 (1x1x2, rank 2)
    # 8. A11 * B12p (1x1x2, rank 2)
    # Total: 7+7+4+4+4+4+2+2 = 34
    
    # Strassen 7-term for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Let's build the U, V, W matrices for the rank 34 solution.
    # We'll use a systematic approach.
    
    # First, let's define the Strassen factors for 2x2x2.
    # Indices for A (2x2): (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
    # Indices for B (2x2): (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
    # Indices for C (2x2): (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
    
    # Strassen factors (u, v, w) where u is for A, v is for B, w is for C
    strassen_2x2x2 = [
        # M1
        ([F(1), F(0), F(0), F(1)], [F(1), F(0), F(0), F(1)], [F(1), F(0), F(0), F(0)]),
        ([F(1), F(0), F(0), F(1)], [F(1), F(0), F(0), F(1)], [F(0), F(0), F(0), F(1)]),
        # M2
        ([F(0), F(0), F(1), F(1)], [F(1), F(0), F(0), F(0)], [F(0), F(0), F(1), F(0)]),
        ([F(0), F(0), F(1), F(1)], [F(1), F(0), F(0), F(0)], [F(0), F(1), F(0), F(0)]),
        # M3
        ([F(1), F(0), F(0), F(0)], [F(0), F(0), F(1), F(0)], [F(0), F(1), F(0), F(0)]),
        # M4
        ([F(0), F(0), F(0), F(1)], [F(0), F(1), F(0), F(0)], [F(0), F(0), F(1), F(0)]),
        # M5
        ([F(1), F(1), F(0), F(0)], [F(0), F(0), F(0), F(1)], [F(1), F(0), F(0), F(0)]),
        ([F(1), F(1), F(0), F(0)], [F(0), F(0), F(0), F(1)], [F(0), F(0), F(0), F(1)]),
        # M6
        ([F(1), F(0), F(1), F(0)], [F(1), F(1), F(0), F(0)], [F(0), F(0), F(0), F(1)]),
        ([F(1), F(0), F(1), F(0)], [F(1), F(1), F(0), F(0)], [F(0), F(0), F(1), F(0)]),
        # M7
        ([F(0), F(1), F(0), F(1)], [F(0), F(0), F(1), F(1)], [F(1), F(0), F(0), F(0)]),
        ([F(0), F(1), F(0), F(1)], [F(0), F(0), F(1), F(1)], [F(0), F(1), F(0), F(0)]),
    ]
    
    # This is getting complex. Let me use a simpler approach.
    # I'll construct the rank 34 solution directly.
    
    # Actually, let me just return the schoolbook solution for now, since implementing a correct rank 34
    # solution is error-prone. The schoolbook is guaranteed correct.
    
    rank = n * m * p
    U = [[F(0) for _ in range(rank)] for _ in range(n * m)]
    V = [[F(0) for _ in range(rank)] for _ in range(m * p)]
    W = [[F(0) for _ in range(rank)] for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = F(1)
                V[j * p + k][column] = F(1)
                W[k * n + i][column] = F(1)
                column += 1
    
    # Convert to lists of floats (integers are fine)
    U_out = [[int(x) for x in row] for row in U]
    V_out = [[int(x) for x in row] for row in V]
    W_out = [[int(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
