"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else len(V[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for M_{n,m,p}."""
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
    return U, V, W, rank

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C12 += M1, C21 += M1, C22 += M1
    # Wait, let me be careful with the convention.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For 2x2x2: i in {0,1}, j in {0,1}, k in {0,1}
    # U has 4 rows (i*2+j), V has 4 rows (j*2+k), W has 4 rows (k*2+i)
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M3 - M4 + M6
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M5 + M7
    
    # c=0: M1
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[1][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11 (k*n+i: k=0,i=0 -> 0; k=1,i=1 -> 3)
    
    # c=1: M2
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    W[2][1] = 1; W[3][1] = 1  # C10, C11 (k=1,i=0 -> 2; k=1,i=1 -> 3)
    
    # c=2: M3
    U[0][2] = 1  # A00
    V[2][2] = 1; V[3][2] = -1  # B10, -B11
    W[0][2] = 1; W[1][2] = 1  # C00, C01 (k=0,i=0 -> 0; k=0,i=1 -> 1)
    
    # c=3: M4
    U[3][3] = 1  # A11
    V[0][3] = 1; V[2][3] = -1  # B00, -B10
    W[2][3] = 1; W[3][3] = 1  # C10, C11
    
    # c=4: M5
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1  # B11
    W[0][4] = 1; W[1][4] = 1  # C00, C01
    
    # c=5: M6
    U[2][5] = 1; U[0][5] = -1  # A10, -A00
    V[0][5] = 1; V[2][5] = 1  # B00, B01
    W[2][5] = 1; W[3][5] = 1  # C10, C11
    
    # c=6: M7
    U[1][6] = 1; U[3][6] = -1  # A01, -A11
    V[1][6] = 1; V[3][6] = 1  # B10, B11
    W[0][6] = 1; W[1][6] = 1  # C00, C01
    
    return U, V, W

def _kronecker(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Kronecker product of two decompositions."""
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    new_rank = rank1 * rank2
    
    U_new = [[0]*new_rank for _ in range(n1*n2)]
    V_new = [[0]*new_rank for _ in range(m1*m2)]
    W_new = [[0]*new_rank for _ in range(p1*p2)]
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1*rank2 + c2
            for i1 in range(n1):
                for i2 in range(n2):
                    idx = i1*n2 + i2
                    U_new[idx][c] = U1[i1*m1 + j1 if False else i1*m1 + 0][c1] * 0  # placeholder
            # Let me redo this properly
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            idx_u = (i1*n2 + i2)*m2 + j2
                            # This isn't right either. Let me think.
                            # U1 has shape (n1*m1, rank1), indexed by (i1,j1) -> row i1*m1+j1
                            # U2 has shape (n2*m2, rank2), indexed by (i2,j2) -> row i2*m2+j2
                            # U_new has shape (n1*n2 * m1*m2, new_rank)
                            # The row index in U_new for (i1,i2,j1,j2) should be (i1*n2+i2)*(m1*m2) + j1*m2+j2
                            # But that's n1*n2*m1*m2 rows, which is (n1*n2)*(m1*m2)
                            pass
    # Let me restart this function
    U_new = [[0]*new_rank for _ in range(n1*n2 * m1*m2)]
    V_new = [[0]*new_rank for _ in range(m1*m2 * p1*p2)]
    W_new = [[0]*new_rank for _ in range(p1*p2 * n1*n2)]
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1*rank2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            row_u = (i1*n2 + i2)*(m1*m2) + j1*m2 + j2
                            U_new[row_u][c] = U1[i1*m1 + j1][c1] * U2[i2*m2 + j2][c2]
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            row_v = (j1*m2 + j2)*(p1*p2) + k1*p2 + k2
                            V_new[row_v][c] = V1[j1*p1 + k1][c1] * V2[j2*p2 + k2][c2]
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            row_w = (k1*p2 + k2)*(n1*n2) + i1*n2 + i2
                            W_new[row_w][c] = W1[k1*n1 + i1][c1] * W2[k2*n2 + i2][c2]
    
    return U_new, V_new, W_new, new_rank

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    deadline = time.time() + 140.0
    
    # Start with schoolbook
    U, V, W, best_rank = _schoolbook(n, m, p)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try block decomposition
    # Split n=3 as 1+2, m=5 as 2+3, p=6 as 3+3
    # Or try other splits
    
    # For now, return schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
