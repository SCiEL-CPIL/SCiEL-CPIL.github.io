"""Matrix multiplication 2,3,5 — rank-28 decomposition via two Strassen 2x2x2 blocks."""

def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication using rank-28 ansatz.
    
    Strategy: Apply Strassen's 7-term decomposition to two 2x2x2 sub-tensors:
    - Block A: i in {0,1}, j in {0,1}, k in {0,1} (8 schoolbook terms -> 7 Strassen)
    - Block B: i in {0,1}, j in {0,1}, k in {3,4} (8 schoolbook terms -> 7 Strassen)
    - Remaining 14 schoolbook terms: j=2 (10 terms) + j in {0,1}, k=2 (4 terms)
    
    Total rank: 7 + 7 + 14 = 28
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (6, 28)
    V = [[0] * rank for _ in range(m * p)]  # (15, 28)
    W = [[0] * rank for _ in range(p * n)]  # (10, 28)
    
    c = 0
    
    # --- Strassen Block A: i in {0,1}, j in {0,1}, k in {0,1} ---
    # Indices: A[i*3+j], B[j*5+k], C[k*2+i]
    
    # m1 = (a00+a11)*(b00+b11) -> c00+=1, c11+=1
    U[0][c] = 1; U[4][c] = 1
    V[0][c] = 1; V[6][c] = 1
    W[0][c] = 1; W[5][c] = 1
    c += 1
    
    # m2 = (a10+a11)*b00 -> c10+=1, c11+=-1
    U[3][c] = 1; U[4][c] = 1
    V[0][c] = 1
    W[1][c] = 1; W[5][c] = -1
    c += 1
    
    # m3 = a00*(b01-b11) -> c01+=1, c11+=1
    U[0][c] = 1
    V[1][c] = 1; V[6][c] = -1
    W[2][c] = 1; W[5][c] = 1
    c += 1
    
    # m4 = a11*(b10-b00) -> c00+=1, c10+=1
    U[4][c] = 1
    V[5][c] = 1; V[0][c] = -1
    W[0][c] = 1; W[1][c] = 1
    c += 1
    
    # m5 = (a00+a01)*b11 -> c00+=-1, c01+=1
    U[0][c] = 1; U[1][c] = 1
    V[6][c] = 1
    W[0][c] = -1; W[2][c] = 1
    c += 1
    
    # m6 = (a10-a00)*(b00+b01) -> c11+=1
    U[3][c] = 1; U[0][c] = -1
    V[0][c] = 1; V[1][c] = 1
    W[5][c] = 1
    c += 1
    
    # m7 = (a01-a11)*(b10+b11) -> c00+=1
    U[1][c] = 1; U[4][c] = -1
    V[5][c] = 1; V[6][c] = 1
    W[0][c] = 1
    c += 1
    
    # --- Strassen Block B: i in {0,1}, j in {0,1}, k in {3,4} ---
    # This is a 2x2x2 sub-tensor with k indices {3,4} instead of {0,1}
    # A[i*3+j], B[j*5+k], C[k*2+i] where k in {3,4}
    
    # For this block, the "local" k indices are 0->3, 1->4
    # So B[j*5+3] and B[j*5+4], C[3*2+i] and C[4*2+i]
    
    # m1 = (a00+a11)*(b03+b14) -> c03+=1, c14+=1
    U[0][c] = 1; U[4][c] = 1
    V[3][c] = 1; V[9][c] = 1
    W[6][c] = 1; W[9][c] = 1
    c += 1
    
    # m2 = (a10+a11)*b03 -> c13+=1, c14+=-1
    U[3][c] = 1; U[4][c] = 1
    V[3][c] = 1
    W[7][c] = 1; W[9][c] = -1
    c += 1
    
    # m3 = a00*(b04-b14) -> c04+=1, c14+=1
    U[0][c] = 1
    V[4][c] = 1; V[9][c] = -1
    W[8][c] = 1; W[9][c] = 1
    c += 1
    
    # m4 = a11*(b13-b03) -> c03+=1, c13+=1
    U[4][c] = 1
    V[8][c] = 1; V[3][c] = -1
    W[6][c] = 1; W[7][c] = 1
    c += 1
    
    # m5 = (a00+a01)*b14 -> c03+=-1, c04+=1
    U[0][c] = 1; U[1][c] = 1
    V[9][c] = 1
    W[6][c] = -1; W[8][c] = 1
    c += 1
    
    # m6 = (a10-a00)*(b03+b04) -> c14+=1
    U[3][c] = 1; U[0][c] = -1
    V[3][c] = 1; V[4][c] = 1
    W[9][c] = 1
    c += 1
    
    # m7 = (a01-a11)*(b13+b14) -> c03+=1
    U[1][c] = 1; U[4][c] = -1
    V[8][c] = 1; V[9][c] = 1
    W[6][c] = 1
    c += 1
    
    # --- Remaining 14 schoolbook terms ---
    # j=2: (i, 2, k) for i in {0,1}, k in {0,1,2,3,4}: 10 terms
    # j in {0,1}, k=2: (i, j, 2) for i in {0,1}, j in {0,1}: 4 terms
    
    for i in range(n):
        for k in range(p):
            j = 2
            U[i*m+j][c] = 1
            V[j*p+k][c] = 1
            W[k*n+i][c] = 1
            c += 1
    
    for i in range(n):
        for j in range(2):
            k = 2
            U[i*m+j][c] = 1
            V[j*p+k][c] = 1
            W[k*n+i][c] = 1
            c += 1
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    return {"U": U, "V": V, "W": W}
