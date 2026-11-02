import time

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _build_rank14():
    """
    Constructs the rank-14 decomposition for 2x2x4 matrix multiplication.
    This is the direct sum of two rank-7 Strassen decompositions for 2x2x2.
    U: (4, 14)
    V: (8, 14)
    W: (8, 14)
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Output:
    # C00 = M1 - M4 + M5 - M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A (2x2):
    # A00=0, A01=1, A10=2, A11=3
    
    # Indices for B (2x4):
    # Left block (cols 0,1): B00=0, B01=1, B10=4, B11=5
    # Right block (cols 2,3): B02=2, B03=3, B12=6, B13=7
    
    # Indices for C (2x4) in column-major (k = c*n + a):
    # C00=0, C10=1, C01=2, C11=3
    # C02=4, C12=5, C03=6, C13=7
    
    # Left Block (Strassen on B cols 0,1) -> C cols 0,1
    # Rank slots 0-6
    
    # M1_L
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2_L
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3_L
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4_L
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5_L
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6_L
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7_L
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1
    
    # Right Block (Strassen on B cols 2,3) -> C cols 2,3
    # Rank slots 7-13
    
    # M1_R
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # M2_R
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # M3_R
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # M4_R
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # M5_R
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # M6_R
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # M7_R
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """
    Verifies the decomposition exactly.
    """
    n, m, p = 2, 2, 4
    rank = len(U[0])
    
    # Check non-zero terms
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                if abs(t_val - 1.0) > 1e-9:
                    return False
    
    # Check zero terms
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                a = i // m
                b = i % m
                b_prime = j // p
                c = j % p
                c_prime = k // n
                a_prime = k % n
                
                if a == a_prime and b == b_prime:
                    continue
                
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                if abs(t_val) > 1e-9:
                    return False
    
    return True

def construct():
    # Directly return the known optimal (or near-optimal) rank 14 solution.
    # Random search for lower ranks is omitted to ensure speed and reliability,
    # as rank 14 is the theoretical lower bound for this tensor using standard methods.
    result = _build_rank14()
    
    # Optional: Verify if time allows, but construction is deterministic and verified.
    # We skip verification in the final return to save time, as it's a known correct solution.
    # However, for safety in a dynamic environment, we can do a quick check.
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    else:
        # Fallback to schoolbook if something went wrong (should not happen)
        n, m, p = 2, 2, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        idx = 0
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    U[i][idx] = 1
                    V[j][idx] = 1
                    W[k][idx] = 1
                    idx += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
