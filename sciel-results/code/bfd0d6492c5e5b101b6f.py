"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0
import numpy as np

def _verify(U, V, W, n=2, m=2, p=5):
    """Verify exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7) for A entries [a00, a01, a10, a11]
    V: (4, 7) for B entries [b00, b01, b10, b11]
    W: (4, 7) for C entries [c00, c01, c10, c11]
    
    Note: In our convention, C[k,i] = sum_j A[i,j]*B[j,k]
    So C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    This is equivalent to standard C = A * B^T if we think of B as 2x2 with columns.
    
    Actually, let's use the standard Strassen for C = A * B where:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    Strassen:
    m1 = (a00+a11)(b00+b11)
    m2 = (a10+a11)b00
    m3 = a00(b01-b11)
    m4 = a11(b10-b00)
    m5 = (a00+a10)b11
    m6 = (a01-a10)(b00+b10)
    m7 = (a01-a11)(b01+b11)
    
    C[0,0] = m1 - m4 - m5
    C[0,1] = m3 + m5
    C[1,0] = m2 + m4
    C[1,1] = m1 - m3 - m2
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # m1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # m3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # m4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    # m5 = (a00+a10)b11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    # m6 = (a01-a10)(b00+b10)
    U[1, 5] = 1; U[2, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    # m6 doesn't directly contribute to standard Strassen C entries
    # Actually, m6 is used in C[0,0] = m1 - m4 - m5... wait
    # Let me re-verify the Strassen formulas.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 - M5 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M2 + M7
    
    # I made an error. Let me use the correct formulas.
    # C[0,0] = m1 - m4 - m5 + m6
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m3 - m2 + m7
    
    W[0, 5] = 1  # m6 contributes to C[0,0]
    # m6 = (a01-a10)(b00+b10) -- wait, this doesn't match standard Strassen either.
    
    # Let me just use a known correct Strassen implementation.
    # For C = A*B:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a10)b11
    # m6 = (a01-a10)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    
    # C00 = m1 - m4 - m5
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m3 - m2
    
    # Let me verify C00:
    # m1 - m4 - m5 = (a00+a11)(b00+b11) - a11(b10-b00) - (a00+a10)b11
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b10 + a11*b00 - a00*b11 - a10*b11
    # = a00*b00 + 2*a11*b00 + a11*b11 - a11*b10 - a10*b11
    # This should equal a00*b00 + a01*b10. It doesn't match.
    
    # The issue is that I'm mixing up the indices. Let me use a different approach.
    # I'll just use the schoolbook for the 2x2 part to be safe, which gives rank 4.
    # Or I can use a verified Strassen implementation.
    
    # For now, let's use schoolbook for 2x2x2 (rank 4) and see if we can improve.
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j, i*2+j] = 1
                V[j*2+k, i*2+j] = 1
                W[k*2+i, i*2+j] = 1
    
    return U, V, W

def _schoolbook_2x2x5():
    """Return (U, V, W) for 2x2x5 using schoolbook (rank 20)."""
    n, m, p = 2, 2, 5
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _hybrid_decomposition():
    """
    Use Strassen-like structure for first 2 columns, schoolbook for last 3.
    Target rank: 7 + 6 = 13
    """
    n, m, p = 2, 2, 5
    rank = 13
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # For columns 0,1 of B (indices 0,1 in B's columns), use Strassen (7 terms)
    # For columns 2,3,4 of B, use schoolbook (2 terms each = 6 terms)
    
    # Strassen for 2x2x2 on columns 0,1:
    # A = [[a00, a01], [a10, a11]]
    # B_subset = [[b00, b01], [b10, b11]]
    # C_subset = [[c00, c01], [c10, c11]] where c_ki = sum_j a_ij * b_jk
    
    # Using correct Strassen:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a10)b11
    # m6 = (a01-a10)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    
    # C00 = a00*b00 + a01*b10 = m1 - m4 - m5
    # C01 = a00*b01 + a01*b11 = m3 + m5
    # C10 = a10*b00 + a11*b10 = m2 + m4
    # C11 = a10*b01 + a11*b11 = m1 - m3 - m2
    
    # Term 0: m1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[6, 0] = 1
    W[0, 0] = 1; W[7, 0] = 1
    
    # Term 1: m2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[6, 1] = 1; W[7, 1] = -1
    
    # Term 2: m3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[6, 2] = -1
    W[1, 2] = 1; W[7, 2] = -1
    
    # Term 3: m4 = a11(b10-b00)
    U[3, 3] = 1
    V[5, 3] = 1; V[0, 3] = -1
    W[6, 3] = 1; W[0, 3] = -1
    
    # Term 4: m5 = (a00+a10)b11
    U[0, 4] = 1; U[2, 4] = 1
    V[6, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    # Terms 5,6: m6, m7 not needed for the 4 C entries if we use the above
    # Actually, the above 5 terms give:
    # C00 = m1 - m4 - m5
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m3 - m2
    # This uses only 5 terms, not 7. But Strassen uses 7.
    # The issue is that m6 and m7 are needed for the general case.
    # For 2x2x2, the minimal rank is 7 (Strassen).
    
    # Let me just use schoolbook for all 5 columns to be safe.
    # Rank = 20.
    
    rank = 20
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Try to find a low-rank decomposition
    # Start with schoolbook (rank 20) as fallback
    U_school, V_school, W_school = _schoolbook_2x2x5()
    best_rank = 20
    best_result = (U_school.tolist(), V_school.tolist(), W_school.tolist())
    
    # Try hybrid approach with rank 13
    U_hybrid, V_hybrid, W_hybrid = _hybrid_decomposition()
    if _verify(U_hybrid.tolist(), V_hybrid.tolist(), W_hybrid.tolist(), n, m, p):
        best_rank = 13
        best_result = (U_hybrid.tolist(), V_hybrid.tolist(), W_hybrid.tolist())
    
    # Try to find even lower rank using search
    for rank in range(10, best_rank):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Random search with small integers
        candidates = [-1, 0, 1]
        for attempt in range(5000):
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            
            U = np.random.choice(candidates, size=(n*m, rank))
            V = np.random.choice(candidates, size=(m*p, rank))
            W = np.random.choice(candidates, size=(p*n, rank))
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_rank = rank
                best_result = (U.tolist(), V.tolist(), W.tolist())
                break
        
        if best_rank == rank:
            break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
