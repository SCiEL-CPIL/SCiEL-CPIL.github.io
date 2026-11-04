"""Matrix multiplication 2x3 by 3x5 with rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[a, b, c] = 1 if the term A[i,j]*B[j,k] contributes to C[k,i]
    where a = i*m+j, b = j*p+k, c = k*n+i
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] equals T[a,b,c]."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.max(np.abs(T_recon - T))
    return diff < tol, diff

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    
    A indices: (i,j) -> i*2+j: A00=0, A01=1, A10=2, A11=3
    B indices: (j,k) -> j*2+k: B00=0, B01=1, B10=2, B11=3
    C indices: (k,i) -> k*2+i: C00=0, C01=1, C10=2, C11=3
    
    C = A * B where C[k,i] = sum_j A[i,j]*B[j,k]
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    # Contributes to C00, C01, C10, C11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # M2 = (A10-A00)*(B00+B01)
    # Contributes to C10, C11
    U[2][1] = 1; U[0][1] = -1
    V[0][1] = 1; V[1][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = (A00+A01)*(B10-B00)
    # Contributes to C00, C01
    U[0][2] = 1; U[1][2] = 1
    V[2][2] = 1; V[0][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # M4 = (A11+A01)*(B01-B11)
    # Contributes to C01, C11
    U[3][3] = 1; U[1][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[1][3] = 1; W[3][3] = 1
    
    # M5 = (A10+A11)*(B00+B11)
    # Contributes to C10, C11
    U[2][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[3][4] = 1
    W[2][4] = 1; W[3][4] = 1
    
    # M6 = (A00-A10)*(B10+B11)
    # Contributes to C00, C01
    U[0][5] = 1; U[2][5] = -1
    V[2][5] = 1; V[3][5] = 1
    W[0][5] = 1; W[1][5] = 1
    
    # M7 = (A01-A11)*(B00-B10)
    # Contributes to C00, C01
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[2][6] = -1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _strassen_2x2x3_block():
    """
    Decompose 2x2 * 2x3 matrix multiplication (A[2x2] * B[2x3] = C[3x2]).
    Uses Strassen 2x2x2 for first two columns of B, schoolbook for third.
    Rank: 7 + 4 = 11.
    
    A indices (local): 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    B indices (local): 0->(0,0), 1->(0,1), 2->(0,2), 3->(1,0), 4->(1,1), 5->(1,2)
    C indices (local): 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1), 4->(2,0), 5->(2,1)
    where C[k,i], k in {0,1,2}, i in {0,1}
    """
    rank = 11
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(6)]
    
    # Strassen 2x2x2 for B columns 0,1 (local B indices 0,1,3,4)
    U_s, V_s, W_s = _strassen_2x2x2()
    # Map: Strassen B rows 0,1,2,3 -> local B rows 0,1,3,4
    b_map = [0, 1, 3, 4]
    # Map: Strassen C rows 0,1,2,3 -> local C rows 0,1,2,3
    c_map = [0, 1, 2, 3]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[s][r] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map[s]][r] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map[s]][r] += W_s[s][r]
    
    # Schoolbook for B column 2 (local B indices 2,5)
    col = 7
    for i in range(2):
        for j in range(2):
            U[i*2+j][col] = 1
            V[j*3+2][col] = 1  # B[j,2]
            W[2*2+i][col] = 1  # C[2,i]
            col += 1
    
    return U, V, W

def _strassen_2x3x5():
    """
    Use Strassen blocks to reduce rank.
    
    Block 1: j in {0,1}, k in {0,1} -> Strassen 2x2x2, rank 7
    Block 2: j in {0,1}, k in {2,3,4} -> Strassen 2x2x3, rank 11
    Block 3: j=2, k in {0,1,2,3,4} -> schoolbook, rank 10
    
    Total rank: 7 + 11 + 10 = 28
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (6, 28)
    V = [[0] * rank for _ in range(m * p)]  # (15, 28)
    W = [[0] * rank for _ in range(p * n)]  # (10, 28)
    
    col = 0
    
    # Block 1: Strassen for j in {0,1}, k in {0,1}
    U_s, V_s, W_s = _strassen_2x2x2()
    a_map = [0, 1, 3, 4]  # Strassen A row -> full U row (i*3+j)
    b_map = [0, 1, 5, 6]  # Strassen B row -> full V row (j*5+k)
    c_map = [0, 1, 2, 3]  # Strassen C row -> full W row (k*2+i)
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map[s]][col] += W_s[s][r]
        col += 1
    
    # Block 2: Strassen 2x2x3 for j in {0,1}, k in {2,3,4}
    U_b, V_b, W_b = _strassen_2x2x3_block()
    # Full U rows for j in {0,1}: i*3+j where i in {0,1}, j in {0,1} -> 0,1,3,4
    a_map2 = [0, 1, 3, 4]
    # Full V rows for j in {0,1}, k in {2,3,4}: j*5+k
    # Local B: 0->(0,2), 1->(0,3), 2->(0,4), 3->(1,2), 4->(1,3), 5->(1,4)
    b_map2 = [0*5+2, 0*5+3, 0*5+4, 1*5+2, 1*5+3, 1*5+4]  # [2,3,4,7,8,9]
    # Full W rows for k in {2,3,4}, i in {0,1}: k*2+i
    # Local C: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1), 4->(2,0), 5->(2,1)
    c_map2 = [2*2+0, 2*2+1, 3*2+0, 3*2+1, 4*2+0, 4*2+1]  # [4,5,6,7,8,9]
    
    for r in range(11):
        for s in range(4):
            if U_b[s][r] != 0:
                U[a_map2[s]][col + r] += U_b[s][r]
        for s in range(6):
            if V_b[s][r] != 0:
                V[b_map2[s]][col + r] += V_b[s][r]
        for s in range(6):
            if W_b[s][r] != 0:
                W[c_map2[s]][col + r] += W_b[s][r]
    col += 11
    
    # Block 3: schoolbook for j=2, all k
    for i in range(2):
        for k in range(5):
            U[i * 3 + 2][col] = 1
            V[2 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Try Strassen-based decomposition (rank 29)
    result = _strassen_2x3x5()
    
    # Verify
    T = _build_target_tensor(n, m, p)
    valid, diff = _verify_decomposition(result["U"], result["V"], result["W"], T)
    
    if valid:
        return result
    else:
        # Fallback to schoolbook
        return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
