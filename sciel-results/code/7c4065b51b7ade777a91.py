"""Program for 4x5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return 7 (A_mask, B_mask, C_mask) for 2x2 matrix multiplication.
    Each mask is a dict: (row, col) -> coefficient."""
    A00, A01, A10, A11 = (0,0), (0,1), (1,0), (1,1)
    B00, B01, B10, B11 = (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6
    
    terms = []
    # M1
    terms.append(( {A00:1, A11:1}, {B00:1, B11:1} ))
    # M2
    terms.append(( {A10:1, A11:1}, {B00:1} ))
    # M3
    terms.append(( {A00:1}, {B01:1, B11:-1} ))
    # M4
    terms.append(( {A11:1}, {B10:1, B00:-1} ))
    # M5
    terms.append(( {A00:1, A01:1}, {B11:1} ))
    # M6
    terms.append(( {A10:1, A00:-1}, {B00:1, B01:1} ))
    # M7
    terms.append(( {A01:1, A11:-1}, {B10:1, B11:1} ))
    
    # C combination: C_ij = sum_q coef[q][ij] * M_q
    C_coefs = []
    # For each term q, C_coefs[q] = {(0,0): c00, (0,1): c01, (1,0): c10, (1,1): c11}
    C_coefs.append({(0,0):1, (0,1):0, (1,0):0, (1,1):1})  # M1
    C_coefs.append({(0,0):0, (0,1):0, (1,0):1, (1,1):1})  # M2
    C_coefs.append({(0,0):0, (0,1):1, (1,0):0, (1,1):-1}) # M3
    C_coefs.append({(0,0):-1,(0,1):0, (1,0):1, (1,1):0})  # M4
    C_coefs.append({(0,0):0, (0,1):1, (1,0):0, (1,1):0})  # M5
    C_coefs.append({(0,0):1, (0,1):0, (1,0):0, (1,1):-1}) # M6
    C_coefs.append({(0,0):-1,(0,1):0, (1,0):0, (1,1):0})  # M7
    
    return terms, C_coefs

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Standard schoolbook decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term for A[i,j] * B[j,k] contributing to C[i,k]
                # U row: i*m + j
                # V row: j*p + k
                # W row: k*n + i (trace convention W[k*n+i, r])
                
                U_mat[i * m + j][idx] = 1
                V_mat[j * p + k][idx] = 1
                W_mat[k * n + i][idx] = 1
                
                idx += 1
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
