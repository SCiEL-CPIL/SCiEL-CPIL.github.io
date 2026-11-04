import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a specific partitioning to apply Strassen to 2x2x2 sub-blocks
    # I1 = {0, 1}, I2 = {2}
    # J1 = {0, 1}, J2 = {2, 3}, J3 = {4}
    # K1 = {0, 1}, K2 = {2, 3}, K3 = {4, 5}, K4 = {6}
    
    # Define the blocks for Strassen
    # (i_set, j_set, k_set)
    strassen_blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [2, 3]),
        ([0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [4, 5]),
    ]
    
    # Identify all (i, j, k) triples covered by Strassen blocks
    covered = set()
    for i_set, j_set, k_set in strassen_blocks:
        for i in i_set:
            for j in j_set:
                for k in k_set:
                    covered.add((i, j, k))
    
    # Identify remaining triples for schoolbook
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    schoolbook_terms.append((i, j, k))
                    
    # Calculate total rank
    rank = len(strassen_blocks) * 7 + len(schoolbook_terms)
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for C elements
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map local indices (0,1) to global indices
    # For a block (i_set, j_set, k_set):
    # i = i_set[0] + i_loc
    # j = j_set[0] + j_loc
    # k = k_set[0] + k_loc
    
    for i_set, j_set, k_set in strassen_blocks:
        i0, i1 = i_set
        j0, j1 = j_set
        k0, k1 = k_set
        
        # M1
        # U: (i0,j0) += 1, (i1,j1) += 1
        # V: (j0,k0) += 1, (j1,k1) += 1
        # W: (k0,i0) += 1, (k1,i1) += 1
        U[i0*m + j0][col] += 1
        U[i1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] += 1
        W[k1*n + i1][col] += 1
        col += 1
        
        # M2
        # U: (i1,j0) += 1, (i1,j1) += 1
        # V: (j0,k0) += 1
        # W: (k0,i1) += 1
        U[i1*m + j0][col] += 1
        U[i1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i1][col] += 1
        col += 1
        
        # M3
        # U: (i0,j0) += 1
        # V: (j0,k1) += 1, (j1,k1) -= 1
        # W: (k1,i0) += 1
        U[i0*m + j0][col] += 1
        V[j0*p + k1][col] += 1
        V[j1*p + k1][col] -= 1
        W[k1*n + i0][col] += 1
        col += 1
        
        # M4
        # U: (i1,j1) += 1
        # V: (j1,k0) += 1, (j0,k0) -= 1
        # W: (k0,i1) += 1
        U[i1*m + j1][col] += 1
        V[j1*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i1][col] += 1
        col += 1
        
        # M5
        # U: (i0,j0) += 1, (i0,j1) += 1
        # V: (j1,k1) += 1
        # W: (k1,i0) += 1
        U[i0*m + j0][col] += 1
        U[i0*m + j1][col] += 1
        V[j1*p + k1][col] += 1
        W[k1*n + i0][col] += 1
        col += 1
        
        # M6
        # U: (i1,j0) += 1, (i0,j0) -= 1
        # V: (j0,k0) += 1, (j0,k1) += 1
        # W: (k0,i1) += 1, (k1,i1) += 1
        U[i1*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + k1][col] += 1
        W[k0*n + i1][col] += 1
        W[k1*n + i1][col] += 1
        col += 1
        
        # M7
        # U: (i0,j1) += 1, (i1,j1) -= 1
        # V: (j1,k0) += 1, (j1,k1) += 1
        # W: (k0,i0) += 1, (k1,i0) += 1
        U[i0*m + j1][col] += 1
        U[i1*m + j1][col] -= 1
        V[j1*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] += 1
        W[k1*n + i0][col] += 1
        col += 1
        
    # Schoolbook terms
    for i, j, k in schoolbook_terms:
        U[i*m + j][col] = 1
        V[j*p + k][col] = 1
        W[k*n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
