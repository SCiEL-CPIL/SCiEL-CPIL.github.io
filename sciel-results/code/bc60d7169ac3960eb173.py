"""Improved program for 6x6 matrix multiplication tensor decomposition.
Uses Kronecker product of Strassen (rank 7) for 2x2 block structure
and schoolbook (rank 27) for 3x3 leaf structure, achieving rank 189."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds to leave before deadline

# Strassen algorithm for 2x2 matrices (rank 7)
# M1 = A11 * (B22 - B12)
# M2 = (A11 + A21) * B12
# M3 = (A11 + A22) * (B12 + B22)
# M4 = A22 * (B11 - B21)
# M5 = (A11 + A12) * B11
# M6 = (A21 - A11) * (B11 + B21)
# M7 = (A12 - A22) * (B21 + B22)
# C11 = M5 + M3 - M7 - M1
# C12 = M1 + M4
# C21 = M2 + M7
# C22 = M3 - M1 + M4 + M6
#
# U[row = i_b * 2 + j_b, col = m] = coefficient of A[i_b, j_b] in M_m
# V[row = j_b * 2 + k_b, col = m] = coefficient of B[j_b, k_b] in M_m
# W[row = k_b * 2 + i_b, col = m] = coefficient of C[i_b, k_b] in M_m (using trace convention W[k*n+i])

STRASSEN_U = [
    # M1: A11*(B22-B12) -> A[1,1]
    [0, 0, 0, 0, 0, 0, 1],
    # M2: (A11+A21)*B12 -> A[1,1], A[2,1]
    [1, 0, 1, 0, 0, 0, 0],
    # M3: (A11+A22)*(B12+B22) -> A[1,1], A[2,2]
    [1, 0, 1, 0, 0, 0, 0],
    # M4: A22*(B11-B21) -> A[2,2]
    [0, 0, 0, 0, 0, 0, 1],
]
# Actually, let me define properly with row = i*2+j (i=row of A, j=col of A)
# A11 = row0,col0; A12 = row0,col1; A21 = row1,col0; A22 = row1,col1
# Row indices: A11->0, A12->1, A21->2, A22->3

STRASSEN_U = np.array([
    # M1: A11*(B22-B12): A11 has coeff 1
    [1, 0, 0, 0, 0, 0, 0],  # row 0 (A11)
    [0, 0, 0, 0, 0, 0, 0],  # row 1 (A12)
    [0, 0, 0, 0, 0, 0, 0],  # row 2 (A21)
    [0, 0, 0, 0, 0, 0, 0],  # row 3 (A22)
])
STRASSEN_U[0, 0] = 1  # M1: A11
STRASSEN_U[0, 1] = 1  # M2: A11
STRASSEN_U[0, 2] = 1  # M3: A11
STRASSEN_U[2, 1] = 1  # M2: A21
STRASSEN_U[0, 4] = 1  # M5: A11
STRASSEN_U[1, 4] = 1  # M5: A12
STRASSEN_U[3, 3] = 1  # M4: A22
STRASSEN_U[3, 2] = 1  # M3: A22
STRASSEN_U[1, 5] = -1  # M6: -A11
STRASSEN_U[2, 5] = 1   # M6: A21
STRASSEN_U[1, 6] = 1   # M7: A12
STRASSEN_U[3, 6] = -1  # M7: -A22

# V: B factors, row = j*2+k (j=row of B, k=col of B)
# B11->row0, B12->row1, B21->row2, B22->row3
STRASSEN_V = np.array([
    [0, 0, 0, 0, 0, 0, 0],  # B11
    [0, 0, 0, 0, 0, 0, 0],  # B12
    [0, 0, 0, 0, 0, 0, 0],  # B21
    [0, 0, 0, 0, 0, 0, 0],  # B22
])
STRASSEN_V[3, 0] = 1   # M1: B22
STRASSEN_V[1, 0] = -1  # M1: -B12
STRASSEN_V[1, 1] = 1   # M2: B12
STRASSEN_V[1, 2] = 1   # M3: B12
STRASSEN_V[3, 2] = 1   # M3: B22
STRASSEN_V[0, 3] = 1   # M4: B11
STRASSEN_V[2, 3] = -1  # M4: -B21
STRASSEN_V[0, 4] = 1   # M5: B11
STRASSEN_V[0, 5] = 1   # M6: B11
STRASSEN_V[2, 5] = 1   # M6: B21
STRASSEN_V[2, 6] = 1   # M7: B21
STRASSEN_V[3, 6] = 1   # M7: B22

# W: C factors, row = k*2+i (k=row of C, i=col of C)
# C11->row0 (k=0,i=0), C12->row1 (k=0,i=1), C21->row2 (k=1,i=0), C22->row3 (k=1,i=1)
STRASSEN_W = np.array([
    [0, 0, 0, 0, 0, 0, 0],  # C11
    [0, 0, 0, 0, 0, 0, 0],  # C12
    [0, 0, 0, 0, 0, 0, 0],  # C21
    [0, 0, 0, 0, 0, 0, 0],  # C22
])
# C11 = M5 + M3 - M7 - M1
STRASSEN_W[0, 4] = 1   # C11: M5
STRASSEN_W[0, 2] = 1   # C11: M3
STRASSEN_W[0, 6] = -1  # C11: -M7
STRASSEN_W[0, 0] = -1  # C11: -M1
# C12 = M1 + M4
STRASSEN_W[1, 0] = 1   # C12: M1
STRASSEN_W[1, 3] = 1   # C12: M4
# C21 = M2 + M7
STRASSEN_W[2, 1] = 1   # C21: M2
STRASSEN_W[2, 6] = 1   # C21: M7
# C22 = M3 - M1 + M4 + M6
STRASSEN_W[3, 2] = 1   # C22: M3
STRASSEN_W[3, 0] = -1  # C22: -M1
STRASSEN_W[3, 3] = 1   # C22: M4
STRASSEN_W[3, 5] = 1   # C22: M6

# Schoolbook for 3x3 matrices (rank 27)
# U[i*3+j, k] = 1 if k == i*3+j
# V[j*3+k, k] = 1 if k == j*3+k  (where k here is the column index in 0..26)
# W[k*3+i, k] = 1 if k == k*3+i
SCHOOL_U = np.zeros((9, 27))
SCHOOL_V = np.zeros((9, 27))
SCHOOL_W = np.zeros((9, 27))
for i in range(3):
    for j in range(3):
        for k in range(3):
            c = i * 9 + j * 3 + k
            SCHOOL_U[i * 3 + j, c] = 1
            SCHOOL_V[j * 3 + k, c] = 1
            SCHOOL_W[k * 3 + i, c] = 1


def construct():
    """Construct a rank-189 decomposition for 6x6 matrix multiplication
    using Kronecker product of Strassen (rank 7) and schoolbook (rank 27)."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # all 6
    
    rank_22 = 7  # Strassen rank for 2x2
    rank_33 = 27  # Schoolbook rank for 3x3
    rank_total = rank_22 * rank_33  # 189
    
    # Build the 6x6 decomposition using Kronecker product structure
    # For each term (c22, c33), the contribution is:
    # U6[i*6+j, c] = U22[i_b*2+j_b, c22] * U33[i_e*3+j_e, c33]
    # where i = i_b*3 + i_e, j = j_b*3 + j_e, c = c22*27 + c33
    
    U = np.zeros((n * m, rank_total))  # (36, 189)
    V = np.zeros((m * p, rank_total))  # (36, 189)
    W = np.zeros((p * n, rank_total))  # (36, 189)
    
    for c22 in range(rank_22):
        for c33 in range(rank_33):
            c = c22 * rank_33 + c33
            
            # For each block position (i_b, j_b) and element position (i_e, j_e)
            for i_b in range(2):
                for j_b in range(2):
                    for i_e in range(3):
                        for j_e in range(3):
                            i = i_b * 3 + i_e
                            j = j_b * 3 + j_e
                            u22 = STRASSEN_U[i_b * 2 + j_b, c22]
                            u33 = SCHOOL_U[i_e * 3 + j_e, c33]
                            U[i * 6 + j, c] = u22 * u33
                            
                            # V factor: B[j_b, k_b] * B[j_e, k_e]
                            for k_b in range(2):
                                for k_e in range(3):
                                    k = k_b * 3 + k_e
                                    v22 = STRASSEN_V[j_b * 2 + k_b, c22]
                                    v33 = SCHOOL_V[j_e * 3 + k_e, c33]
                                    V[j * 6 + k, c] = v22 * v33
                            
                            # W factor: C[k_b, i_b] * C[k_e, i_e]
                            for k_b in range(2):
                                for k_e in range(3):
                                    k = k_b * 3 + k_e
                                    w22 = STRASSEN_W[k_b * 2 + i_b, c22]
                                    w33 = SCHOOL_W[k_e * 3 + i_e, c33]
                                    W[k * 6 + i, c] = w22 * w33
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Return what we have (should be complete by now)
        pass
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
