"""Matrix multiplication 6x6x6 via Kronecker product of Strassen (2x2) and schoolbook (3x3)."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-189 decomposition of 6x6 matrix multiplication.
    
    The 6x6 multiplication tensor is decomposed as the Kronecker product of:
    - Strassen's rank-7 algorithm for 2x2 matrix multiplication
    - Schoolbook rank-27 algorithm for 3x3 matrix multiplication
    
    Total rank: 7 * 27 = 189
    
    Index convention:
    - U[i*6+j, r] for A_{i,j} input
    - V[j*6+k, r] for B_{j,k} input  
    - W[k*6+i, r] for C_{i,k} output (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Strassen algorithm for 2x2 matrix multiplication
    # W convention: W[k*2+i] for C[i,k]
    # C11: W[0*2+0] = W[0]
    # C12: W[0*2+1] = W[1]
    # C21: W[1*2+0] = W[2]
    # C22: W[1*2+1] = W[3]
    
    # U convention: U[i*2+j] for A[i,j]
    # A11: U[0*2+0] = U[0]
    # A12: U[0*2+1] = U[1]
    # A21: U[1*2+0] = U[2]
    # A22: U[1*2+1] = U[3]
    
    # V convention: V[j*2+k] for B[j,k]
    # B11: V[0*2+0] = V[0]
    # B12: V[0*2+1] = V[1]
    # B21: V[1*2+0] = V[2]
    # B22: V[1*2+1] = V[3]
    
    # Strassen terms: each is (U_dict, V_dict, W_dict)
    # Keys are (i,j) or (j,k) or (k,i) as appropriate
    
    strassen_terms = [
        # Term 0: M1 = (A11 + A22) * (B11 + B22), contributes to C11 + C22
        (
            {(0,0): 1, (1,1): 1},   # A11 + A22
            {(0,0): 1, (1,1): 1},   # B11 + B22
            {(0,0): 1, (1,1): 1}    # C11 + C22: W[0*2+0]=W[0], W[1*2+1]=W[3]
        ),
        # Term 1: M2 = (A21 + A22) * B11, contributes to C21 - C22
        (
            {(1,0): 1, (1,1): 1},   # A21 + A22
            {(0,0): 1},             # B11
            {(1,0): 1, (1,1): -1}   # C21 - C22: W[1*2+0]=W[2], W[1*2+1]=W[3]
        ),
        # Term 2: M3 = A11 * (B21 - B11), contributes to -C11 + C12
        (
            {(0,0): 1},             # A11
            {(1,0): 1, (0,0): -1},  # B21 - B11
            {(0,0): -1, (0,1): 1}   # -C11 + C12: W[0*2+0]=W[0], W[0*2+1]=W[1]
        ),
        # Term 3: M4 = A22 * (B22 - B12), contributes to C11 + C21
        (
            {(1,1): 1},             # A22
            {(1,1): 1, (0,1): -1},  # B22 - B12
            {(0,0): 1, (1,0): 1}    # C11 + C21: W[0*2+0]=W[0], W[1*2+0]=W[2]
        ),
        # Term 4: M5 = (A11 + A12) * B22, contributes to -C11 + C12
        (
            {(0,0): 1, (0,1): 1},   # A11 + A12
            {(1,1): 1},             # B22
            {(0,0): -1, (0,1): 1}   # -C11 + C12: W[0*2+0]=W[0], W[0*2+1]=W[1]
        ),
        # Term 5: M6 = (A21 - A11) * (B11 + B12), contributes to C22
        (
            {(1,0): 1, (0,0): -1},  # A21 - A11
            {(0,0): 1, (0,1): 1},   # B11 + B12
            {(1,1): 1}              # C22: W[1*2+1]=W[3]
        ),
        # Term 6: M7 = (A12 - A22) * (B21 + B22), contributes to C11 + C22
        (
            {(0,1): 1, (1,1): -1},  # A12 - A22
            {(1,0): 1, (1,1): 1},   # B21 + B22
            {(0,0): 1, (1,1): 1}    # C11 + C22: W[0*2+0]=W[0], W[1*2+1]=W[3]
        ),
    ]
    
    # Schoolbook terms for 3x3
    # U: A[i3,j3], V: B[j3,k3], W: C[i3,k3] at W[k3*3+i3]
    school_terms = []
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                school_terms.append(
                    (
                        {(i3,j3): 1},           # A[i3,j3]
                        {(j3,k3): 1},           # B[j3,k3]
                        {(k3,i3): 1}            # C[i3,k3] at W[k3*3+i3]
                    )
                )
    
    # Construct full 6x6 decomposition
    rank = 7 * 27  # 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for r2 in range(7):
        U2, V2, W2 = strassen_terms[r2]
        for r3 in range(27):
            U3, V3, W3 = school_terms[r3]
            r = r2 * 27 + r3
            
            # U: global index (a*3+i3)*6 + (b*3+j3)
            for (a, b), coeff2 in U2.items():
                for (i3, j3), coeff3 in U3.items():
                    i = a * 3 + i3
                    j = b * 3 + j3
                    U[i * 6 + j][r] += coeff2 * coeff3
            
            # V: global index (b*3+j3)*6 + (c*3+k3)
            for (b, c), coeff2 in V2.items():
                for (j3, k3), coeff3 in V3.items():
                    j = b * 3 + j3
                    k = c * 3 + k3
                    V[j * 6 + k][r] += coeff2 * coeff3
            
            # W: global index (c*3+k3)*6 + (a*3+i3)
            for (c, a), coeff2 in W2.items():
                for (k3, i3), coeff3 in W3.items():
                    k = c * 3 + k3
                    i = a * 3 + i3
                    W[k * 6 + i][r] += coeff2 * coeff3
    
    elapsed = time.time() - start_time
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
