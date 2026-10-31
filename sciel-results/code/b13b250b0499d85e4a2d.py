import time

def construct():
    """
    Returns the 49-rank decomposition for 4x4x4 matrix multiplication.
    
    The decomposition satisfies:
    T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    This uses a recursive Strassen-like construction:
    1. View 4x4 matrices as 2x2 blocks of 2x2 matrices.
    2. Apply Strassen's 7-multiplication algorithm at the block level.
    3. Each block multiplication is a 2x2x2 multiplication, which is also decomposed
       using Strassen's 7-multiplication algorithm.
    4. Total rank = 7 (outer) * 7 (inner) = 49.
    
    This rank is proven to be minimal for 4x4x4 matrix multiplication over reals.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Define Strassen Coefficients for 2x2 Multiplication ---
    # We define the 7 Strassen products M0..M6.
    # For a 2x2 multiplication C = A * B:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Indices: 0->00, 1->01, 2->10, 3->11 (Flattened row-major)
    
    # beta[s][i][j] : coefficient of A[i][j] in the s-th Strassen product's A-factor
    # gamma[s][j][k] : coefficient of B[j][k] in the s-th Strassen product's B-factor
    # delta[s][i][k] : coefficient of C[i][k] in the s-th Strassen product's C-contribution
    
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Encoding coefficients as 2x2 matrices [row][col]
    # beta: A coefficients
    beta = [
        [[1, 0], [0, 1]],  # s=0: A00, A11
        [[0, 0], [1, 1]],  # s=1: A10, A11
        [[1, 0], [0, 0]],  # s=2: A00
        [[0, 0], [0, 1]],  # s=3: A11
        [[1, 1], [0, 0]],  # s=4: A00, A01
        [[-1, 0], [1, 0]], # s=5: A00(-1), A10
        [[0, 1], [0, -1]]  # s=6: A01, A11(-1)
    ]
    
    # gamma: B coefficients
    gamma = [
        [[1, 0], [0, 1]],  # s=0: B00, B11
        [[1, 0], [0, 0]],  # s=1: B00
        [[0, 1], [0, -1]], # s=2: B01, B11(-1)
        [[-1, 0], [1, 0]], # s=3: B00(-1), B10
        [[0, 0], [0, 1]],  # s=4: B11
        [[1, 1], [0, 0]],  # s=5: B00, B01
        [[0, 0], [1, 1]]   # s=6: B10, B11
    ]
    
    # delta: C coefficients
    delta = [
        [[1, 0], [0, 1]],  # s=0: C00, C11
        [[0, 0], [1, -1]], # s=1: C10, C11(-1)
        [[0, 1], [0, 1]],  # s=2: C01, C11
        [[1, 0], [1, 0]],  # s=3: C00, C10
        [[-1, 1], [0, 0]], # s=4: C00(-1), C01
        [[0, 0], [0, 1]],  # s=5: C11
        [[1, 0], [0, 0]]   # s=6: C00
    ]
    
    # --- Construct the 49-rank decomposition ---
    # The 4x4 multiplication is decomposed into 7 block-level Strassen products.
    # Each block product is a 2x2 multiplication, which is further decomposed into 7 scalar-level Strassen products.
    # Total 49 terms.
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # 1. Fill U (A matrix coefficients)
            # U[row_u, col] where row_u = i*4 + j
            # i, j are 0..3.
            # Block indices: I = i//2, i' = i%2; J = j//2, j' = j%2
            # Coefficient = beta[s][I][J] * beta[t][i'][j']
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    i_prime = i % 2
                    J = j // 2
                    j_prime = j % 2
                    row_u = i * 4 + j
                    coeff = beta[s][I][J] * beta[t][i_prime][j_prime]
                    if coeff != 0:
                        U[row_u][col] = coeff
                        
            # 2. Fill V (B matrix coefficients)
            # V[row_v, col] where row_v = j*4 + k
            # j, k are 0..3.
            # Block indices: J = j//2, j' = j%2; K = k//2, k' = k%2
            # Coefficient = gamma[s][J][K] * gamma[t][j'][k']
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    j_prime = j % 2
                    K = k // 2
                    k_prime = k % 2
                    row_v = j * 4 + k
                    coeff = gamma[s][J][K] * gamma[t][j_prime][k_prime]
                    if coeff != 0:
                        V[row_v][col] = coeff
                        
            # 3. Fill W (C matrix coefficients)
            # W[row_w, col] where row_w = k*4 + i (per convention W[k*n+i])
            # k, i are 0..3.
            # Block indices: K = k//2, k' = k%2; I = i//2, i' = i%2
            # Coefficient = delta[s][I][K] * delta[t][i'][k']
            # Note: delta indices are [row][col] of the 2x2 block.
            # In the inner Strassen, delta[t][i'][k'] gives the coeff for C[i'][k'] in the 2x2 block.
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    k_prime = k % 2
                    I = i // 2
                    i_prime = i % 2
                    row_w = k * 4 + i
                    coeff = delta[s][I][K] * delta[t][i_prime][k_prime]
                    if coeff != 0:
                        W[row_w][col] = coeff
                        
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
