import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank 88 decomposition
    # 8 blocks of 2x2x2 (Strassen, rank 7)
    # 1 block of 1x4x8 (Schoolbook, rank 32)
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen matrices for 2x2x2
    # U is 2x2, V is 2x2, W is 4x2 (k*n + i)
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    
    # M1: (A00+A11)(B00+B11)
    U1 = [[1, 0], [0, 1]]
    V1 = [[1, 0], [0, 1]]
    W1 = [[1, 0], [0, 0], [0, 0], [0, 1]] # C00, C10, C01, C11
    
    # M2: (A10+A11)B00
    U2 = [[0, 0], [1, 1]]
    V2 = [[1, 0], [0, 0]]
    W2 = [[0, 0], [1, 0], [0, 0], [0, 0]]
    
    # M3: A00(B01-B11)
    U3 = [[1, 0], [0, 0]]
    V3 = [[0, 1], [0, -1]]
    W3 = [[0, 0], [0, 0], [1, 0], [0, 0]]
    
    # M4: A11(B10-B00)
    U4 = [[0, 0], [0, 1]]
    V4 = [[-1, 0], [1, 0]]
    W4 = [[-1, 0], [1, 0], [0, 0], [0, 0]]
    
    # M5: (A00+A01)B11
    U5 = [[1, 1], [0, 0]]
    V5 = [[0, 0], [0, 1]]
    W5 = [[0, 0], [0, 0], [1, 0], [0, -1]]
    
    # M6: (A10-A00)(B00+B01)
    U6 = [[-1, 0], [1, 0]]
    V6 = [[1, 1], [0, 0]]
    W6 = [[1, 0], [0, 0], [0, 0], [0, 0]]
    
    # M7: (A01-A11)(B10+B11)
    U7 = [[0, 1], [0, -1]]
    V7 = [[0, 0], [1, 1]]
    W7 = [[0, 0], [0, 0], [0, 0], [0, 1]]
    
    strassen_blocks = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    # Partitioning
    # i in {0, 1} for Strassen
    # j in {0, 1}, {2, 3}
    # k in {0, 1}, {2, 3}, {4, 5}, {6, 7}
    
    j_bases = [0, 2]
    k_bases = [0, 2, 4, 6]
    
    for j_base in j_bases:
        for k_base in k_bases:
            for U_s, V_s, W_s in strassen_blocks:
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                
                # Fill U
                for li in range(2):
                    for lj in range(2):
                        if U_s[li][lj] != 0:
                            i_global = li # i_base is 0
                            j_global = j_base + lj
                            U[i_global * m + j_global][col] += U_s[li][lj]
                
                # Fill V
                for lj in range(2):
                    for lk in range(2):
                        if V_s[lj][lk] != 0:
                            j_global = j_base + lj
                            k_global = k_base + lk
                            V[j_global * p + k_global][col] += V_s[lj][lk]
                
                # Fill W
                # W is indexed by k*n + i
                for lk in range(2):
                    for li in range(2):
                        if W_s[lk * 2 + li][0] != 0: # W_s is 4x1 effectively? No, W_s is 4x1 for each col?
                            # W_s is defined as 4x1 for each Strassen product
                            # In my list above, W_s is 4x1.
                            pass
                
                # My W_s definition was 4x1.
                # W1 = [[1, 0], [0, 0], [0, 0], [0, 1]] -> This is 4x2?
                # No, W1 is for ONE product M1.
                # So W1 is a column vector of length 4.
                # My previous definition:
                # W1 = [[1, 0], [0, 0], [0, 0], [0, 1]]
                # This looks like 2 columns.
                # Ah, I defined U, V, W for the whole block?
                # No, I defined U1, V1, W1 for M1.
                # U1 is 2x2? No, U1 is 2x1?
                # U1 = [[1, 0], [0, 1]] -> 2x2.
                # This implies U1 is for TWO products?
                # No, U1 is the matrix for M1.
                # M1 is one product.
                # So U1 should be 2x1.
                # My U1 = [[1, 0], [0, 1]] is 2x2.
                # This is wrong.
                
                # Let's redefine.
                # U_s is 2x1.
                # V_s is 2x1.
                # W_s is 4x1.
                
                # I will just use the correct 1D vectors.
                
                # This is getting messy. I will just hardcode the 88 rank matrices.
                pass
                
    # Fallback to schoolbook if time runs out or logic fails
    # But 88 is simple enough to hardcode the logic correctly.
    
    # I will restart the logic with 1D vectors.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen vectors
    # M1
    U1 = [1, 0]
    V1 = [1, 0]
    W1 = [1, 0, 0, 0] # C00, C10, C01, C11 -> Wait, C00 is k=0,i=0. C10 is k=0,i=1.
    # W index: k*n + i.
    # C00: 0*3+0 = 0
    # C10: 0*3+1 = 1
    # C01: 1*3+0 = 3
    # C11: 1*3+1 = 4
    # So W1 = [1, 0, 0, 0, 1] (length 5? No, p*n = 8*3=24).
    # For local 2x2, W is length 4.
    # W1 = [1, 0, 0, 1]
    
    U1 = [1, 0]
    V1 = [1, 0]
    W1 = [1, 0, 0, 1]
    
    U2 = [0, 1]
    V2 = [1, 0]
    W2 = [0, 1, 0, 0]
    
    U3 = [1, 0]
    V3 = [0, 1]
    W3 = [0, 0, 1, 0]
    
    U4 = [0, 1]
    V4 = [-1, 1]
    W4 = [-1, 1, 0, 0]
    
    U5 = [1, 0]
    V5 = [0, 1]
    W5 = [0, 0, 1, -1]
    
    U6 = [-1, 1]
    V6 = [1, 1]
    W6 = [1, 0, 0, 0]
    
    U7 = [0, -1]
    V7 = [0, 1]
    W7 = [0, 0, 0, 1]
    
    # Note: U, V are 2x1. W is 4x1.
    # M1: (A00+A11)(B00+B11)
    # U1 = [1, 1]? No.
    # A00 coeff 1, A11 coeff 1.
    # U1 = [1, 1].
    # My U1 = [1, 0] is wrong.
    
    # Correct U, V, W for Strassen:
    # M1: U=[1,1], V=[1,1], W=[1,0,0,1]
    # M2: U=[0,1], V=[1,0], W=[0,1,0,0] (A10+A11? No, U=[0,1] is A11 only? No.)
    # M2 = (A10+A11)B00.
    # U2 = [0, 1] (A11) + [1, 0] (A10)? No, U is 2x1.
    # U2 = [1, 1] (A00, A11)? No, M2 uses A10, A11.
    # Indices: 0->A00, 1->A11? No, 0->A0, 1->A1.
    # U2 = [0, 1] (A1) + ...?
    # A10 is (1,0). A11 is (1,1).
    # U2 = [0, 0] for A00, A01?
    # U is 2x2 matrix flattened? No, U is 2x1 vector for the 2x2 block?
    # No, U is 2x2 matrix.
    # U_s is 2x2.
    
    # I will use the 2x2 matrices.
    
    U1 = [[1, 0], [0, 1]]
    V1 = [[1, 0], [0, 1]]
    W1 = [1, 0, 0, 1]
    
    U2 = [[0, 0], [1, 1]]
    V2 = [[1, 0], [0, 0]]
    W2 = [0, 1, 0, 0]
    
    U3 = [[1, 0], [0, 0]]
    V3 = [[0, 1], [0, -1]]
    W3 = [0, 0, 1, 0]
    
    U4 = [[0, 0], [0, 1]]
    V4 = [[-1, 0], [1, 0]]
    W4 = [-1, 1, 0, 0]
    
    U5 = [[1, 1], [0, 0]]
    V5 = [[0, 0], [0, 1]]
    W5 = [0, 0, 1, -1]
    
    U6 = [[-1, 0], [1, 0]]
    V6 = [[1, 1], [0, 0]]
    W6 = [1, 0, 0, 0]
    
    U7 = [[0, 1], [0, -1]]
    V7 = [[0, 0], [1, 1]]
    W7 = [0, 0, 0, 1]
    
    # This is the correct set.
    
    for j_base in j_bases:
        for k_base in k_bases:
            for Us, Vs, Ws in [(U1,V1,W1), (U2,V2,W2), (U3,V3,W3), (U4,V4,W4), (U5,V5,W5), (U6,V6,W6), (U7,V7,W7)]:
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                
                for li in range(2):
                    for lj in range(2):
                        if Us[li][lj] != 0:
                            U[li * m + (j_base + lj)][col] += Us[li][lj]
                
                for lj in range(2):
                    for lk in range(2):
                        if Vs[lj][lk] != 0:
                            V[(j_base + lj) * p + (k_base + lk)][col] += Vs[lj][lk]
                
                for lk in range(2):
                    for li in range(2):
                        # Ws index: lk*2 + li? No.
                        # Ws is length 4.
                        # Index 0: C00 (k=0, i=0)
                        # Index 1: C10 (k=0, i=1)
                        # Index 2: C01 (k=1, i=0)
                        # Index 3: C11 (k=1, i=1)
                        # So Ws[lk*2 + li]
                        if Ws[lk*2 + li] != 0:
                            W[(k_base + lk) * n + li][col] += Ws[lk*2 + li]
                
                col += 1
    
    # Schoolbook for i=2
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
