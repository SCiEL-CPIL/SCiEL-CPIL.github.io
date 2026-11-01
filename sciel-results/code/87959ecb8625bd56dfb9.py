import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Block Strassen Approach
    # Split p into blocks of size 2 and 1
    # p=15 -> 7 blocks of 2, 1 block of 1
    blocks = []
    remaining = p
    while remaining > 0:
        if remaining >= 2:
            blocks.append(2)
            remaining -= 2
        else:
            blocks.append(1)
            remaining -= 1
            
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen_block(start_k):
        # Indices in V and W for this block
        # V indices: j*p + k. j in 0..1, k in start_k..start_k+1
        # W indices: k*n + i. k in start_k..start_k+1, i in 0..1
        
        # Strassen 7 products for 2x2 matrices
        # A is 2x2. B is 2x2 (columns start_k, start_k+1)
        # C is 2x2.
        
        # M1 = (A00+A11) * (B00+B11)
        # M2 = (A10+A11) * B00
        # M3 = A00 * (B01-B11)
        # M4 = A11 * (B10-B00)
        # M5 = (A00+A01) * B11
        # M6 = (A10-A00) * (B00+B01)
        # M7 = (A01-A11) * (B10+B11)
        
        # We need to map these to the tensor form U, V, W
        # Each multiplication corresponds to a rank-1 term.
        # U_r is coefficients for A.
        # V_r is coefficients for B.
        # W_r is coefficients for C.
        
        # Term 1:
        # U: A00=1, A11=1 -> [1,0,0,1] (flattened row-major: 00,01,10,11)
        # V: B00=1, B11=1 -> B00 is j=0,k=start_k. B11 is j=1,k=start_k+1.
        #    V indices: 0*p+start_k = start_k, 1*p+start_k+1 = p+start_k+1
        # W: C00=M1, C11=M1? No, C = M1 + ...
        # Strassen reconstruction:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We add 7 terms to U, V, W lists.
        
        # Term 1: (A00+A11)(B00+B11)
        U1 = [1,0,0,1]
        V1 = [0]* (m*p)
        V1[0*p + start_k] = 1
        V1[1*p + start_k+1] = 1
        W1 = [0]* (p*n)
        # This term contributes to C00 and C11?
        # In Strassen, M1 is added to C00 and C11.
        # C00 index: k=start_k, i=0 -> start_k*n + 0
        # C11 index: k=start_k+1, i=1 -> (start_k+1)*n + 1
        W1[start_k*n + 0] = 1
        W1[(start_k+1)*n + 1] = 1
        U_list.append(U1); V_list.append(V1); W_list.append(W1)
        
        # Term 2: (A10+A11)B00
        U2 = [0,0,1,1]
        V2 = [0]* (m*p)
        V2[0*p + start_k] = 1
        W2 = [0]* (p*n)
        # M2 contributes to C10 and C11 (with -)
        # C10: k=start_k, i=1 -> start_k*n + 1
        # C11: k=start_k+1, i=1 -> (start_k+1)*n + 1
        W2[start_k*n + 1] = 1
        W2[(start_k+1)*n + 1] = -1
        U_list.append(U2); V_list.append(V2); W_list.append(W2)
        
        # Term 3: A00(B01-B11)
        U3 = [1,0,0,0]
        V3 = [0]* (m*p)
        V3[0*p + start_k+1] = 1
        V3[1*p + start_k+1] = -1
        W3 = [0]* (p*n)
        # M3 contributes to C01 and C11
        # C01: k=start_k+1, i=0 -> (start_k+1)*n + 0
        # C11: k=start_k+1, i=1 -> (start_k+1)*n + 1
        W3[(start_k+1)*n + 0] = 1
        W3[(start_k+1)*n + 1] = 1
        U_list.append(U3); V_list.append(V3); W_list.append(W3)
        
        # Term 4: A11(B10-B00)
        U4 = [0,0,0,1]
        V4 = [0]* (m*p)
        V4[1*p + start_k] = 1
        V4[0*p + start_k] = -1
        W4 = [0]* (p*n)
        # M4 contributes to C00 and C10
        # C00: k=start_k, i=0 -> start_k*n + 0
        # C10: k=start_k, i=1 -> start_k*n + 1
        W4[start_k*n + 0] = 1
        W4[start_k*n + 1] = 1
        U_list.append(U4); V_list.append(V4); W_list.append(W4)
        
        # Term 5: (A00+A01)B11
        U5 = [1,1,0,0]
        V5 = [0]* (m*p)
        V5[1*p + start_k+1] = 1
        W5 = [0]* (p*n)
        # M5 contributes to C01 and C00 (with -)
        # C01: k=start_k+1, i=0 -> (start_k+1)*n + 0
        # C00: k=start_k, i=0 -> start_k*n + 0
        W5[(start_k+1)*n + 0] = 1
        W5[start_k*n + 0] = -1
        U_list.append(U5); V_list.append(V5); W_list.append(W5)
        
        # Term 6: (A10-A00)(B00+B01)
        U6 = [ -1,0,1,0]
        V6 = [0]* (m*p)
        V6[0*p + start_k] = 1
        V6[0*p + start_k+1] = 1
        W6 = [0]* (p*n)
        # M6 contributes to C11
        # C11: k=start_k+1, i=1 -> (start_k+1)*n + 1
        W6[(start_k+1)*n + 1] = 1
        U_list.append(U6); V_list.append(V6); W_list.append(W6)
        
        # Term 7: (A01-A11)(B10+B11)
        U7 = [0,1,0,-1]
        V7 = [0]* (m*p)
        V7[1*p + start_k] = 1
        V7[1*p + start_k+1] = 1
        W7 = [0]* (p*n)
        # M7 contributes to C00
        # C00: k=start_k, i=0 -> start_k*n + 0
        W7[start_k*n + 0] = 1
        U_list.append(U7); V_list.append(V7); W_list.append(W7)

    def add_schoolbook_block(start_k, size):
        for j in range(m):
            for k in range(size):
                for i in range(n):
                    U_i = [0]*(n*m)
                    U_i[i*m+j] = 1
                    V_i = [0]*(m*p)
                    V_i[j*p + start_k+k] = 1
                    W_i = [0]*(p*n)
                    W_i[(start_k+k)*n + i] = 1
                    U_list.append(U_i)
                    V_list.append(V_i)
                    W_list.append(W_i)

    current_k = 0
    for block_size in blocks:
        if block_size == 2:
            add_strassen_block(current_k)
        else:
            add_schoolbook_block(current_k, block_size)
        current_k += block_size

    rank_block = len(U_list)
    if rank_block < best_rank:
        best_rank = rank_block
        best_U = U_list
        best_V = V_list
        best_W = W_list

    return {"U": best_U, "V": best_V, "W": best_W}
