import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 4, 6
    
    # Strassen 2x2x2 decomposition
    # For 2x2 matrices A, B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Strassen terms as (U_coeffs, V_coeffs, W_coeffs) for 2x2x2
    # U is indexed by (i, j) for A[i,j], V by (j, k) for B[j,k], W by (k, i) for C[k,i]
    # For 2x2x2: U has 4 entries (i*2+j), V has 4 entries (j*2+k), W has 4 entries (k*2+i)
    
    strassen_terms = []
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> U[0]=1, U[3]=1
    # V: B00=1, B11=1 -> V[0]=1, V[3]=1
    # W: C00+=1, C11+=1 -> W[0]=1, W[3]=1
    strassen_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A10+A11)B00
    # U: A10=1, A11=1 -> U[2]=1, U[3]=1
    # V: B00=1 -> V[0]=1
    # W: C10+=1 -> W[1]=1
    strassen_terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,0]))
    
    # M3 = A00(B01-B11)
    # U: A00=1 -> U[0]=1
    # V: B01=1, B11=-1 -> V[1]=1, V[3]=-1
    # W: C01+=1, C11+=1 -> W[1]=1, W[3]=1
    strassen_terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    
    # M4 = A11(B10-B00)
    # U: A11=1 -> U[3]=1
    # V: B10=1, B00=-1 -> V[2]=1, V[0]=-1
    # W: C00+=1, C10+=1 -> W[0]=1, W[1]=1
    strassen_terms.append(([0,0,0,1], [-1,0,1,0], [1,1,0,0]))
    
    # M5 = (A00+A10)B11
    # U: A00=1, A10=1 -> U[0]=1, U[2]=1
    # V: B11=1 -> V[3]=1
    # W: C00-=1, C01+=1 -> W[0]=-1, W[1]=1
    strassen_terms.append(([1,0,1,0], [0,0,0,1], [-1,1,0,0]))
    
    # M6 = (A01-A00)(B00+B10)
    # U: A01=1, A00=-1 -> U[1]=1, U[0]=-1
    # V: B00=1, B10=1 -> V[0]=1, V[2]=1
    # W: C11+=1 -> W[3]=1
    strassen_terms.append(([-1,1,0,0], [1,0,1,0], [0,0,0,1]))
    
    # M7 = (A10-A11)(B01+B11)
    # U: A10=1, A11=-1 -> U[2]=1, U[3]=-1
    # V: B01=1, B11=1 -> V[1]=1, V[3]=1
    # W: C00+=1 -> W[0]=1
    strassen_terms.append(([0,0,1,-1], [0,1,0,1], [1,0,0,0]))
    
    # Now build the full 2x4x6 decomposition
    # A is 2x4: split into A[:, 0:2] and A[:, 2:4]
    # B is 4x6: split rows into B[0:2, :] and B[2:4, :]
    # C is 2x6: split columns into C[:, 0:2], C[:, 2:4], C[:, 4:6]
    
    # For each pair (A_block, B_block) where A_block is 2x2 and B_block is 2x2,
    # and for each 2-column block of output, we use Strassen.
    
    # Total: 2 A-blocks × 3 B-row-blocks × 3 C-col-blocks... 
    # Wait, let me reconsider.
    
    # A is 2x4. B is 4x6.
    # We can think of this as: C = A @ B
    # Split A into [A1 | A2] where A1 is 2x2, A2 is 2x2
    # Split B into [B1; B2] where B1 is 2x6, B2 is 2x6
    # C = A1 @ B1 + A2 @ B2
    # Each A_i @ B_i is 2x6, which we split into three 2x2 multiplications.
    
    # So: C = sum over i in {1,2} of (A_i @ B_i)
    # Each A_i @ B_i: A_i is 2x2, B_i is 2x6
    # Split B_i into [B_i1 | B_i2 | B_i3] each 2x2
    # A_i @ B_i = (A_i @ B_i1) | (A_i @ B_i2) | (A_i @ B_i3)
    # Each A_i @ B_ij is a 2x2x2 Strassen multiplication.
    
    # Total Strassen blocks: 2 (A blocks) × 3 (B column blocks) = 6
    # Each uses 7 rank terms, so total rank = 6 × 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    col = 0
    for a_block in range(2):  # A[:, a_block*2 : a_block*2+2]
        for b_col_block in range(3):  # B[:, b_col_block*2 : b_col_block*2+2]
            # Local indices for 2x2x2
            # A local: i in {0,1}, j in {0,1} -> global j = a_block*2 + j
            # B local: j in {0,1}, k in {0,1} -> global j = a_block*2 + j, global k = b_col_block*2 + k
            # C local: k in {0,1}, i in {0,1} -> global k = b_col_block*2 + k, global i = i
            
            for s in range(7):
                u_local, v_local, w_local = strassen_terms[s]
                for i in range(2):
                    for j in range(2):
                        if u_local[i*2+j] != 0:
                            U[i*m + (a_block*2 + j)][col] = u_local[i*2+j]
                for j in range(2):
                    for k in range(2):
                        if v_local[j*2+k] != 0:
                            V[(a_block*2 + j)*p + (b_col_block*2 + k)][col] = v_local[j*2+k]
                for k in range(2):
                    for i in range(2):
                        if w_local[k*2+i] != 0:
                            W[(b_col_block*2 + k)*n + i][col] = w_local[k*2+i]
                col += 1
    
    return {"U": U, "V": V, "W": W}

