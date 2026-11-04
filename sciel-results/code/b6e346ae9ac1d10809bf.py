import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    
    # U: (n*m, rank) = (6, 44)
    # V: (m*p, rank) = (24, 44)  
    # W: (p*n, rank) = (16, 44)
    # Indexing: U[i*m+j, r] for A[i,j]
    #            V[j*p+k, r] for B[j,k]
    #            W[k*n+i, r] for C[i,k]
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # For each half of B (columns 0-3 and 4-7)
    for half in range(2):
        b_off = half * 4  # offset in B column index
        
        # Part 1: A[0:2, 0:2] * B[0:2, b_off:b_off+4]
        # Split into two 2x2 x 2x2 Strassen blocks
        for sub in range(2):
            sb = b_off + sub * 2  # start column in B for this sub-block
            
            # Strassen 7 terms for 2x2 * 2x2
            # A indices: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
            # B indices: B00=(0,sb), B01=(0,sb+1), B10=(1,sb), B11=(1,sb+1)
            # C indices: C00=(0,sb), C01=(0,sb+1), C10=(1,sb), C11=(1,sb+1)
            
            a00, a01, a10, a11 = 0, 1, 3, 4  # U indices
            b00 = 0*p + sb; b01 = 0*p + sb+1; b10 = 1*p + sb; b11 = 1*p + sb+1
            c00 = sb*n + 0; c01 = (sb+1)*n + 0; c10 = sb*n + 1; c11 = (sb+1)*n + 1
            
            # M1: (A00+A11)(B00+B11) -> C00, C11
            U[a00, col] += 1; U[a11, col] += 1
            V[b00, col] += 1; V[b11, col] += 1
            W[c00, col] += 1; W[c11, col] += 1
            col += 1
            
            # M2: (A10+A11)B00 -> C10, -C11
            U[a10, col] += 1; U[a11, col] += 1
            V[b00, col] += 1
            W[c10, col] += 1; W[c11, col] += -1
            col += 1
            
            # M3: A00(B01-B11) -> C01, C11
            U[a00, col] += 1
            V[b01, col] += 1; V[b11, col] += -1
            W[c01, col] += 1; W[c11, col] += 1
            col += 1
            
            # M4: A11(B10-B00) -> C00, C10
            U[a11, col] += 1
            V[b10, col] += 1; V[b00, col] += -1
            W[c00, col] += 1; W[c10, col] += 1
            col += 1
            
            # M5: (A00+A01)B11 -> -C00, C01
            U[a00, col] += 1; U[a01, col] += 1
            V[b11, col] += 1
            W[c00, col] += -1; W[c01, col] += 1
            col += 1
            
            # M6: (A10-A00)(B00+B01) -> C11
            U[a10, col] += 1; U[a00, col] += -1
            V[b00, col] += 1; V[b01, col] += 1
            W[c11, col] += 1
            col += 1
            
            # M7: (A01-A11)(B10+B11) -> C00
            U[a01, col] += 1; U[a11, col] += -1
            V[b10, col] += 1; V[b11, col] += 1
            W[c00, col] += 1
            col += 1
        
        # Part 2: A[0:2, 2] * B[2, b_off:b_off+4]
        # Direct terms: 2 x 4 = 8
        for i in range(2):
            for k in range(4):
                u_idx = i*m + 2  # A[i,2]
                v_idx = 2*p + (b_off + k)  # B[2, b_off+k]
                w_idx = (b_off + k)*n + i  # C[i, b_off+k]
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
