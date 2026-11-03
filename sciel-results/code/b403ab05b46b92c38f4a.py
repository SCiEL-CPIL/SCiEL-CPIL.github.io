import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 100  # Leave margin for 120s limit
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=5
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    
    # Strategy: Block decomposition using Strassen for 2x2x2 blocks
    # Split M (4) into 2+2.
    # Split N (3) into 2+1.
    # Split P (5) into 2+2+1.
    
    # Blocks:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen 7)
    # 2. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen 7)
    # 3. A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (Schoolbook 4)
    # 4. A[2:3, 0:2] * B[0:2, 0:5] -> C[2:3, 0:5] (Schoolbook 10)
    # 5. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (Strassen 7)
    # 6. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (Strassen 7)
    # 7. A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5] (Schoolbook 4)
    # 8. A[2:3, 2:4] * B[2:4, 0:5] -> C[2:3, 0:5] (Schoolbook 10)
    
    R_total = 7 + 7 + 4 + 10 + 7 + 7 + 4 + 10
    R_total = 56
    
    U = [[0] * R_total for _ in range(n * m)]
    V = [[0] * R_total for _ in range(m * p)]
    W = [[0] * R_total for _ in range(p * n)]
    
    offset = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen_222(offset, a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        # a_r0, a_c0: top-left of A block
        # b_r0, b_c0: top-left of B block
        # c_r0, c_c0: top-left of C block
        # A indices: (a_r0+i)*m + (a_c0+j)
        # B indices: (b_r0+j)*p + (b_c0+k)
        # C indices: (c_r0+k)*n + (c_c0+i)
        
        # M1 = (A00+A11)(B00+B11)
        U[(a_r0+0)*m + (a_c0+0)][offset] += 1
        U[(a_r0+1)*m + (a_c0+1)][offset] += 1
        V[(b_r0+0)*p + (b_c0+0)][offset] += 1
        V[(b_r0+1)*p + (b_c0+1)][offset] += 1
        W[(c_r0+0)*n + (c_c0+0)][offset] += 1
        W[(c_r0+1)*n + (c_c0+1)][offset] += 1
        
        # M2 = (A10+A11)B00
        U[(a_r0+1)*m + (a_c0+0)][offset+1] += 1
        U[(a_r0+1)*m + (a_c0+1)][offset+1] += 1
        V[(b_r0+0)*p + (b_c0+0)][offset+1] += 1
        W[(c_r0+1)*n + (c_c0+0)][offset+1] += 1
        
        # M3 = A00(B01-B11)
        U[(a_r0+0)*m + (a_c0+0)][offset+2] += 1
        V[(b_r0+0)*p + (b_c0+1)][offset+2] += 1
        V[(b_r0+1)*p + (b_c0+1)][offset+2] -= 1
        W[(c_r0+0)*n + (c_c0+1)][offset+2] += 1
        W[(c_r0+1)*n + (c_c0+1)][offset+2] += 1
        
        # M4 = A11(B10-B00)
        U[(a_r0+1)*m + (a_c0+1)][offset+3] += 1
        V[(b_r0+1)*p + (b_c0+0)][offset+3] += 1
        V[(b_r0+0)*p + (b_c0+0)][offset+3] -= 1
        W[(c_r0+1)*n + (c_c0+0)][offset+3] += 1
        W[(c_r0+0)*n + (c_c0+0)][offset+3] += 1
        
        # M5 = (A00+A01)B11
        U[(a_r0+0)*m + (a_c0+0)][offset+4] += 1
        U[(a_r0+0)*m + (a_c0+1)][offset+4] += 1
        V[(b_r0+1)*p + (b_c0+1)][offset+4] += 1
        W[(c_r0+0)*n + (c_c0+0)][offset+4] -= 1
        W[(c_r0+0)*n + (c_c0+1)][offset+4] += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(a_r0+1)*m + (a_c0+0)][offset+5] += 1
        U[(a_r0+0)*m + (a_c0+0)][offset+5] -= 1
        V[(b_r0+0)*p + (b_c0+0)][offset+5] += 1
        V[(b_r0+0)*p + (b_c0+1)][offset+5] += 1
        W[(c_r0+1)*n + (c_c0+1)][offset+5] += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[(a_r0+0)*m + (a_c0+1)][offset+6] += 1
        U[(a_r0+1)*m + (a_c0+1)][offset+6] -= 1
        V[(b_r0+1)*p + (b_c0+0)][offset+6] += 1
        V[(b_r0+1)*p + (b_c0+1)][offset+6] += 1
        W[(c_r0+0)*n + (c_c0+0)][offset+6] += 1
        W[(c_r0+1)*n + (c_c0+1)][offset+6] += 1
        
        return offset + 7

    # Helper to add Schoolbook 2x2x1
    def add_schoolbook_221(offset, a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        # A[2x2] * B[2x1] -> C[2x1]
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        
        # Term 1: A00 B00 -> C00
        U[(a_r0+0)*m + (a_c0+0)][offset] += 1
        V[(b_r0+0)*p + (b_c0+0)][offset] += 1
        W[(c_r0+0)*n + (c_c0+0)][offset] += 1
        
        # Term 2: A01 B10 -> C00
        U[(a_r0+0)*m + (a_c0+1)][offset+1] += 1
        V[(b_r0+1)*p + (b_c0+0)][offset+1] += 1
        W[(c_r0+0)*n + (c_c0+0)][offset+1] += 1
        
        # Term 3: A10 B00 -> C10
        U[(a_r0+1)*m + (a_c0+0)][offset+2] += 1
        V[(b_r0+0)*p + (b_c0+0)][offset+2] += 1
        W[(c_r0+1)*n + (c_c0+0)][offset+2] += 1
        
        # Term 4: A11 B10 -> C10
        U[(a_r0+1)*m + (a_c0+1)][offset+3] += 1
        V[(b_r0+1)*p + (b_c0+0)][offset+3] += 1
        W[(c_r0+1)*n + (c_c0+0)][offset+3] += 1
        
        return offset + 4

    # Helper to add Schoolbook 1x2x5
    def add_schoolbook_125(offset, a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        # A[1x2] * B[2x5] -> C[1x5]
        # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k]
        
        for k in range(5):
            # Term 1: A00 B0k -> C0k
            U[(a_r0+0)*m + (a_c0+0)][offset] += 1
            V[(b_r0+0)*p + (b_c0+k)][offset] += 1
            W[(c_r0+0)*n + (c_c0+k)][offset] += 1
            offset += 1
            
            # Term 2: A01 B1k -> C0k
            U[(a_r0+0)*m + (a_c0+1)][offset] += 1
            V[(b_r0+1)*p + (b_c0+k)][offset] += 1
            W[(c_r0+0)*n + (c_c0+k)][offset] += 1
            offset += 1
            
        return offset

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    offset = add_strassen_222(offset, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    offset = add_strassen_222(offset, 0, 0, 0, 2, 0, 2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    offset = add_schoolbook_221(offset, 0, 0, 0, 4, 0, 4)
    
    # Block 4: A[2:3, 0:2] * B[0:2, 0:5] -> C[2:3, 0:5]
    offset = add_schoolbook_125(offset, 2, 0, 0, 0, 2, 0)
    
    # Block 5: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    offset = add_strassen_222(offset, 0, 2, 2, 0, 0, 0)
    
    # Block 6: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    offset = add_strassen_222(offset, 0, 2, 2, 2, 0, 2)
    
    # Block 7: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    offset = add_schoolbook_221(offset, 0, 2, 2, 4, 0, 4)
    
    # Block 8: A[2:3, 2:4] * B[2:4, 0:5] -> C[2:3, 0:5]
    offset = add_schoolbook_125(offset, 2, 2, 2, 0, 2, 0)
    
    if offset != R_total:
        # Fallback to schoolbook if something went wrong (should not happen)
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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
# EVOLVE-BLOCK-END
