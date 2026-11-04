"""
Task: Matrix multiplication 2,3,10
We construct a rank-55 tensor decomposition by splitting the 10 output columns into five 2-column blocks.
For each 2-column block, we apply Strassen's algorithm to the 2x2x2 sub-problem (rank 7) and
schoolbook for the remaining 2x1x2 part (rank 4), giving 11 per block, total 55.
"""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    rank = 55
    
    size_u = n * m  # 6
    size_v = m * p  # 30
    size_w = p * n  # 20
    
    U = [[0.0] * rank for _ in range(size_u)]
    V = [[0.0] * rank for _ in range(size_v)]
    W = [[0.0] * rank for _ in range(size_w)]
    
    # Strassen coefficients for 2x2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P1 + P2 - P6
    # C10 = P3 + P5
    # C11 = P1 - P2 + P4 + P6
    
    # For each block b (0..4), local k = 0,1 maps to global k = 2b, 2b+1
    
    col = 0
    for b in range(5):
        # Local k indices: 0 -> 2b, 1 -> 2b+1
        # Strassen part: j in {0,1}, i in {0,1}
        # U rows: i*m + j = i*3 + j
        # V rows: j*p + k = j*10 + (2b + local_k)
        # W rows: k*n + i = (2b + local_k)*2 + i
        
        # P1 = (A00+A11)(B00+B11)
        # U: A00 -> row 0*3+0=0, A11 -> row 1*3+1=4
        # V: B00 -> row 0*10+(2b+0)=2b, B11 -> row 1*10+(2b+1)=10+2b+1=11+2b
        # W: output C00 (k=0,i=0) -> row (2b+0)*2+0=4b, C01 (k=1,i=0) -> row (2b+1)*2+0=4b+2
        #     C10 (k=0,i=1) -> row (2b+0)*2+1=4b+1, C11 (k=1,i=1) -> row (2b+1)*2+1=4b+3
        
        # P1
        U[0][col] += 1.0   # A00
        U[4][col] += 1.0   # A11
        V[2*b][col] += 1.0      # B00 (j=0, k=2b)
        V[10 + 2*b + 1][col] += 1.0  # B11 (j=1, k=2b+1)
        W[4*b][col] += 1.0      # C00
        W[4*b + 2][col] += 1.0  # C01
        W[4*b + 1][col] += 0.0  # C10: not in P1
        W[4*b + 3][col] += 1.0  # C11
        
        # P2 = (A10+A11)B00
        # U: A10 -> row 1*3+0=3, A11 -> row 1*3+1=4
        # V: B00 -> row 0*10+(2b)=2b
        # W: C01 (k=1,i=0), C11 (k=1,i=1)
        col += 1
        U[3][col] += 1.0   # A10
        U[4][col] += 1.0   # A11
        V[2*b][col] += 1.0      # B00
        W[4*b + 2][col] += 1.0  # C01
        W[4*b + 3][col] += -1.0 # C11
        
        # P3 = A00(B10-B11)
        # U: A00 -> row 0
        # V: B10 -> row 1*10+(2b)=10+2b, B11 -> row 1*10+(2b+1)=11+2b
        # W: C01 (k=1,i=0), C10 (k=0,i=1)
        col += 1
        U[0][col] += 1.0   # A00
        V[10 + 2*b][col] += 1.0      # B10 (j=1, k=2b)
        V[10 + 2*b + 1][col] += -1.0 # B11 (j=1, k=2b+1)
        W[4*b + 2][col] += 1.0  # C01
        W[4*b + 1][col] += 1.0  # C10
        
        # P4 = A11(B01-B00)
        # U: A11 -> row 4
        # V: B01 -> row 0*10+(2b+1)=2b+1, B00 -> row 0*10+(2b)=2b
        # W: C00 (k=0,i=0), C11 (k=1,i=1)
        col += 1
        U[4][col] += 1.0   # A11
        V[2*b + 1][col] += 1.0      # B01 (j=0, k=2b+1)
        V[2*b][col] += -1.0         # B00 (j=0, k=2b)
        W[4*b][col] += 1.0      # C00
        W[4*b + 3][col] += 1.0  # C11
        
        # P5 = (A00+A01)B11
        # U: A00 -> row 0, A01 -> row 0*3+1=1
        # V: B11 -> row 1*10+(2b+1)=11+2b
        # W: C00 (k=0,i=0), C10 (k=0,i=1)
        col += 1
        U[0][col] += 1.0   # A00
        U[1][col] += 1.0   # A01
        V[10 + 2*b + 1][col] += 1.0  # B11
        W[4*b][col] += -1.0      # C00
        W[4*b + 1][col] += 1.0   # C10
        
        # P6 = (A10-A00)(B00+B01)
        # U: A10 -> row 3, A00 -> row 0
        # V: B00 -> row 2b, B01 -> row 2b+1
        # W: C01 (k=1,i=0), C11 (k=1,i=1)
        col += 1
        U[3][col] += 1.0   # A10
        U[0][col] += -1.0  # A00
        V[2*b][col] += 1.0      # B00
        V[2*b + 1][col] += 1.0  # B01
        W[4*b + 2][col] += -1.0 # C01
        W[4*b + 3][col] += 1.0  # C11
        
        # P7 = (A01-A11)(B10+B11)
        # U: A01 -> row 1, A11 -> row 4
        # V: B10 -> row 10+2b, B11 -> row 11+2b
        # W: C00 (k=0,i=0)
        col += 1
        U[1][col] += 1.0   # A01
        U[4][col] += -1.0  # A11
        V[10 + 2*b][col] += 1.0      # B10
        V[10 + 2*b + 1][col] += 1.0  # B11
        W[4*b][col] += 1.0      # C00
        
        # Schoolbook part: j=2, i in {0,1}, k in {0,1}
        # C[i][2b+local_k] += A[i][2] * B[2][2b+local_k]
        
        # Term for i=0, k=0: A02 * B2,2b
        col += 1
        U[0*3 + 2][col] += 1.0  # A02, row 2
        V[2*10 + 2*b][col] += 1.0  # B2,2b, row 20+2b
        W[4*b][col] += 1.0  # C00
        
        # Term for i=0, k=1: A02 * B2,2b+1
        col += 1
        U[0*3 + 2][col] += 1.0  # A02, row 2
        V[2*10 + 2*b + 1][col] += 1.0  # B2,2b+1, row 20+2b+1
        W[4*b + 2][col] += 1.0  # C01
        
        # Term for i=1, k=0: A12 * B2,2b
        col += 1
        U[1*3 + 2][col] += 1.0  # A12, row 5
        V[2*10 + 2*b][col] += 1.0  # B2,2b, row 20+2b
        W[4*b + 1][col] += 1.0  # C10
        
        # Term for i=1, k=1: A12 * B2,2b+1
        col += 1
        U[1*3 + 2][col] += 1.0  # A12, row 5
        V[2*10 + 2*b + 1][col] += 1.0  # B2,2b+1, row 20+2b+1
        W[4*b + 3][col] += 1.0  # C11
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
