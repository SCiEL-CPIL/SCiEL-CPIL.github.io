"""Matrix multiplication 4x5x6 - rank-minimal exact tensor decomposition."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strategy: Block decomposition
    # Split m=5 into 2+3
    # C = A[:,:2]@B[:2,:] + A[:,2:]@B[2:,:]
    # Block 1: 4x2 @ 2x6
    # Block 2: 4x3 @ 3x6
    
    # For 4x2x6: split n=4 into 2+2
    #   2x2x6 + 2x2x6
    # For 2x2x6: split p=6 into 2+2+2
    #   3 * (2x2x2), each Strassen rank 7 → total 21
    # So 4x2x6: 21+21 = 42
    
    # For 4x3x6: split m=3 into 1+2
    #   4x1x6 + 4x2x6
    # 4x1x6: rank 4 (each row independently)
    # 4x2x6: rank 42
    # So 4x3x6: 4+42 = 46
    
    # Total: 42 + 46 = 88
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(U_block, V_block, W_block):
        """Append block matrices to the lists."""
        U_list.append(U_block)
        V_list.append(V_block)
        W_list.append(W_block)
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        # n=2, m=2, p=2
        # U: (4,7), V: (4,7), W: (4,7)
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Using the correct Strassen decomposition
        # After careful verification, use the following:
        # This is the standard rank-7 decomposition of the 2x2x2 all-ones tensor
        
        # Term 0: (a+d)(e+h) contributes to positions where...
        # Let me use a known correct decomposition.
        
        # From literature, the 7 vectors for 2x2x2 multiplication tensor:
        # U rows: [a, b, c, d] (indices 0,1,2,3)
        # V rows: [e, f, g, h] (indices 0,1,2,3)  
        # W rows: [C00, C01, C10, C11] (indices 0,1,2,3) where C[k*n+i]
        
        # Actually for n=m=p=2, W[k*2+i]:
        # W[0]=W[k=0,i=0], W[1]=W[k=0,i=1], W[2]=W[k=1,i=0], W[3]=W[k=1,i=1]
        
        # I'll use a verified decomposition. Let me construct it by solving.
        # For now, use schoolbook for 2x2x2 (rank 8) as fallback if Strassen is wrong.
        
        # Actually, let me just use schoolbook for all sub-blocks to ensure correctness,
        # and then I can try to optimize later.
        
        # Schoolbook 2x2x2: rank 8
        U = [[0]*8 for _ in range(4)]
        V = [[0]*8 for _ in range(4)]
        W = [[0]*8 for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j][col] = 1
                    V[j*2+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W
    
    # Given the risk of Strassen errors, let me use a safer approach:
    # Use schoolbook for all sub-blocks but with a smarter split that might help.
    # Actually, schoolbook always gives n*m*p regardless of splitting.
    # So I NEED Strassen or some other low-rank method to beat 120.
    
    # Let me implement Strassen correctly by solving the linear system.
    # For 2x2x2, I'll find 7 vectors u_c, v_c, w_c such that
    # sum_c u_c[i*2+j] * v_c[j*2+k] * w_c[k*2+i] = 1 for all i,j,k in {0,1}
    
    # This is a nonlinear system, but I know Strassen works. Let me hardcode a known solution.
    
    # Known rank-7 decomposition of the 2x2x2 multiplication tensor:
    # Reference: Various sources give this.
    
    # After careful derivation, here's a correct set:
    # The 7 terms correspond to the 7 products in Strassen's algorithm.
    
    # Let me define it based on the bilinear forms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*2+j, c] = coeff of A[i,j] in M_c
    # V[j*2+k, c] = coeff of B[j,k] in M_c
    # W[k*2+i, c] = coeff of M_c in C[i,k]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11): c=0
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2 = (A10+A11)B00: c=1
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10
    
    # M3 = A00(B01-B11): c=2
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1  # B11
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4 = A11(B10-B00): c=3
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1  # B00
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5 = (A00+A01)B11: c=4
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1  # C00
    W[1][4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01): c=5
    U[2][5] = 1  # A10
    U[0][5] = -1  # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[3][5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11): c=6
    U[1][6] = 1  # A01
    U[3][6] = -1  # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00
    
    # Verify in code
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U[i*2+j][c] * V[j*2+k][c] * W[k*2+i][c]
                if s != 1:
                    # Fallback to schoolbook
                    U = [[0]*8 for _ in range(4)]
                    V = [[0]*8 for _ in range(4)]
                    W = [[0]*8 for _ in range(4)]
                    col = 0
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                U[ii*2+jj][col] = 1
                                V[jj*2+kk][col] = 1
                                W[kk*2+ii][col] = 1
                                col += 1
                    return U, V, W
    
    return U, V, W

# The above function definition is getting messy. Let me restructure.
# EVOLVE-BLOCK-END
