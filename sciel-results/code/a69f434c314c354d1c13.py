"""Matrix multiplication 4x4x5 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strassen 2x2x2 decomposition
    # For 2x2 * 2x2 = 2x2, rank 7
    # U indices: i*2+j for i,j in {0,1}
    # V indices: j*2+k for j,k in {0,1}  
    # W indices: k*2+i for k,i in {0,1}
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 products for 2x2 matrices:
        # M1 = (a00 + a11)(b00 + b11)
        # M2 = (a10 + a11)b00
        # M3 = a00(b10 - b11)
        # M4 = a11(b01 - b00)
        # M5 = (a00 + a01)b11
        # M6 = (a10 - a00)(b00 + b10)
        # M7 = (a01 - a11)(b01 + b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U row for (i,j), V row for (j,k), W row for (k,i)
        # U[0,0]=a00, U[0,1]=a01, U[1,0]=a10, U[1,1]=a11
        # V[0,0]=b00, V[0,1]=b01, V[1,0]=b10, V[1,1]=b11
        # W[0,0]=c00, W[0,1]=c01, W[1,0]=c10, W[1,1]=c11
        
        # Product 1: (a00+a11)(b00+b11)
        U[0, 0] = 1; U[1, 0] = 1  # a00, a11
        V[0, 0] = 1; V[1, 0] = 1  # b00, b11
        W[0, 0] = 1; W[1, 0] = 1  # c00, c11
        
        # Product 2: (a10+a11)b00
        U[1, 1] = 1; U[3, 1] = 1  # a10, a11
        V[0, 1] = 1  # b00
        W[1, 1] = 1; W[3, 1] = 1  # c10, c11
        
        # Product 3: a00(b10-b11)
        U[0, 2] = 1  # a00
        V[2, 2] = 1; V[3, 2] = -1  # b10, -b11
        W[0, 2] = 1; W[1, 2] = 1  # c00, c01
        
        # Product 4: a11(b01-b00)
        U[3, 3] = 1  # a11
        V[1, 3] = 1; V[0, 3] = -1  # b01, -b00
        W[0, 3] = 1; W[1, 3] = 1  # c00, c10
        
        # Product 5: (a00+a01)b11
        U[0, 4] = 1; U[1, 4] = 1  # a00, a01
        V[3, 4] = 1  # b11
        W[0, 4] = -1; W[1, 4] = 1  # -c00, c01
        
        # Product 6: (a10-a00)(b00+b10)
        U[1, 5] = 1; U[0, 5] = -1  # a10, -a00
        V[0, 5] = 1; V[2, 5] = 1  # b00, b10
        W[3, 5] = 1  # c11
        
        # Product 7: (a01-a11)(b01+b11)
        U[1, 6] = 1; U[3, 6] = -1  # a01, -a11
        V[1, 6] = 1; V[3, 6] = 1  # b01, b11
        W[0, 6] = 1  # c00
        
        return U, V, W
    
    # For 4x4x4, use two independent 2x2x2 Strassen decompositions
    # Split A = [[A1, A2], [A3, A4]] where each is 2x2
    # B = [[B1, B2], [B3, B4]] where each is 2x2
    # C = [[A1*B1 + A2*B3, A1*B2 + A2*B4],
    #      [A3*B1 + A4*B3, A3*B2 + A4*B4]]
    
    # Each product like A1*B1 uses 7 rank-1 terms
    # Total: 4 block products * 7 = 28 terms for 4x4x4
    
    def strassen_4x4x4():
        """Return (U, V, W) for 4x4x4 matrix multiplication using Strassen."""
        rank = 28
        U = np.zeros((16, rank))  # 4x4 = 16 entries
        V = np.zeros((16, rank))  # 4x4 = 16 entries  
        W = np.zeros((16, rank))  # 4x4 = 16 entries
        
        sU, sV, sW = strassen_2x2x2()
        
        # Block (0,0): A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # A indices: 0,1,4,5 (rows 0-1, cols 0-1)
        # B indices: 0,1,4,5 (rows 0-1, cols 0-1)
        # C indices: 0,1,4,5 (rows 0-1, cols 0-1)
        for r in range(7):
            U[0, r] = sU[0, r]; U[1, r] = sU[1, r]
            U[4, r] = sU[2, r]; U[5, r] = sU[3, r]
            V[0, r] = sV[0, r]; V[1, r] = sV[1, r]
            V[4, r] = sV[2, r]; V[5, r] = sV[3, r]
            W[0, r] = sW[0, r]; W[1, r] = sW[1, r]
            W[4, r] = sW[2, r]; W[5, r] = sW[3, r]
        
        # Block (0,1): A[0:2,0:2] * B[2:4,0:2] -> C[0:2,2:4]
        # A indices: 0,1,4,5
        # B indices: 2,3,6,7 (rows 0-1, cols 2-3)
        # C indices: 2,3,6,7 (rows 0-1, cols 2-3)
        for r in range(7):
            U[0, 7+r] = sU[0, r]; U[1, 7+r] = sU[1, r]
            U[4, 7+r] = sU[2, r]; U[5, 7+r] = sU[3, r]
            V[2, 7+r] = sV[0, r]; V[3, 7+r] = sV[1, r]
            V[6, 7+r] = sV[2, r]; V[7, 7+r] = sV[3, r]
            W[2, 7+r] = sW[0, r]; W[3, 7+r] = sW[1, r]
            W[6, 7+r] = sW[2, r]; W[7, 7+r] = sW[3, r]
        
        # Block (1,0): A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]
        # A indices: 8,9,12,13 (rows 2-3, cols 0-1)
        # B indices: 0,1,4,5
        # C indices: 8,9,12,13 (rows 2-3, cols 0-1)
        for r in range(7):
            U[8, 14+r] = sU[0, r]; U[9, 14+r] = sU[1, r]
            U[12, 14+r] = sU[2, r]; U[13, 14+r] = sU[3, r]
            V[0, 14+r] = sV[0, r]; V[1, 14+r] = sV[1, r]
            V[4, 14+r] = sV[2, r]; V[5, 14+r] = sV[3, r]
            W[8, 14+r] = sW[0, r]; W[9, 14+r] = sW[1, r]
            W[12, 14+r] = sW[2, r]; W[13, 14+r] = sW[3, r]
        
        # Block (1,1): A[2:4,0:2] * B[2:4,0:2] -> C[2:4,2:4]
        # A indices: 8,9,12,13
        # B indices: 2,3,6,7
        # C indices: 10,11,14,15 (rows 2-3, cols 2-3)
        for r in range(7):
            U[8, 21+r] = sU[0, r]; U[9, 21+r] = sU[1, r]
            U[12, 21+r] = sU[2, r]; U[13, 21+r] = sU[3, r]
            V[2, 21+r] = sV[0, r]; V[3, 21+r] = sV[1, r]
            V[6, 21+r] = sV[2, r]; V[7, 21+r] = sV[3, r]
            W[10, 21+r] = sW[0, r]; W[11, 21+r] = sW[1, r]
            W[14, 21+r] = sW[2, r]; W[15, 21+r] = sW[3, r]
        
        return U, V, W
    
    # For 4x4x5: C[i,k] = sum_j A[i,j] * B[j,k]
    # Split B into B_left (4x4) and B_right (4x1)
    # C_left (4x4) = A * B_left using Strassen rank 28
    # C_right (4x1) = A * B_right, rank 4 (schoolbook)
    # Total rank: 28 + 4 = 32
    
    def construct_4x4x5():
        """Construct rank-32 decomposition for 4x4x5."""
        rank = 32
        U = np.zeros((16, rank))  # n*m = 4*4 = 16
        V = np.zeros((20, rank))  # m*p = 4*5 = 20
        W = np.zeros((20, rank))  # p*n = 5*4 = 20
        
        # Part 1: Strassen for 4x4x4 (columns 0-3 of B, rows 0-3 of C)
        U_s, V_s, W_s = strassen_4x4x4()
        
        # Map 4x4x4 indices to 4x4x5 indices
        # A is same: U indices unchanged (0-15)
        # B: columns 0-3, so V[j*5+k] for k in 0-3
        # C: rows 0-3, so W[k*4+i] for k in 0-3
        
        for r in range(28):
            U[:, r] = U_s[:, r]
            for j in range(4):
                for k in range(4):
                    V[j*5 + k, r] = V_s[j*4 + k, r]
            for k in range(4):
                for i in range(4):
                    W[k*4 + i, r] = W_s[k*4 + i, r]
        
        # Part 2: Schoolbook for 4x4x1 (column 4 of B, row 4 of C)
        # C[*, 4] = A * B[:, 4]
        # C[i, 4] = sum_j A[i,j] * B[j, 4]
        for i in range(4):
            for j in range(4):
                r = 28 + i*4 + j
                U[i*4 + j, r] = 1
                V[j*5 + 4, r] = 1
                W[4*4 + i, r] = 1  # W[16+i, r] for k=4, i in 0-3
        
        return U, V, W
    
    # Verify the construction
    U, V, W = construct_4x4x5()
    
    # Check correctness
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    else:
        rank = U.shape[1]
    
    # Convert to lists
    def to_list(arr):
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest = round(val * 2) / 2
                if abs(val - nearest) < 1e-9:
                    if nearest == int(nearest):
                        row.append(int(nearest))
                    else:
                        row.append(nearest)
                else:
                    row.append(round(val * 2) / 2)
            result.append(row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
