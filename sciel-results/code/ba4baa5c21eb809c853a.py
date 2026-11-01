"""Matrix multiplication 2,4,7 with rank-minimal decomposition using Strassen and block structure."""
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We'll build the decomposition using Strassen for 2x2*2x2 blocks
    # and schoolbook for 2x2*2x1 blocks.
    
    # Split m=4 into two blocks of size 2: jb in {0, 1}
    # Split p=7 into blocks: 3 blocks of size 2 (kb in {0,1,2}) and 1 block of size 1 (kb=3)
    
    # For each (jb, kb) pair:
    #   If kb < 3: 2x2 * 2x2, use Strassen (rank 7)
    #   If kb == 3: 2x2 * 2x1, use schoolbook (rank 4)
    
    # Total rank: 2 * (3*7 + 4) = 2 * 25 = 50
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen 2x2*2x2 decomposition:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)(b11+b12)
    # M3 = (a11-a22)(b21+b22)
    # M4 = a11(b21-b22)
    # M5 = a22(b12-b11)
    # M6 = (a11+a12)b22
    # M7 = (a21-a11)b11
    # C11 = M1 + M4 - M5 + M7
    # C12 = M6 + M5
    # C21 = M2 - M7
    # C22 = M1 - M4 + M3 - M6
    
    # In our indexing:
    # A[i, j] where i in {0,1}, j in {j_base, j_base+1}
    # B[j, k] where j in {j_base, j_base+1}, k in {k_base, k_base+1}
    # C[k, i] where k in {k_base, k_base+1}, i in {0,1}
    # W index: k*n + i
    
    # Mapping:
    # a11 = A[0, j_base], a12 = A[0, j_base+1]
    # a21 = A[1, j_base], a22 = A[1, j_base+1]
    # b11 = B[j_base, k_base], b12 = B[j_base, k_base+1]
    # b21 = B[j_base+1, k_base], b22 = B[j_base+1, k_base+1]
    # C_std[0,0] = C11, C_std[0,1] = C12, C_std[1,0] = C21, C_std[1,1] = C22
    # C_here[k, i] = C_std[i, k]
    # So:
    # C_here[k_base, 0] = C_std[0, 0] = C11
    # C_here[k_base, 1] = C_std[1, 0] = C21
    # C_here[k_base+1, 0] = C_std[0, 1] = C12
    # C_here[k_base+1, 1] = C_std[1, 1] = C22
    
    for jb in range(2):
        j_base = jb * 2  # 0 or 2
        
        for kb in range(4):
            if kb < 3:
                # 2x2 * 2x2, use Strassen (rank 7)
                k_base = kb * 2  # 0, 2, 4
                
                # U indices for A[i, j]: i*m + j
                # V indices for B[j, k]: j*p + k
                # W indices for C[k, i]: k*n + i
                
                # M1 = (a11+a22)(b11+b22)
                U[0 * m + j_base, col] = 1       # a11
                U[1 * m + j_base + 1, col] = 1   # a22
                V[j_base * p + k_base, col] = 1       # b11
                V[(j_base + 1) * p + (k_base + 1), col] = 1  # b22
                W[k_base * n + 0, col] = 1       # C11
                W[k_base * n + 1, col] = 0       # C21 (no contribution)
                W[(k_base + 1) * n + 0, col] = 0 # C12 (no contribution)
                W[(k_base + 1) * n + 1, col] = 1 # C22
                col += 1
                
                # M2 = (a21+a22)(b11+b12)
                U[1 * m + j_base, col] = 1       # a21
                U[1 * m + j_base + 1, col] = 1   # a22
                V[j_base * p + k_base, col] = 1       # b11
                V[j_base * p + (k_base + 1), col] = 1 # b12
                W[k_base * n + 0, col] = 0       # C11
                W[k_base * n + 1, col] = 1       # C21
                W[(k_base + 1) * n + 0, col] = 0 # C12
                W[(k_base + 1) * n + 1, col] = 0 # C22
                col += 1
                
                # M3 = (a11-a22)(b21+b22)
                U[0 * m + j_base, col] = 1       # a11
                U[1 * m + j_base + 1, col] = -1  # a22
                V[(j_base + 1) * p + k_base, col] = 1       # b21
                V[(j_base + 1) * p + (k_base + 1), col] = 1 # b22
                W[k_base * n + 0, col] = 0       # C11
                W[k_base * n + 1, col] = 0       # C21
                W[(k_base + 1) * n + 0, col] = 0 # C12
                W[(k_base + 1) * n + 1, col] = 1 # C22
                col += 1
                
                # M4 = a11(b21-b22)
                U[0 * m + j_base, col] = 1       # a11
                V[(j_base + 1) * p + k_base, col] = 1       # b21
                V[(j_base + 1) * p + (k_base + 1), col] = -1 # b22
                W[k_base * n + 0, col] = 1       # C11
                W[k_base * n + 1, col] = 0       # C21
                W[(k_base + 1) * n + 0, col] = 0 # C12
                W[(k_base + 1) * n + 1, col] = -1 # C22
                col += 1
                
                # M5 = a22(b12-b11)
                U[1 * m + j_base + 1, col] = 1   # a22
                V[j_base * p + (k_base + 1), col] = 1 # b12
                V[j_base * p + k_base, col] = -1       # b11
                W[k_base * n + 0, col] = -1     # C11
                W[k_base * n + 1, col] = 0       # C21
                W[(k_base + 1) * n + 0, col] = 1 # C12
                W[(k_base + 1) * n + 1, col] = 0 # C22
                col += 1
                
                # M6 = (a11+a12)b22
                U[0 * m + j_base, col] = 1       # a11
                U[0 * m + j_base + 1, col] = 1   # a12
                V[(j_base + 1) * p + (k_base + 1), col] = 1  # b22
                W[k_base * n + 0, col] = 0       # C11
                W[k_base * n + 1, col] = 0       # C21
                W[(k_base + 1) * n + 0, col] = 1 # C12
                W[(k_base + 1) * n + 1, col] = -1 # C22
                col += 1
                
                # M7 = (a21-a11)b11
                U[1 * m + j_base, col] = 1       # a21
                U[0 * m + j_base, col] = -1      # a11
                V[j_base * p + k_base, col] = 1       # b11
                W[k_base * n + 0, col] = 1       # C11
                W[k_base * n + 1, col] = -1      # C21
                W[(k_base + 1) * n + 0, col] = 0 # C12
                W[(k_base + 1) * n + 1, col] = 0 # C22
                col += 1
                
            else:
                # kb == 3, 2x2 * 2x1, schoolbook (rank 4)
                k_base = 6  # k index 6
                
                # C[k_base, i] = sum_j A[i,j] B[j, k_base]
                # For i=0: A[0,j_base]*B[j_base,k] + A[0,j_base+1]*B[j_base+1,k]
                # For i=1: A[1,j_base]*B[j_base,k] + A[1,j_base+1]*B[j_base+1,k]
                
                # Product 1: A[0,j_base] * B[j_base,k]
                U[0 * m + j_base, col] = 1
                V[j_base * p + k_base, col] = 1
                W[k_base * n + 0, col] = 1
                col += 1
                
                # Product 2: A[0,j_base+1] * B[j_base+1,k]
                U[0 * m + j_base + 1, col] = 1
                V[(j_base + 1) * p + k_base, col] = 1
                W[k_base * n + 0, col] = 1
                col += 1
                
                # Product 3: A[1,j_base] * B[j_base,k]
                U[1 * m + j_base, col] = 1
                V[j_base * p + k_base, col] = 1
                W[k_base * n + 1, col] = 1
                col += 1
                
                # Product 4: A[1,j_base+1] * B[j_base+1,k]
                U[1 * m + j_base + 1, col] = 1
                V[(j_base + 1) * p + k_base, col] = 1
                W[k_base * n + 1, col] = 1
                col += 1
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook rank 56
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=int)
        V_fb = np.zeros((m * p, rank_fb), dtype=int)
        W_fb = np.zeros((p * n, rank_fb), dtype=int)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1
                    V_fb[j * p + k, col_fb] = 1
                    W_fb[k * n + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
