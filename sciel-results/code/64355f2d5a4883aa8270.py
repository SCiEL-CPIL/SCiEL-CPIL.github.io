"""Program for matrix multiplication 3x3 * 3x7 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    deadline = time.time() + 140.0
    
    # We'll use a hybrid approach:
    # - Strassen for the 2x2 block (i,j in {0,1}, k in {0,1}) -> 7 factors
    # - Schoolbook for the remaining entries -> 15 factors
    # Total rank = 22
    
    # Actually, let me think about whether we can do better.
    # The tensor is of size 3x3x7 = 63 entries.
    # Strassen on 2x2x2 gives 7 instead of 8 for that block.
    # The remaining 5 output entries (those with i=2 or k=2) require 15 factors.
    # Total: 7 + 15 = 22.
    
    # Can we do better than 22? 
    # Maybe by using Strassen-like tricks on other sub-blocks?
    # For example, the entries with i=2, k in {0,1,2} form a 1x3x3 sub-problem.
    # That's already schoolbook optimal (3*3=9? No, it's 1*3*3=9 factors).
    # Wait, C[2, k] = sum_j A[2,j] B[j,k] for k=0,1,2. That's 3 outputs, each needs 3 mults = 9.
    # Similarly C[0,2] and C[1,2] each need 3 mults = 6.
    # Total for the "rest": 9 + 6 = 15. Yes.
    
    # Let's build the 22-rank decomposition.
    U = []  # Will be list of lists, shape (9, 22)
    V = []  # Shape (21, 22)
    W = []  # Shape (21, 22) -- p*n = 7*3 = 21
    
    # Strassen factors for the 2x2x2 sub-block
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Mapping: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*m + j = i*3 + j
    # V index: j*p + k = j*7 + k
    # W index: k*n + i = k*3 + i
    
    # Factor 1: M1 = (A00+A11)(B00+B11)
    # U: [0,0]=1, [1,1]=1 -> rows 0 and 4
    # V: [0,0]=1, [1,1]=1 -> rows 0 and 8
    # W: C00 += 1, C11 += 1 -> W[0*3+0]=1, W[1*3+1]=1 -> rows 0 and 4
    U.append([1,0,0,0,1,0,0,0,0])  # i=0,j=0 and i=1,j=1
    U.append([0,0,0,0,0,0,0,0,0])  # placeholder, will fill properly
    
    # Let me be more systematic. I'll build U, V, W as 2D arrays.
    rank = 22
    U = [[0]*rank for _ in range(n*m)]  # 9 x 22
    V = [[0]*rank for _ in range(m*p)]  # 21 x 22
    W = [[0]*rank for _ in range(p*n)]  # 21 x 22
    
    col = 0
    # Strassen factors
    # M1
    U[0][col] = 1  # A[0,0]
    U[4][col] = 1  # A[1,1]
    V[0][col] = 1  # B[0,0]
    V[8][col] = 1  # B[1,1]
    W[0][col] = 1  # C[0,0]
    W[4][col] = 1  # C[1,1]
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1  # A[1,0]
    U[4][col] = 1  # A[1,1]
    V[0][col] = 1  # B[0,0]
    W[3][col] = 1  # C[1,0]
    W[4][col] = -1  # C[1,1]
    col += 1
    
    # M3 = A00(B10-B11)
    U[0][col] = 1  # A[0,0]
    V[7][col] = 1  # B[1,0]
    V[8][col] = -1  # B[1,1]
    W[1][col] = 1  # C[0,1]
    W[3][col] = 1  # C[1,0]
    col += 1
    
    # M4 = A11(B01-B10)
    U[4][col] = 1  # A[1,1]
    V[1][col] = 1  # B[0,1]
    V[7][col] = -1  # B[1,0]
    W[0][col] = 1  # C[0,0]
    W[3][col] = 1  # C[1,0]
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1  # A[0,0]
    U[1][col] = 1  # A[0,1]
    V[8][col] = 1  # B[1,1]
    W[0][col] = -1  # C[0,0]
    W[1][col] = 1  # C[0,1]
    col += 1
    
    # M6 = (A10-A00)(B00+B10)
    U[3][col] = 1  # A[1,0]
    U[0][col] = -1  # A[0,0]
    V[0][col] = 1  # B[0,0]
    V[7][col] = 1  # B[1,0]
    W[4][col] = 1  # C[1,1]
    col += 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1][col] = 1  # A[0,1]
    U[4][col] = -1  # A[1,1]
    V[1][col] = 1  # B[0,1]
    V[8][col] = 1  # B[1,1]
    W[0][col] = 1  # C[0,0]
    col += 1
    
    # Now schoolbook for remaining entries
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    for j in range(3):
        U[0*3 + j][col] = 1
        V[j*7 + 2][col] = 1
        W[2*3 + 0][col] = 1
        col += 1
    
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    for j in range(3):
        U[1*3 + j][col] = 1
        V[j*7 + 2][col] = 1
        W[2*3 + 1][col] = 1
        col += 1
    
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0]
    for j in range(3):
        U[2*3 + j][col] = 1
        V[j*7 + 0][col] = 1
        W[0*3 + 2][col] = 1
        col += 1
    
    # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1] + A[2,2]B[2,1]
    for j in range(3):
        U[2*3 + j][col] = 1
        V[j*7 + 1][col] = 1
        W[1*3 + 2][col] = 1
        col += 1
    
    # C[2,2] = A[2,0]B[0,2] + A[2,1]B[1,2] + A[2,2]B[2,2]
    for j in range(3):
        U[2*3 + j][col] = 1
        V[j*7 + 2][col] = 1
        W[2*3 + 2][col] = 1
        col += 1
    
    assert col == 22, f"Expected 22 factors, got {col}"
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # The tensor T[i*m+j, j*p+k, k*n+i] should be reconstructed
        # as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Check a few random entries
        import random
        random.seed(42)
        for _ in range(1000):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            
            # Expected value: 1 if this is a valid multiplication term, else 0
            # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k)
            # since C[i,k] = sum_j A[i,j] B[j,k]
            expected = 1.0
            
            # Computed value
            u_val = U_mat[i*m + j]
            v_val = V_mat[j*p + k]
            w_val = W_mat[k*n + i]
            computed = float(np.dot(u_val, v_val) * w_val)  # Wait, this isn't right
            
            # Actually, the tensor entry is sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            computed = 0.0
            for c in range(rank):
                computed += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
            
            if abs(computed - expected) > 1e-9:
                return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
