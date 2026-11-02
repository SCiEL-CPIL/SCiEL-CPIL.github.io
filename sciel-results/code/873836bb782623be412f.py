"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 10 matrix.
    
    Strategy:
    1. Split the middle dimension m=3 into m1=2 and m2=1.
    2. The problem decomposes into:
       - Part 1: (2x2) * (2x10) -> (2x10)
       - Part 2: (2x1) * (1x10) -> (2x10)
    3. Part 2 is a simple outer product sum, requiring 2*1*10 = 20 multiplications.
    4. Part 1 can be optimized using Strassen's algorithm for the 2x2 matrix multiplication logic.
       For each of the 10 columns of the output, we perform a 2x2 * 2x1 multiplication.
       Standard rank for 2x2 * 2x1 is 4.
       Using Strassen-like reductions (or simply noting that M_{2,2,2} has rank 7, 
       but here we have specific structure), we can reduce the multiplications.
       
       Actually, let's look at the 2x2 * 2x10 case more closely.
       C = A * B, where A is 2x2, B is 2x10.
       This is equivalent to 10 independent 2x2 * 2x1 multiplications?
       No, the A is shared.
       C[:, k] = A * B[:, k].
       
       Let A = [[a, b], [c, d]] and B_k = [x, y]^T.
       C_0 = a x + b y
       C_1 = c x + d y
       
       This is a bilinear map R^4 x R^2 -> R^2.
       The tensor rank of this specific map is 4?
       Let's check if we can do it in 3.
       If we could, the total rank would be 3*10 + 20 = 50.
       
       However, the inspiration claims 55.
       55 = 35 + 20.
       This implies the 2x2 * 2x10 part uses 35 multiplications.
       35 / 10 = 3.5? That's not an integer.
       
       Wait, maybe the sharing is across the 10 columns?
       If we use Strassen for the 2x2 block, we compute 7 products of 2x2 matrices.
       But B is 2x10.
       We can embed the 2x10 B into a larger structure?
       
       Let's look at the known result for M_{2,3,10}.
       Schoolbook: 60.
       If we split m=3 into 2+1:
       M_{2,2,10} + M_{2,1,10}.
       Rank(M_{2,1,10}) = 20.
       Rank(M_{2,2,10})?
       
       If we treat M_{2,2,10} as 10 instances of M_{2,2,1}, and they are independent, rank is 10*4=40.
       Total 60.
       
       How to get 55?
       Maybe M_{2,2,10} has rank 35?
       This would require an algorithm that computes 2x2 * 2x10 with 35 multiplications.
       
       Let's try to construct a rank 35 decomposition for 2x2 * 2x10.
       Or maybe the split is different?
       
       Actually, there is a known result that Rank(M_{2,3,10}) <= 55.
       The decomposition uses the fact that M_{2,2,2} has rank 7.
       
       If we group the 10 columns of B into 5 pairs of 2 columns?
       B is 2x10. Split B into 5 blocks of 2x2.
       B = [B_1, B_2, B_3, B_4, B_5] where each B_i is 2x2.
       C = A * B = [A B_1, A B_2, ..., A B_5].
       
       For each block i, we compute A * B_i (2x2 * 2x2).
       Using Strassen, each 2x2 * 2x2 takes 7 multiplications.
       Total for 5 blocks: 5 * 7 = 35 multiplications.
       
       Then we add the remaining part:
       A is 2x3. Split A into A_1 (2x2) and A_2 (2x1).
       B is 3x10. Split B into B_1 (2x10) and B_2 (1x10).
       
       C = A_1 B_1 + A_2 B_2.
       
       We computed A_1 B_1 using the block method above.
       A_1 is 2x2. B_1 is 2x10.
       We split B_1 into 5 blocks of 2x2.
       Each block multiplication A_1 * B_{1,i} (2x2 * 2x2) uses 7 mults.
       Total 35 mults for A_1 B_1.
       
       Now for A_2 B_2.
       A_2 is 2x1. B_2 is 1x10.
       This is a rank 20 operation (outer product).
       Total rank = 35 + 20 = 55.
       
       This matches the inspiration!
       
       So the strategy is:
       1. Split the 3x10 matrix B into B_top (2x10) and B_bottom (1x10).
       2. Split the 2x3 matrix A into A_left (2x2) and A_right (2x1).
       3. Compute C = A_left * B_top + A_right * B_bottom.
       
       For A_right * B_bottom:
       This is (2x1) * (1x10).
       Rank = 20.
       Decomposition:
       For i in {0,1} (rows of A_right/cols of C):
       For k in {0..9} (cols of B_bottom/cols of C):
       Term: A_right[i] * B_bottom[k] -> C[i,k].
       
       For A_left * B_top:
       A_left is 2x2. B_top is 2x10.
       Split B_top into 5 blocks of 2x2: B_top[0:2, 0:2], B_top[0:2, 2:4], etc.
       Let B_top_j be the j-th 2x2 block (j=0..4).
       We need to compute A_left * B_top_j for each j.
       This is 2x2 * 2x2 multiplication.
       Use Strassen's algorithm for each block.
       
       Strassen for 2x2 * 2x2:
       Inputs: X (2x2), Y (2x2).
       M1 = (X00 + X11) * (Y00 + Y11)
       M2 = (X10 + X11) * Y00
       M3 = X00 * (Y01 - Y11)
       M4 = X11 * (Y10 - Y00)
       M5 = (X00 + X01) * Y11
       M6 = (X10 - X00) * (Y00 + Y01)
       M7 = (X01 - X11) * (Y10 + Y11)
       
       Output Z:
       Z00 = M1 + M4 - M5 + M7
       Z01 = M3 + M5
       Z10 = M2 + M4
       Z11 = M1 - M2 + M3 + M6
       
       Each M is a product of two linear forms in X and Y.
       Since we have 5 independent blocks, we perform this 5 times.
       Total multiplications = 5 * 7 = 35.
       
       Total Rank = 35 + 20 = 55.
       
       Implementation details:
       We need to construct U, V, W matrices.
       U is (n*m, rank) = (6, 55).
       V is (m*p, rank) = (30, 55).
       W is (p*n, rank) = (20, 55).
       
       Indices:
       A is flattened row-major: A[i, j] -> index i*m + j.
       n=2, m=3.
       A[0,0]->0, A[0,1]->1, A[0,2]->2
       A[1,0]->3, A[1,1]->4, A[1,2]->5
       
       B is flattened row-major: B[j, k] -> index j*p + k.
       m=3, p=10.
       B[0,0]->0 ... B[0,9]->9
       B[1,0]->10 ... B[1,9]->19
       B[2,0]->20 ... B[2,9]->29
       
       C is flattened with convention W[k*n+i].
       C[i, k] -> index k*n + i.
       p=10, n=2.
       C[0,0]->0, C[1,0]->1
       C[0,1]->2, C[1,1]->3
       ...
       C[0,k] -> 2k, C[1,k] -> 2k+1.
       
       We will build the decomposition term by term.
       
       Part 1: A_right * B_bottom
       A_right corresponds to A[:, 2]. Indices 2 and 5 in U.
       B_bottom corresponds to B[2, :]. Indices 20..29 in V.
       C[:, k] gets contribution.
       
       For i in 0..1 (row of A_right):
         a_idx = 2 + i  # A[0,2]=2, A[1,2]=5
         For k in 0..9 (col of B_bottom):
           b_idx = 20 + k # B[2,0]=20 ... B[2,9]=29
           c_idx = k * 2 + i # C[i,k]
           
           U[a_idx][col] = 1
           V[b_idx][col] = 1
           W[c_idx][col] = 1
           col += 1
           
       Part 2: A_left * B_top
       A_left corresponds to A[:, 0:2]. Indices 0,1 and 3,4.
       B_top corresponds to B[0:2, :]. Indices 0..19.
       
       We split B_top into 5 blocks of 2 columns.
       Block j (j=0..4) covers columns 2j and 2j+1 of B_top.
       B_top[0, 2j] -> V index 0 + 2j
       B_top[0, 2j+1] -> V index 0 + 2j + 1
       B_top[1, 2j] -> V index 10 + 2j
       B_top[1, 2j+1] -> V index 10 + 2j + 1
       
       For each block j, we apply Strassen.
       The "matrix" Y is the 2x2 block:
       Y00 = B[0, 2j]
       Y01 = B[0, 2j+1]
       Y10 = B[1, 2j]
       Y11 = B[1, 2j+1]
       
       The "matrix" X is A_left:
       X00 = A[0,0] (U idx 0)
       X01 = A[0,1] (U idx 1)
       X10 = A[1,0] (U idx 3)
       X11 = A[1,1] (U idx 4)
       
       The output Z is 2x2, corresponding to C[:, 2j] and C[:, 2j+1].
       Z00 -> C[0, 2j] -> W idx 2j*2 + 0 = 4j
       Z01 -> C[0, 2j+1] -> W idx (2j+1)*2 + 0 = 4j+2
       Z10 -> C[1, 2j] -> W idx 2j*2 + 1 = 4j+1
       Z11 -> C[1, 2j+1] -> W idx (2j+1)*2 + 1 = 4j+3
       
       Strassen Steps for Block j:
       
       1. M1 = (X00 + X11) * (Y00 + Y11)
          U: X00(0)=1, X11(4)=1
          V: Y00(2j)=1, Y11(10+2j+1)=1
          W: 
             Z00(4j) += 1
             Z11(4j+3) += 1
             Z01(4j+2) -= 1  (Wait, Z00 = M1 + M4 - M5 + M7, Z11 = M1 - M2 + M3 + M6)
             Let's map the linear combinations to W entries.
             
          The contribution of M1 to the output is:
          Z00 += M1
          Z11 += M1
          
          So for this term:
          U[0] += 1, U[4] += 1
          V[2j] += 1, V[10+2j+1] += 1
          W[4j] += 1
          W[4j+3] += 1
          
       2. M2 = (X10 + X11) * Y00
          U: X10(3)=1, X11(4)=1
          V: Y00(2j)=1
          W:
             Z10 += M2
             Z11 -= M2
          So:
          U[3] += 1, U[4] += 1
          V[2j] += 1
          W[4j+1] += 1
          W[4j+3] -= 1
          
       3. M3 = X00 * (Y01 - Y11)
          U: X00(0)=1
          V: Y01(2j+1)=1, Y11(10+2j+1)=-1
          W:
             Z01 += M3
             Z11 += M3
          So:
          U[0] += 1
          V[2j+1] += 1, V[10+2j+1] -= 1
          W[4j+2] += 1
          W[4j+3] += 1
          
       4. M4 = X11 * (Y10 - Y00)
          U: X11(4)=1
          V: Y10(10+2j)=1, Y00(2j)=-1
          W:
             Z00 += M4
             Z10 += M4
          So:
          U[4] += 1
          V[10+2j] += 1, V[2j] -= 1
          W[4j] += 1
          W[4j+1] += 1
          
       5. M5 = (X00 + X01) * Y11
          U: X00(0)=1, X01(1)=1
          V: Y11(10+2j+1)=1
          W:
             Z00 -= M5
             Z01 += M5
          So:
          U[0] += 1, U[1] += 1
          V[10+2j+1] += 1
          W[4j] -= 1
          W[4j+2] += 1
          
       6. M6 = (X10 - X00) * (Y00 + Y01)
          U: X10(3)=1, X00(0)=-1
          V: Y00(2j)=1, Y01(2j+1)=1
          W:
             Z11 += M6
          So:
          U[3] += 1, U[0] -= 1
          V[2j] += 1, V[2j+1] += 1
          W[4j+3] += 1
          
       7. M7 = (X01 - X11) * (Y10 + Y11)
          U: X01(1)=1, X11(4)=-1
          V: Y10(10+2j)=1, Y11(10+2j+1)=1
          W:
             Z00 += M7
          So:
          U[1] += 1, U[4] -= 1
          V[10+2j] += 1, V[10+2j+1] += 1
          W[4j] += 1

    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A_right (2x1) * B_bottom (1x10)
    # A_right is A[:, 2]. Indices in U: 2, 5.
    # B_bottom is B[2, :]. Indices in V: 20..29.
    # C[:, k] gets A[i, 2] * B[2, k].
    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i
            
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
            
    # Part 2: A_left (2x2) * B_top (2x10) using Strassen blocks
    # A_left is A[:, 0:2]. Indices in U: 0,1 (row 0), 3,4 (row 1).
    # B_top is B[0:2, :]. Indices in V: 0..9 (row 0), 10..19 (row 1).
    
    # Strassen indices for X (A_left)
    X00, X01, X10, X11 = 0, 1, 3, 4
    
    for j in range(5):
        # B block columns: 2j, 2j+1
        # Y00 = B[0, 2j] -> V[2j]
        # Y01 = B[0, 2j+1] -> V[2j+1]
        # Y10 = B[1, 2j] -> V[10+2j]
        # Y11 = B[1, 2j+1] -> V[10+2j+1]
        
        Y00 = 2 * j
        Y01 = 2 * j + 1
        Y10 = 10 + 2 * j
        Y11 = 10 + 2 * j + 1
        
        # C block columns: 2j, 2j+1
        # Z00 = C[0, 2j] -> W[4j]
        # Z01 = C[0, 2j+1] -> W[4j+2]
        # Z10 = C[1, 2j] -> W[4j+1]
        # Z11 = C[1, 2j+1] -> W[4j+3]
        
        Z00 = 4 * j
        Z01 = 4 * j + 2
        Z10 = 4 * j + 1
        Z11 = 4 * j + 3
        
        # M1 = (X00 + X11) * (Y00 + Y11)
        U[X00, col] += 1
        U[X11, col] += 1
        V[Y00, col] += 1
        V[Y11, col] += 1
        W[Z00, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M2 = (X10 + X11) * Y00
        U[X10, col] += 1
        U[X11, col] += 1
        V[Y00, col] += 1
        W[Z10, col] += 1
        W[Z11, col] -= 1
        col += 1
        
        # M3 = X00 * (Y01 - Y11)
        U[X00, col] += 1
        V[Y01, col] += 1
        V[Y11, col] -= 1
        W[Z01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M4 = X11 * (Y10 - Y00)
        U[X11, col] += 1
        V[Y10, col] += 1
        V[Y00, col] -= 1
        W[Z00, col] += 1
        W[Z10, col] += 1
        col += 1
        
        # M5 = (X00 + X01) * Y11
        U[X00, col] += 1
        U[X01, col] += 1
        V[Y11, col] += 1
        W[Z00, col] -= 1
        W[Z01, col] += 1
        col += 1
        
        # M6 = (X10 - X00) * (Y00 + Y01)
        U[X10, col] += 1
        U[X00, col] -= 1
        V[Y00, col] += 1
        V[Y01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M7 = (X01 - X11) * (Y10 + Y11)
        U[X01, col] += 1
        U[X11, col] -= 1
        V[Y10, col] += 1
        V[Y11, col] += 1
        W[Z00, col] += 1
        col += 1
        
    # Verify col == 55
    if col != 55:
        # Fallback to schoolbook if logic error
        rank = 60
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
