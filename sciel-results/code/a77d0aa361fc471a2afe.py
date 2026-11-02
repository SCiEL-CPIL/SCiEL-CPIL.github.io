import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 130.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_rank32():
    """
    Build a rank-32 decomposition for 3x4 * 4x4 -> 3x4 multiplication.
    
    Strategy:
    - Rows 0 and 1 of A (size 4 each) and columns 0,1 of B (size 4x2) form a 2x4 * 4x2 block.
      Use 2x2 Strassen on each 2x2 sub-block of this 2x4 * 4x2 product.
      This gives 7 multiplications per 2x2 output block.
      There are 2 output rows x 2 output cols = 4 output 2x2 blocks? No.
      Actually, the output for rows 0,1 and cols 0,1 is a 2x2 matrix.
      We split the input A rows 0,1 (size 4) into two 2-vectors: A_top = A[0:2], A_bot = A[2:4].
      We split the input B cols 0,1 (size 4x2) into two 4x1 vectors: B_left = B[:,0], B_right = B[:,1].
      Wait, B is 4x4. The first 2 columns are 4x2.
      Let's use the standard Strassen for (2x4) * (4x2).
      A = [[A11, A12], [A21, A22]] where each is 1x2? No, A is 2x4.
      Let's view A as two rows of length 4.
      A = [a1, a2, a3, a4] (row 0)
          [b1, b2, b3, b4] (row 1)
      B = 4x2 matrix [c1, c2] (cols 0,1)
           c1 is 4x1, c2 is 4x1.
      Product C = A * B is 2x2.
      C00 = a1*c1[0] + a2*c1[1] + a3*c1[2] + a4*c1[3]
      C01 = a1*c2[0] + a2*c2[1] + a3*c2[2] + a4*c2[3]
      C10 = b1*c1[0] + b2*c1[1] + b3*c1[2] + b4*c1[3]
      C11 = b1*c2[0] + b2*c2[1] + b3*c2[2] + b4*c2[3]
      
      This is equivalent to multiplying a 2x4 matrix by a 4x2 matrix.
      We can split this into four 1x2 * 2x1 multiplications? No.
      Let's split A into two 1x2 blocks: A_top = [a1, a2], A_bot = [a3, a4].
      Let's split B into two 2x1 blocks: B_top = [c1[0], c1[1]; c2[0], c2[1]], B_bot = [c1[2], c1[3]; c2[2], c2[3]].
      Then C = A_top * B_top + A_bot * B_bot.
      A_top * B_top is 1x2 * 2x2 = 1x2.
      A_bot * B_bot is 1x2 * 2x2 = 1x2.
      So C_top_row = A_top * B_top + A_bot * B_bot.
      C_bot_row = (similarly for row 1 of A).
      
      This doesn't reduce rank directly unless we use Strassen on the 2x2 blocks.
      Actually, the 2x4 * 4x2 multiplication can be done in rank 14?
      Let's use the known result: M(2x4, 4x2) <= 14.
      Or just use 4 separate 1x4 * 4x1 multiplications? No, that's rank 4 per output element, total 16.
      But we can share.
      
      Let's stick to a simpler, provably rank-32 construction:
      Use rank 8 for the 2x4 * 4x2 block (trivial: 2 rows * 2 cols * 2? No).
      Trivial rank for 2x4 * 4x2 is 2*4*2 = 16.
      Trivial rank for 1x4 * 4x4 is 1*4*4 = 16.
      Total trivial = 16 + 16 = 32.
      This is already better than 44.
      
      Can we do better?
      M(2x4, 4x2) = 14 is known (Strassen-like).
      M(1x4, 4x4) = 4? No, M(1xn, nxp) = p if n=1?
      Wait, 1x4 * 4x4 -> 1x4.
      C_k = sum_j A_j B_jk.
      This is just extracting columns. Rank is 4.
      So total rank = 14 + 4 = 18?
      Let's verify M(1x4, 4x4) rank.
      A is 1x4, B is 4x4, C is 1x4.
      C = A B.
      This is a linear map from (A,B) to C.
      C_k = sum_j A_j B_jk.
      This is a tensor of rank 4.
      So if we use rank 14 for the top 2 rows and rank 4 for the bottom row, total rank is 18.
      
      Let's construct the rank 14 for 2x4 * 4x2.
      A is 2x4, B is 4x2, C is 2x2.
      Split A into A1 (1x2) and A2 (1x2)? No, A is 2 rows.
      Let A = [[a,b], [c,d]] where each is 1x2.
      Let B = [[e,f], [g,h]] where each is 2x1.
      Then C = A B = [[ae+cg, af+ch], [ce+dg, cf+dh]].
      This is exactly 2x2 * 2x2 matrix multiplication, but with "vectors" of size 2.
      The rank of tensor multiplication of two 2x2 matrices where entries are vectors of size 2 is the same as the rank of 2x2 matrix multiplication, which is 7 (Strassen).
      However, the "multiplication" of two vectors of size 2 (e.g., a*e where a, e are 1x2 and 2x1) is a dot product, which is rank 2.
      So the total rank is 7 * 2 = 14.
      
      So we can achieve rank 14 for the top 2 rows.
      And rank 4 for the bottom row.
      Total rank = 18.
      
      Let's implement this.
      """
    n, m, p = 3, 4, 4
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Part 1: Top 2 rows (indices 0,1) and first 2 cols of B (indices 0,1)
    # A_top: 2x4, B_top: 4x2, C_top: 2x2
    # Decompose using Strassen on 2x2 "macro" elements.
    # A_top = [[A11, A12], [A21, A22]] where Aij are 1x2 vectors.
    # B_top = [[B11, B12], [B21, B22]] where Bij are 2x1 vectors.
    # C = A B.
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    col = 0
    # A11 = A[0, 0:2], A12 = A[0, 2:4]
    # A21 = A[1, 0:2], A22 = A[1, 2:4]
    # B11 = B[0:2, 0], B12 = B[0:2, 1]
    # B21 = B[2:4, 0], B22 = B[2:4, 1]
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11 + A22: U[0,0]=1, U[0,1]=1, U[1,2]=1, U[1,3]=1
    # B11 + B22: V[0,0]=1, V[1,0]=1, V[2,1]=1, V[3,1]=1
    # C11 += M1, C22 += M1
    U[0, 0] = 1; U[0, 1] = 1; U[1, 2] = 1; U[1, 3] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 1] = 1; V[3, 1] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[3, 1] = 1; W[4, 1] = 1
    col += 1
    
    # M2 = (A21 + A22)B11
    # A21 + A22: U[1,0]=1, U[1,1]=1, U[1,2]=1, U[1,3]=1
    # B11: V[0,0]=1, V[1,0]=1
    # C21 += M2, C22 -= M2
    U[1, 0] = 1; U[1, 1] = 1; U[1, 2] = 1; U[1, 3] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[3, 1] = 1; W[4, 1] = 1
    W[3, 2] = -1; W[4, 2] = -1
    col += 1
    
    # M3 = A11(B12 - B22)
    # A11: U[0,0]=1, U[0,1]=1
    # B12 - B22: V[0,1]=1, V[1,1]=1, V[2,1]=-1, V[3,1]=-1
    # C12 += M3, C22 += M3
    U[0, 0] = 1; U[0, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1; V[2, 1] = -1; V[3, 1] = -1
    W[1, 0] = 1; W[2, 0] = 1
    W[3, 1] = 1; W[4, 1] = 1
    col += 1
    
    # M4 = A22(B21 - B11)
    # A22: U[1,2]=1, U[1,3]=1
    # B21 - B11: V[0,0]=-1, V[1,0]=-1, V[2,0]=1, V[3,0]=1
    # C11 += M4, C21 += M4
    U[1, 2] = 1; U[1, 3] = 1
    V[0, 0] = -1; V[1, 0] = -1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1
    W[3, 1] = 1; W[4, 1] = 1
    col += 1
    
    # M5 = (A11 + A12)B22
    # A11 + A12: U[0,0]=1, U[0,1]=1, U[0,2]=1, U[0,3]=1
    # B22: V[2,1]=1, V[3,1]=1
    # C11 -= M5, C12 += M5
    U[0, 0] = 1; U[0, 1] = 1; U[0, 2] = 1; U[0, 3] = 1
    V[2, 1] = 1; V[3, 1] = 1
    W[0, 0] = -1; W[1, 0] = -1
    W[1, 0] = 1; W[2, 0] = 1
    col += 1
    
    # M6 = (A21 - A11)(B11 + B12)
    # A21 - A11: U[0,0]=-1, U[0,1]=-1, U[1,0]=1, U[1,1]=1
    # B11 + B12: V[0,0]=1, V[0,1]=1, V[1,0]=1, V[1,1]=1
    # C22 += M6
    U[0, 0] = -1; U[0, 1] = -1; U[1, 0] = 1; U[1, 1] = 1
    V[0, 0] = 1; V[0, 1] = 1; V[1, 0] = 1; V[1, 1] = 1
    W[3, 1] = 1; W[4, 1] = 1
    col += 1
    
    # M7 = (A12 - A22)(B21 + B22)
    # A12 - A22: U[0,2]=1, U[0,3]=1, U[1,2]=-1, U[1,3]=-1
    # B21 + B22: V[2,0]=1, V[2,1]=1, V[3,0]=1, V[3,1]=1
    # C11 += M7
    U[0, 2] = 1; U[0, 3] = 1; U[1, 2] = -1; U[1, 3] = -1
    V[2, 0] = 1; V[2, 1] = 1; V[3, 0] = 1; V[3, 1] = 1
    W[0, 0] = 1; W[1, 0] = 1
    col += 1
    
    # Part 2: Bottom row (index 2) and all 4 cols of B
    # A_bot: 1x4, B: 4x4, C_bot: 1x4
    # C_k = sum_j A_j B_jk
    # This is rank 4.
    # r=0: A[0], B[:,0] -> C[0]
    # r=1: A[1], B[:,1] -> C[1]
    # r=2: A[2], B[:,2] -> C[2]
    # r=3: A[3], B[:,3] -> C[3]
    for k in range(4):
        U[2, k] = 1
        for j in range(4):
            V[j, k] = 1
        W[k, 2] = 1
    
    return U, V, W

def construct():
    n, m, p = 3, 4, 4
    
    U, V, W = _build_rank32()
    
    return {"U": U, "V": V, "W": W}
