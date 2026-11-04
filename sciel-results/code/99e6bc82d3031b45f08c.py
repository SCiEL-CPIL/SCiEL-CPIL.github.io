"""
Rank-minimal decomposition for 2x4 * 4x6 matrix multiplication.

Uses a 2x2 block Strassen decomposition on the input rows/cols and on the
output, yielding rank 42.

n=2, m=4, p=6.
U: (n*m)=8 rows, rank columns
V: (m*p)=24 rows, rank columns
W: (p*n)=12 rows, rank columns

We treat the 2x4 matrix A as blocks:
  A11: 1x4 (row 0, cols 0..3)
  A21: 1x4 (row 1, cols 0..3)

Wait, n=2 rows, m=4 cols. So A is 2x4.
Block decomposition on rows: 1x1 blocks? No, we want 2x2 blocks.
But n=2, so we can't split into two 1x1 blocks for the row dimension if we want 2x2 blocks.

Let me reconsider the block structure.
The multiplication is C = A * B where A is 2x4 and B is 4x6.
C is 2x6.

We can think of this as:
C[:, 0:2] = A * B[:, 0:2]  (2x2 output)
C[:, 2:4] = A * B[:, 2:4]  (2x2 output)
C[:, 4:6] = A * B[:, 4:6]  (2x2 output)

For each 2x2 output block, we need to compute A (2x4) times a 4x2 block of B.
The 4x2 block can be split into two 2x2 blocks:
B[:, 0:2] = [[B11, B12], [B21, B22]] where each is 2x1? No, 2x1.

Actually, let's split the 4 columns of A and 4 rows of B into pairs:
A = [A_top, A_bottom] where A_top is 2x2 (cols 0,1) and A_bottom is 2x2 (cols 2,3)
B block for output cols (k, k+1) = [B_top, B_bottom] where B_top is 2x2 (rows 0,1) and B_bottom is 2x2 (rows 2,3)

So for output block C[:, k:k+2], we compute:
C_block = A_top * B_top + A_bottom * B_bottom

Each product A_top * B_top is 2x2 * 2x2 = 2x2, which can be done with Strassen rank 7.
Same for A_bottom * B_bottom.

So for each of the 3 output blocks (cols 0:2, 2:4, 4:6), we need 7 + 7 = 14 multiplications.
Total rank = 3 * 14 = 42.

Let me verify this.

For output block 0 (cols 0,1):
C[:, 0] = A[:, 0:2] * B[0:2, 0] + A[:, 2:4] * B[2:4, 0]
C[:, 1] = A[:, 0:2] * B[0:2, 1] + A[:, 2:4] * B[2:4, 1]

So C[:, 0:2] = A_left * B_left + A_right * B_right
where A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
B_left = B[0:2, 0:2] (2x2), B_right = B[2:4, 0:2] (2x2)

Each 2x2 * 2x2 product uses Strassen with 7 scalar multiplications.
So for each output block, we use 14 multiplications.
3 output blocks -> 42 total.

Let me construct the matrices.

For one Strassen 2x2*2x2:
Input: X (2x2), Y (2x2), Output: Z = X*Y (2x2)
Strassen:
M1 = (X00+X11)(Y00+Y11)
M2 = (X10+X11)Y00
M3 = X00(Y01-Y11)
M4 = X11(Y10-Y00)
M5 = (X00+X01)Y11
M6 = (X10-X00)(Y00+Y01)
M7 = (X01-X11)(Y10+Y11)

Z00 = M1 - M4 - M6 + M7
Z01 = M3 + M5
Z10 = M2 + M4
Z11 = M1 - M3 + M2 - M5

For our problem:
n=2, m=4, p=6
U shape: (8, rank)
V shape: (24, rank)
W shape: (12, rank)

U row index: i*m + j = i*4 + j, where i in [0,1], j in [0,3]
V row index: j*p + k = j*6 + k, where j in [0,3], k in [0,5]
W row index: k*n + i = k*2 + i, where k in [0,5], i in [0,1]

For each output block b in [0, 1, 2]:
  Output columns: 2*b, 2*b+1
  B rows for left: 0, 1 (for b=0,1,2? No, B rows are fixed)
  
Wait, B is 4x6. The rows of B are 0,1,2,3.
For output block b (cols 2b, 2b+1):
  B_left = B[0:2, 2b:2b+2] (rows 0,1; cols 2b, 2b+1)
  B_right = B[2:4, 2b:2b+2] (rows 2,3; cols 2b, 2b+1)

A_left = A[0:2, 0:2] (rows 0,1; cols 0,1)
A_right = A[0:2, 2:4] (rows 0,1; cols 2,3)

So for each block b, we compute:
  P_left = A_left * B_left  (2x2)
  P_right = A_right * B_right  (2x2)
  C_block = P_left + P_right

For P_left = A_left * B_left:
  A_left[0,0] = A[0,0] = U row 0
  A_left[0,1] = A[0,1] = U row 1
  A_left[1,0] = A[1,0] = U row 4
  A_left[1,1] = A[1,1] = U row 5
  
  B_left[0,0] = B[0,2b] = V row 0*6+2b = V row 2b
  B_left[0,1] = B[0,2b+1] = V row 2b+1
  B_left[1,0] = B[1,2b] = V row 6+2b
  B_left[1,1] = B[1,2b+1] = V row 6+2b+1

For P_right = A_right * B_right:
  A_right[0,0] = A[0,2] = U row 2
  A_right[0,1] = A[0,3] = U row 3
  A_right[1,0] = A[1,2] = U row 6
  A_right[1,1] = A[1,3] = U row 7
  
  B_right[0,0] = B[2,2b] = V row 12+2b
  B_right[0,1] = B[2,2b+1] = V row 12+2b+1
  B_right[1,0] = B[3,2b] = V row 18+2b
  B_right[1,1] = B[3,2b+1] = V row 18+2b+1

Output C_block:
  C[0,2b] = P_left[0,0] + P_right[0,0] = W row (2b)*2 + 0 = W row 4b
  C[0,2b+1] = P_left[0,1] + P_right[0,1] = W row (2b+1)*2 + 0 = W row 4b+2
  C[1,2b] = P_left[1,0] + P_right[1,0] = W row (2b)*2 + 1 = W row 4b+1
  C[1,2b+1] = P_left[1,1] + P_right[1,1] = W row (2b+1)*2 + 1 = W row 4b+3

Now, for each block b, we have 14 Strassen multiplications.
Let me label them:
  For P_left: M1_L, M2_L, M3_L, M4_L, M5_L, M6_L, M7_L
  For P_right: M1_R, M2_R, M3_R, M4_R, M5_R, M6_R, M7_R

U contributions:
  M1_L: (X00+X11) -> U[0] + U[4] -> U[0,col]=1, U[4,col]=1
  M2_L: (X10+X11) -> U[4] + U[5] -> U[4,col]=1, U[5,col]=1
  M3_L: X00 -> U[0,col]=1
  M4_L: X11 -> U[5,col]=1
  M5_L: (X00+X01) -> U[0] + U[1] -> U[0,col]=1, U[1,col]=1
  M6_L: (X10-X00) -> U[4] - U[0] -> U[4,col]=1, U[0,col]=-1
  M7_L: (X01-X11) -> U[1] - U[5] -> U[1,col]=1, U[5,col]=-1

  M1_R: (X00+X11) -> U[2] + U[6] -> U[2,col]=1, U[6,col]=1
  M2_R: (X10+X11) -> U[6] + U[7] -> U[6,col]=1, U[7,col]=1
  M3_R: X00 -> U[2,col]=1
  M4_R: X11 -> U[7,col]=1
  M5_R: (X00+X01) -> U[2] + U[3] -> U[2,col]=1, U[3,col]=1
  M6_R: (X10-X00) -> U[6] - U[2] -> U[6,col]=1, U[2,col]=-1
  M7_R: (X01-X11) -> U[3] - U[7] -> U[3,col]=1, U[7,col]=-1

V contributions:
  M1_L: (Y00+Y11) -> V[2b] + V[6+2b+1] -> V[2b,col]=1, V[7+2b,col]=1
  M2_L: Y00 -> V[2b,col]=1
  M3_L: (Y01-Y11) -> V[2b+1] - V[7+2b] -> V[2b+1,col]=1, V[7+2b,col]=-1
  M4_L: (Y10-Y00) -> V[6+2b] - V[2b] -> V[6+2b,col]=1, V[2b,col]=-1
  M5_L: Y11 -> V[7+2b,col]=1
  M6_L: (Y00+Y01) -> V[2b] + V[2b+1] -> V[2b,col]=1, V[2b+1,col]=1
  M7_L: (Y10+Y11) -> V[6+2b] + V[7+2b] -> V[6+2b,col]=1, V[7+2b,col]=1

  M1_R: (Y00+Y11) -> V[12+2b] + V[18+2b+1] -> V[12+2b,col]=1, V[19+2b,col]=1
  M2_R: Y00 -> V[12+2b,col]=1
  M3_R: (Y01-Y11) -> V[12+2b+1] - V[19+2b] -> V[12+2b+1,col]=1, V[19+2b,col]=-1
  M4_R: (Y10-Y00) -> V[18+2b] - V[12+2b] -> V[18+2b,col]=1, V[12+2b,col]=-1
  M5_R: Y11 -> V[19+2b,col]=1
  M6_R: (Y00+Y01) -> V[12+2b] + V[12+2b+1] -> V[12+2b,col]=1, V[12+2b+1,col]=1
  M7_R: (Y10+Y11) -> V[18+2b] + V[19+2b] -> V[18+2b,col]=1, V[19+2b,col]=1

W contributions:
  C[0,2b] = P_left[0,0] + P_right[0,0]
    = (M1_L - M4_L - M6_L + M7_L) + (M1_R - M4_R - M6_R + M7_R)
    W[4b, col_M1_L] = 1
    W[4b, col_M4_L] = -1
    W[4b, col_M6_L] = -1
    W[4b, col_M7_L] = 1
    W[4b, col_M1_R] = 1
    W[4b, col_M4_R] = -1
    W[4b, col_M6_R] = -1
    W[4b, col_M7_R] = 1

  C[0,2b+1] = P_left[0,1] + P_right[0,1]
    = (M3_L + M5_L) + (M3_R + M5_R)
    W[4b+2, col_M3_L] = 1
    W[4b+2, col_M5_L] = 1
    W[4b+2, col_M3_R] = 1
    W[4b+2, col_M5_R] = 1

  C[1,2b] = P_left[1,0] + P_right[1,0]
    = (M2_L + M4_L) + (M2_R + M4_R)
    W[4b+1, col_M2_L] = 1
    W[4b+1, col_M4_L] = 1
    W[4b+1, col_M2_R] = 1
    W[4b+1, col_M4_R] = 1

  C[1,2b+1] = P_left[1,1] + P_right[1,1]
    = (M1_L - M3_L + M2_L - M5_L) + (M1_R - M3_R + M2_R - M5_R)
    W[4b+3, col_M1_L] = 1
    W[4b+3, col_M3_L] = -1
    W[4b+3, col_M2_L] = 1
    W[4b+3, col_M5_L] = -1
    W[4b+3, col_M1_R] = 1
    W[4b+3, col_M3_R] = -1
    W[4b+3, col_M2_R] = 1
    W[4b+3, col_M5_R] = -1
"""

import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    col = 0
    for b in range(3):  # 3 output blocks
        # P_left = A_left * B_left
        # A_left rows: 0,1,4,5 (U indices)
        # B_left rows: 2b, 2b+1, 6+2b, 7+2b (V indices)
        
        # M1_L: U[0]+U[4], V[2b]+V[7+2b]
        U[0][col] += 1
        U[4][col] += 1
        V[2*b][col] += 1
        V[7+2*b][col] += 1
        # W: C[0,2b] += M1_L, C[1,2b+1] += M1_L
        W[4*b][col] += 1
        W[4*b+3][col] += 1
        col += 1

        # M2_L: U[4]+U[5], V[2b]
        U[4][col] += 1
        U[5][col] += 1
        V[2*b][col] += 1
        # W: C[1,2b] += M2_L, C[1,2b+1] += M2_L
        W[4*b+1][col] += 1
        W[4*b+3][col] += 1
        col += 1

        # M3_L: U[0], V[2b+1]-V[7+2b]
        U[0][col] += 1
        V[2*b+1][col] += 1
        V[7+2*b][col] -= 1
        # W: C[0,2b+1] += M3_L, C[1,2b+1] -= M3_L
        W[4*b+2][col] += 1
        W[4*b+3][col] -= 1
        col += 1

        # M4_L: U[5], V[6+2b]-V[2b]
        U[5][col] += 1
        V[6+2*b][col] += 1
        V[2*b][col] -= 1
        # W: C[0,2b] -= M4_L, C[1,2b] += M4_L
        W[4*b][col] -= 1
        W[4*b+1][col] += 1
        col += 1

        # M5_L: U[0]+U[1], V[7+2b]
        U[0][col] += 1
        U[1][col] += 1
        V[7+2*b][col] += 1
        # W: C[0,2b+1] += M5_L, C[1,2b+1] -= M5_L
        W[4*b+2][col] += 1
        W[4*b+3][col] -= 1
        col += 1

        # M6_L: U[4]-U[0], V[2b]+V[2b+1]
        U[4][col] += 1
        U[0][col] -= 1
        V[2*b][col] += 1
        V[2*b+1][col] += 1
        # W: C[0,2b] -= M6_L
        W[4*b][col] -= 1
        col += 1

        # M7_L: U[1]-U[5], V[6+2b]+V[7+2b]
        U[1][col] += 1
        U[5][col] -= 1
        V[6+2*b][col] += 1
        V[7+2*b][col] += 1
        # W: C[0,2b] += M7_L
        W[4*b][col] += 1
        col += 1

        # P_right = A_right * B_right
        # A_right rows: 2,3,6,7 (U indices)
        # B_right rows: 12+2b, 13+2b, 18+2b, 19+2b (V indices)

        # M1_R: U[2]+U[6], V[12+2b]+V[19+2b]
        U[2][col] += 1
        U[6][col] += 1
        V[12+2*b][col] += 1
        V[19+2*b][col] += 1
        # W: C[0,2b] += M1_R, C[1,2b+1] += M1_R
        W[4*b][col] += 1
        W[4*b+3][col] += 1
        col += 1

        # M2_R: U[6]+U[7], V[12+2b]
        U[6][col] += 1
        U[7][col] += 1
        V[12+2*b][col] += 1
        # W: C[1,2b] += M2_R, C[1,2b+1] += M2_R
        W[4*b+1][col] += 1
        W[4*b+3][col] += 1
        col += 1

        # M3_R: U[2], V[13+2b]-V[19+2b]
        U[2][col] += 1
        V[13+2*b][col] += 1
        V[19+2*b][col] -= 1
        # W: C[0,2b+1] += M3_R, C[1,2b+1] -= M3_R
        W[4*b+2][col] += 1
        W[4*b+3][col] -= 1
        col += 1

        # M4_R: U[7], V[18+2b]-V[12+2b]
        U[7][col] += 1
        V[18+2*b][col] += 1
        V[12+2*b][col] -= 1
        # W: C[0,2b] -= M4_R, C[1,2b] += M4_R
        W[4*b][col] -= 1
        W[4*b+1][col] += 1
        col += 1

        # M5_R: U[2]+U[3], V[19+2b]
        U[2][col] += 1
        U[3][col] += 1
        V[19+2*b][col] += 1
        # W: C[0,2b+1] += M5_R, C[1,2b+1] -= M5_R
        W[4*b+2][col] += 1
        W[4*b+3][col] -= 1
        col += 1

        # M6_R: U[6]-U[2], V[12+2b]+V[13+2b]
        U[6][col] += 1
        U[2][col] -= 1
        V[12+2*b][col] += 1
        V[13+2*b][col] += 1
        # W: C[0,2b] -= M6_R
        W[4*b][col] -= 1
        col += 1

        # M7_R: U[3]-U[7], V[18+2b]+V[19+2b]
        U[3][col] += 1
        U[7][col] -= 1
        V[18+2*b][col] += 1
        V[19+2*b][col] += 1
        # W: C[0,2b] += M7_R
        W[4*b][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
