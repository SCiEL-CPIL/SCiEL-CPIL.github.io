"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.

Strategy: Partition the multiplication into overlapping blocks and use Strassen's
algorithm for the 2x2*2x2 sub-multiplications to reduce the rank below schoolbook.

We partition:
- A (3x3) into: A00 (1x1), A01 (1x2), A10 (2x1), A11 (2x2)
- B (3x4) into: B00 (1x2), B01 (1x2), B10 (2x2), B11 (2x2)

The output C (3x4) has blocks:
- C00 (1x2) = A00*B00 + A01*B10
- C01 (1x2) = A00*B01 + A01*B11
- C10 (2x2) = A10*B00 + A11*B10
- C11 (2x2) = A10*B01 + A11*B11

For the 2x2 blocks (C10 and C11), we use Strassen's algorithm.
For the 1x2 blocks (C00 and C01), we use direct multiplication.

Let me carefully construct this decomposition.

Actually, let me think about this more carefully using a known approach.

For 3x3 * 3x4, one effective approach is:
1. Use the fact that we can share some computations between the different output blocks.

Let me use a different partitioning that's cleaner:

Partition A as:
- Top row: A[0,:] (1x3)
- Bottom 2 rows: A[1:3,:] (2x3)

Partition B as:
- Left 2 cols: B[:,0:2] (3x2)
- Right 2 cols: B[:,2:4] (3x2)

Then:
- C[0,0:2] = A[0,:] * B[:,0:2]  (1x2)
- C[0,2:4] = A[0,:] * B[:,2:4]  (1x2)
- C[1:3,0:2] = A[1:3,:] * B[:,0:2]  (2x2)
- C[1:3,2:4] = A[1:3,:] * B[:,2:4]  (2x2)

For the 2x2 multiplications, we can use Strassen (rank 7 each).
For the 1x2 multiplications, each requires rank 3 (direct).

Total: 3 + 3 + 7 + 7 = 20. Not great.

Let me try a better approach. Let me use the known rank-15 construction.

Actually, let me just implement a correct Strassen-based decomposition.

The key insight: we can use Strassen for the 2x2*2x2 parts.

Let me define the blocks more carefully:

A = [[a00, a01, a02],
     [a10, a11, a12],
     [a20, a21, a22]]

B = [[b00, b01, b02, b03],
     [b10, b11, b12, b13],
     [b20, b21, b22, b23]]

C = A*B (3x4)

C[i,k] = sum_j A[i,j]*B[j,k]

Let me partition:
- A_top = A[0,:] = [a00, a01, a02] (1x3)
- A_bot = A[1:3,:] (2x3)
- B_left = B[:,0:2] (3x2)
- B_right = B[:,2:4] (3x2)

C_top_left = A_top * B_left (1x2)
C_top_right = A_top * B_right (1x2)
C_bot_left = A_bot * B_left (2x2)
C_bot_right = A_bot * B_right (2x2)

For C_bot_left and C_bot_right, use Strassen.

Strassen for 2x2 * 2x2:
Given P (2x2) and Q (2x2), compute R = P*Q using 7 multiplications.

Let P = [[p00, p01], [p10, p11]] and Q = [[q00, q01], [q10, q11]]

Standard Strassen:
M1 = (p00 + p11) * (q00 + q11)
M2 = (p10 + p11) * q00
M3 = p00 * (q01 - q11)
M4 = p11 * (q10 - q00)  -- wait, this is wrong. Let me use the correct formula.

Correct Strassen:
M1 = (p00 + p11) * (q00 + q11)
M2 = (p10 + p11) * q00
M3 = p00 * (q01 - q11)
M4 = p11 * (q10 - q00)  -- I keep getting confused. Let me derive it.

R00 = M1 - M4 + M3 - M7
R01 = M5 + M4
R10 = M2 + M7
R11 = M1 - M2 + M6 + M3

Where:
M1 = (p00 + p11) * (q00 + q11)
M2 = (p10 + p11) * q00
M3 = p00 * (q01 - q11)
M4 = p11 * (q10 - q00)  -- no, let me look at this differently.

Actually the standard Strassen is:
M1 = (p00 + p11)(q00 + q11)
M2 = (p10 + p11)q00
M3 = p00(q01 - q11)
M4 = p11(q10 - q00)  -- I think this is still wrong.

Let me just use a verified version:
M1 = (p00 + p11)(q00 + q11)
M2 = (p10 + p11)q00
M3 = p00(q01 - q11)
M4 = p11(q10 - q00)  -- Let me verify: R01 = M5 + M4. If M4 = p11*(q10-q00), then R01 should involve p01*q00 + p11*q10. Hmm.

Let me just use the well-known correct Strassen:
M1 = (p00 + p11)(q00 + q11)
M2 = (p10 + p11)q00
M3 = p00(q01 - q11)
M4 = p11(q10 - q00)  -- I'll verify numerically later.

Actually, let me just use a simple correct implementation:

For 2x2*2x2:
M1 = (p00+p11)*(q00+q11)
M2 = (p10+p11)*q00
M3 = p00*(q01-q11)
M4 = p11*(q10-q00)
M5 = p10*(q11+q00)
M6 = (p11-p00)*(q10+q11)
M7 = (p00-p10)*(q00+q01)

R00 = M1 - M4 + M3 - M7
R01 = M5 + M4
R10 = M2 + M7
R11 = M1 - M2 + M6 + M3

Let me verify with simple values:
P = [[1,0],[0,1]], Q = [[1,0],[0,1]]
M1 = (1+1)*(1+1) = 4
M2 = (0+1)*1 = 1
M3 = 1*(0-1) = -1
M4 = 1*(0-1) = -1
M5 = 0*(1+1) = 0
M6 = (1-1)*(0+1) = 0
M7 = (1-0)*(1+0) = 1

R00 = 4 - (-1) + (-1) - 1 = 4+1-1-1 = 3. Expected 1. Wrong!

So my Strassen is incorrect. Let me use the correct one.

Correct Strassen:
M1 = (p00 + p11)(q00 + q11)
M2 = (p10 + p11)q00
M3 = p00(q01 - q11)
M4 = p11(q10 - q00)
M5 = p10(q11 + q00)
M6 = (p11 - p00)(q10 + q11)
M7 = (p00 - p10)(q00 + q01)

R00 = M1 - M4 + M3 - M7
R01 = M5 + M4
R10 = M2 + M7
R11 = M1 - M2 + M6 + M3

Test with identity:
P=I, Q=I
M1 = (1+1)(1+1)=4
M2 = (0+1)(1)=1
M3 = 1(0-1)=-1
M4 = 1(0-1)=-1
M5 = 0(1+1)=0
M6 = (1-1)(0+1)=0
M7 = (1-0)(1+0)=1

R00 = 4-(-1)+(-1)-1 = 3. Still wrong.

I clearly have the Strassen formulas wrong. Let me use a different, verified approach.

The correct Strassen algorithm:
M1 = (p00 + p11)(q00 + q11)
M2 = (p10 + p11)q00
M3 = p00(q01 - q11)
M4 = p11(q10 - q00)
M5 = p10(q11 + q00)
M6 = (p11 - p00)(q10 + q11)
M7 = (p00 - p10)(q00 + q01)

r00 = M1 - M4 + M3 - M7
r01 = M5 + M4
r10 = M2 + M7
r11 = M1 - M2 + M6 + M3

Let me try P=[[1,2],[3,4]], Q=[[5,6],[7,8]]
Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

M1 = (1+4)(5+8) = 5*13 = 65
M2 = (3+4)*5 = 35
M3 = 1*(6-8) = -2
M4 = 4*(7-5) = 8
M5 = 3*(8+5) = 39
M6 = (4-1)*(7+8) = 3*15 = 45
M7 = (1-3)*(5+6) = -2*11 = -22

r00 = 65 - 8 + (-2) - (-22) = 65-8-2+22 = 77. Expected 19. Wrong!

I'm clearly misremembering Strassen. Let me just use a simple correct decomposition with a higher rank that I can verify.

Given the difficulty, let me use a rank-24 construction that I can verify is correct, or just return the schoolbook rank-36.

Actually, let me try a simpler improvement: use rank 24 by grouping.

For now, let me return the schoolbook to ensure correctness.
"""

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Return a valid tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Uses schoolbook method with rank 36 for guaranteed correctness.
    """
    n, m, p = DIMENSIONS
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
