"""
Matrix multiplication 2x4x6: rank-minimal exact tensor decomposition.

Strategy: Use a block decomposition to reduce the problem to smaller
multiplications. Specifically, we exploit the structure:
  A (2x4) * B (4x6) = C (2x6)

We can split the 4-dimensional middle index into blocks.
Key insight: M_{2,4,6} can be bounded by combining smaller known results.

Known results:
- M_{2,2,2} = 7 (Strassen)
- M_{2,2,3} <= 11 (Bini et al.)
- M_{2,4,4} <= 32 (Lysyk)

Approach: Split B (4x6) into two 4x3 blocks: B = [B1 | B2]
Then C = [A*B1 | A*B2], where each A*B1 is 2x3.
So M_{2,4,6} <= 2 * M_{2,4,3}

For M_{2,4,3}: Split A (2x4) into two 2x2 blocks: A = [A1 | A2]
Split B (4x3) into two parts: B1 (2x3) and B2 (2x3)
Then A*B = A1*B1 + A2*B2
Each A1*B1 is 2x2 * 2x3 = M_{2,2,3}
So M_{2,4,3} <= 2 * M_{2,2,3}

If M_{2,2,3} <= 11, then M_{2,4,3} <= 22, and M_{2,4,6} <= 44.
But we can do better.

Alternative: Use the fact that M_{2,2,3} can be done in rank 11.
Actually, let me use a more direct approach.

Let me construct the decomposition explicitly using Strassen-like identities.

For 2x2x2, Strassen gives rank 7:
  m1 = (a11+a22)(b11+b22)
  m2 = (a21+a22)b11
  m3 = a11(b12-b22)
  m4 = a22(b21-b11)
  m5 = (a11+a12)b22
  m6 = (a21-a11)(b11+b12)
  m7 = (a12-a22)(b21+b22)

For 2x4x6, let's use a block approach:
A = [[A11, A12, A13, A14],
     [A21, A22, A23, A24]]
B = [[B11, B12, B13, B14, B15, B16],
     [B21, B22, B23, B24, B25, B26],
     [B31, B32, B33, B34, B35, B36],
     [B41, B42, B43, B44, B45, B46]]

C[i,k] = sum_j A[i,j] * B[j,k]

Let me try a different structural approach: use the embedding
M_{2,4,6} <= M_{4,4,6} by padding, but that's worse.

Actually, let me just implement a known good bound. The Strassen-based
approach from the inspiration gives rank 42. Let me try to improve on that.

Key idea: M_{2,4,6} can be split as:
- Split the 6 columns into 3+3: M_{2,4,6} <= 2 * M_{2,4,3}
- Split M_{2,4,3}: the 4 rows of B split as 2+2: M_{2,4,3} <= M_{2,2,3} + M_{2,2,3} = 2*M_{2,2,3}

So M_{2,4,6} <= 4 * M_{2,2,3}

If M_{2,2,3} = 11, then M_{2,4,6} <= 44. Still worse than 42.

Let me try: M_{2,4,6} <= M_{2,4,4} + M_{2,4,2}
M_{2,4,4} <= 32, M_{2,4,2} = 16 (schoolbook: 2*4*2=16, but can we do better?)
M_{2,4,2}: A(2x4)*B(4x2) = C(2x2). This is like M_{2,4,2}.
Actually M_{2,4,2} = M_{2,2,2} + M_{2,2,2} = 14? No...
Split A into two 2x2 blocks: A=[A1|A2], B=[B1;B2] where B1,B2 are 2x2.
C = A1*B1 + A2*B2, each is M_{2,2,2} = 7. So M_{2,4,2} <= 14.
Thus M_{2,4,6} <= 32 + 14 = 46. Worse.

Let me just implement the rank-42 Strassen-based solution and see if I can
find a better one through a different structural decomposition.

Actually, let me try: M_{2,4,6} via 3 blocks of M_{2,4,2}:
Split 6 into 2+2+2: M_{2,4,6} <= 3 * M_{2,4,2} <= 3*14 = 42. Same as before.

Let me try a more clever approach. What if I use the tensor structure directly?

I'll implement a rank-42 decomposition based on 3 Strassen 2x2x2 blocks
for the 2x4x2 subproblems, applied to each pair of columns.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for 2x4 * 4x6 matrix multiplication.
    
    Strategy: Split the 6 columns into three groups of 2.
    For each group of 2 columns, we have a 2x4 * 4x2 subproblem.
    M_{2,4,2} can be done in rank 14 using two Strassen 2x2x2 blocks.
    Total rank: 3 * 14 = 42.
    
    But we can potentially do better by sharing structure.
    Let me try a direct construction.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We'll build the decomposition using Strassen for 2x2x2 blocks.
    # Split A (2x4) into two 2x2 blocks: A = [A_L | A_R]
    # Split B (4x6) into top 2 rows and bottom 2 rows: B = [B_T; B_B]
    # For each pair of output columns, we need A * B[:, 2k:2k+2]
    
    # Let me construct it column-pair by column-pair.
    # For columns 0,1: C[:,0:2] = A * B[:,0:2]
    #   = A_L * B_T[:,0:2] + A_R * B_B[:,0:2]
    #   Each term is 2x2 * 2x2, done via Strassen (rank 7 each)
    #   Total for this pair: 14
    
    # Similarly for columns 2,3 and 4,5.
    # Total: 3 * 14 = 42
    
    # Let me build the matrices.
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (12, rank)
    
    # Strassen 2x2x2 decomposition:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For our problem, we need to map these into the global indices.
    # A indices: i in {0,1}, j in {0,1,2,3}
    #   A_L: j=0,1; A_R: j=2,3
    # B indices: j in {0,1,2,3}, k in {0,1,2,3,4,5}
    #   B_T: j=0,1; B_B: j=2,3
    # C indices: i in {0,1}, k in {0,1,2,3,4,5}
    #   W[k*n + i, c] per the trace convention
    
    # Let me define the Strassen factors for a 2x2x2 block.
    # For a block with A rows (i, i2), A cols (j, j2), B rows (j, j2), B cols (k, k2):
    # U indices: i*m + j, i*m + j2
    # V indices: j*p + k, j*p + k2, j2*p + k, j2*p + k2
    # W indices: k*n + i, k2*n + i, k*n + i2, k2*n + i2
    
    # Strassen coefficients for the 7 products:
    # m1: (A[i,j]+A[i2,j2]) * (B[j,k]+B[j2,k2])
    # m2: (A[i2,j]+A[i2,j2]) * B[j,k]
    # m3: A[i,j] * (B[j,k2]-B[j2,k2])
    # m4: A[i2,j2] * (B[j2,k]-B[j,k])
    # m5: (A[i,j]+A[i,j2]) * B[j2,k2]
    # m6: (A[i2,j]-A[i,j]) * (B[j,k]+B[j,k2])
    # m7: (A[i,j2]-A[i2,j2]) * (B[j2,k]+B[j2,k2])
    
    # Let me build the full decomposition.
    
    rank = 0
    U_rows = []  # list of (row, col, val)
    V_rows = []
    W_rows = []
    
    def add_U(row, col, val):
        U_rows.append((row, col, val))
    
    def add_V(row, col, val):
        V_rows.append((row, col, val))
    
    def add_W(row, col, val):
        W_rows.append((row, col, val))
    
    # For each column pair (k0, k0+1), do a 2x4x2 multiplication.
    # A = [A_L | A_R] where A_L uses cols 0,1 and A_R uses cols 2,3
    # B[:, k0:k0+2] = [B_T; B_B] where B_T uses rows 0,1 and B_B uses rows 2,3
    # C[:, k0:k0+2] = A_L * B_T + A_R * B_B
    
    # A_L * B_T: 2x2 * 2x2, Strassen rank 7
    # A_R * B_B: 2x2 * 2x2, Strassen rank 7
    # Total per column pair: 14
    
    # Let me define a helper for Strassen 2x2x2.
    def strassen_222(i0, j0, k0, i1, j1, k1, base_rank):
        """
        Add Strassen 2x2x2 factors for A[i0:i2, j0:j2] * B[j0:j2, k0:k2] -> C[i0:i2, k0:k2]
        where i1=i0+1, j1=j0+1, k1=k0+1 for the 2x2 case.
        """
        # A indices: (i0, j0), (i0, j1), (i1, j0), (i1, j1)
        # B indices: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
        # C indices: (i0, k0), (i0, k1), (i1, k0), (i1, k1)
        
        # m1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        r = base_rank
        add_U(i0 * m + j0, r, 1)
        add_U(i1 * m + j1, r, 1)
        add_V(j0 * p + k0, r, 1)
        add_V(j1 * p + k1, r, 1)
        # C[i0,k0] += m1, C[i1,k1] += m1
        add_W(k0 * n + i0, r, 1)
        add_W(k1 * n + i1, r, 1)
        
        # m2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        r = base_rank + 1
        add_U(i1 * m + j0, r, 1)
        add_U(i1 * m + j1, r, 1)
        add_V(j0 * p + k0, r, 1)
        # C[i1,k0] += m2
        add_W(k0 * n + i1, r, 1)
        
        # m3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        r = base_rank + 2
        add_U(i0 * m + j0, r, 1)
        add_V(j0 * p + k1, r, 1)
        add_V(j1 * p + k1, r, -1)
        # C[i0,k1] += m3, C[i1,k1] -= m3
        add_W(k1 * n + i0, r, 1)
        add_W(k1 * n + i1, r, -1)
        
        # m4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        r = base_rank + 3
        add_U(i1 * m + j1, r, 1)
        add_V(j1 * p + k0, r, 1)
        add_V(j0 * p + k0, r, -1)
        # C[i1,k0] += m4, C[i0,k0] += m4
        add_W(k0 * n + i1, r, 1)
        add_W(k0 * n + i0, r, 1)
        
        # m5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        r = base_rank + 4
        add_U(i0 * m + j0, r, 1)
        add_U(i0 * m + j1, r, 1)
        add_V(j1 * p + k1, r, 1)
        # C[i1,k1] -= m5, C[i0,k1] += m5
        add_W(k1 * n + i1, r, -1)
        add_W(k1 * n + i0, r, 1)
        
        # m6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        r = base_rank + 5
        add_U(i1 * m + j0, r, 1)
        add_U(i0 * m + j0, r, -1)
        add_V(j0 * p + k0, r, 1)
        add_V(j0 * p + k1, r, 1)
        # C[i1,k1] += m6
        add_W(k1 * n + i1, r, 1)
        
        # m7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        r = base_rank + 6
        add_U(i0 * m + j1, r, 1)
        add_U(i1 * m + j1, r, -1)
        add_V(j1 * p + k0, r, 1)
        add_V(j1 * p + k1, r, 1)
        # C[i0,k0] += m7
        add_W(k0 * n + i0, r, 1)
        
        return base_rank + 7
    
    # For each column pair (k0, k0+1):
    # A_L * B_T: A rows 0,1; A cols 0,1; B rows 0,1; B cols k0,k0+1
    # A_R * B_B: A rows 0,1; A cols 2,3; B rows 2,3; B cols k0,k0+1
    
    current_rank = 0
    for k0 in [0, 2, 4]:
        k1 = k0 + 1
        # A_L * B_T: i=0,1; j=0,1; B j=0,1; k=k0,k1
        current_rank = strassen_222(0, 0, k0, 1, 1, k1, current_rank)
        # A_R * B_B: i=0,1; j=2,3; B j=2,3; k=k0,k1
        current_rank = strassen_222(0, 2, k0, 1, 3, k1, current_rank)
    
    rank = current_rank
    
    # Build the matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for row, col, val in U_rows:
        U[row][col] = val
    for row, col, val in V_rows:
        V[row][col] = val
    for row, col, val in W_rows:
        W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
