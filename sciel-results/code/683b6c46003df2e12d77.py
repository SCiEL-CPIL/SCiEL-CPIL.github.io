"""
Rank-minimal exact tensor decomposition for 2x2 by 2x6 matrix multiplication.

Strategy: Use Strassen-style bilinear decomposition for the 2x2 part,
then extend to the 2x6 case. We exploit the structure that multiplying
a 2x2 matrix A by a 2x6 matrix B gives a 2x6 result C where:
  C[i,k] = sum_j A[i,j] * B[j,k]

We can use 7 multiplications for the 2x2 core (Strassen) and handle
the additional columns efficiently.

Actually, let me think more carefully. The tensor is T_{i,j,k} = delta_{i,k} for
the standard multiplication. For 2x2 * 2x6 -> 2x6:
  C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]  for i in {0,1}, k in {0,...,5}

The tensor has 2*2*6 = 24 entries. We need to find rank r such that
T = sum_{c=1}^{r} U_c ⊗ V_c ⊗ W_c where U_c is in R^{2*2}, V_c in R^{2*6}, W_c in R^{6*2}.

Wait, let me re-read the interface:
- U: shape (n*m, rank) = (4, rank)
- V: shape (m*p, rank) = (12, rank)  
- W: shape (p*n, rank) = (12, rank)

The decomposition is: T[i*m+j, j*p+k, k*n+i] ... no wait.

The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
T[i,j,k] = 1 if the output index is (i,k) and inputs are (i,j) and (j,k).

In the flattened form:
- U has rows indexed by (i,j) in range n*m = 4
- V has rows indexed by (j,k) in range m*p = 12
- W has rows indexed by (k,i) in range p*n = 12

The tensor entry T[(i,j), (j,k), (k,i)] = 1.

For the schoolbook, rank = n*m*p = 24.

Let me try to find a lower rank decomposition. 

For 2x2 * 2x2, the minimum rank is 7 (Strassen).
For 2x2 * 2x6, we can think of it as three separate 2x2 * 2x2 multiplications (for columns 0-1, 2-3, 4-5) but sharing the same A.

Actually, C = A * B where A is 2x2 and B is 2x6. We can write B = [B1, B2, B3] where each Bj is 2x2.
Then C = [A*B1, A*B2, A*B3].

If we use Strassen for each, that's 3*7 = 21 multiplications. But we can do better by sharing.

Let me use the Strassen formulas for 2x2:
  M1 = (A00+A11)(B00+B11)
  M2 = (A10+A11)B00
  M3 = A00(B10-B11)
  M4 = A11(B01-B00)
  M5 = (A00+A01)B11
  M6 = (A10-A00)(B00+B10)
  M7 = (A01-A11)(B01+B11)

Then:
  C00 = M1 + M4 - M5 + M7
  C01 = M3 + M5
  C10 = M2 + M4
  C11 = M1 - M2 + M3 + M6

For our case, we have three 2x2 blocks of B. Let me denote B columns as B[:,0:2], B[:,2:4], B[:,4:6].

Using Strassen for each block independently gives rank 21.

Can we do better? Let me try to find if rank 20 is possible.

Actually, let me just implement the rank-21 Strassen-based decomposition first, then try to optimize.

For the 2x2 * 2x6 case with Strassen on each 2x2 block:

Block 1 (columns 0,1): B1 = [[B00, B01], [B10, B11]]
Block 2 (columns 2,3): B2 = [[B20, B21], [B30, B31]]
Block 3 (columns 4,5): B3 = [[B40, B41], [B50, B51]]

Strassen for block q (q=0,1,2):
  M1_q = (A00+A11)(B_{q,0,0}+B_{q,1,1})
  M2_q = (A10+A11)B_{q,0,0}
  M3_q = A00(B_{q,1,0}-B_{q,1,1})
  M4_q = A11(B_{q,0,1}-B_{q,0,0})
  M5_q = (A00+A01)B_{q,1,1}
  M6_q = (A10-A00)(B_{q,0,0}+B_{q,1,0})
  M7_q = (A01-A11)(B_{q,0,1}+B_{q,1,1})

Where B_{q,a,b} refers to element (a,b) of the q-th 2x2 block.

This gives rank 21. Let me construct the U, V, W matrices.

U has 4 rows (indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1))
V has 12 rows (indices: (j,k) for j in {0,1}, k in {0..5})
W has 12 rows (indices: (k,i) for k in {0..5}, i in {0,1})

For each Strassen multiplication M_c, we have:
U row entries, V row entries, W row entries.

Let me code this up.
"""

import numpy as np
import time

def construct():
    n, m, p = 2, 2, 6
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen indices for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U rows: 0=(A00), 1=(A01), 2=(A10), 3=(A11)
    # V rows: (j,k) where j in {0,1}, k in {0..5}
    #   V[0]=B00, V[1]=B01, V[2]=B02, V[3]=B03, V[4]=B04, V[5]=B05
    #   V[6]=B10, V[7]=B11, V[8]=B12, V[9]=B13, V[10]=B14, V[11]=B15
    # W rows: (k,i) where k in {0..5}, i in {0,1}
    #   W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11, W[4]=C02, W[5]=C12,
    #   W[6]=C03, W[7]=C13, W[8]=C04, W[9]=C14, W[10]=C05, W[11]=C15
    
    col = 0
    for block in range(3):
        # B block columns: 2*block, 2*block+1
        b00 = 2 * block      # V index for B[j=0, k=2*block]
        b01 = 2 * block + 1  # V index for B[j=0, k=2*block+1]
        b10 = 6 + 2 * block      # V index for B[j=1, k=2*block]
        b11 = 6 + 2 * block + 1  # V index for B[j=1, k=2*block+1]
        
        # W indices for output columns
        c00 = 2 * block      # W index for C[i=0, k=2*block]
        c01 = 2 * block + 1  # W index for C[i=0, k=2*block+1]
        c10 = 6 + 2 * block      # W index for C[i=1, k=2*block]
        c11 = 6 + 2 * block + 1  # W index for C[i=1, k=2*block+1]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][col] = 1  # A00
        U[3][col] = 1  # A11
        V[b00][col] = 1  # B00
        V[b11][col] = 1  # B11
        W[c00][col] = 1  # C00
        W[c11][col] = 1  # C11
        col += 1
        
        # M2 = (A10+A11)B00
        U[2][col] = 1  # A10
        U[3][col] = 1  # A11
        V[b00][col] = 1  # B00
        W[c10][col] = 1  # C10
        col += 1
        
        # M3 = A00(B10-B11)
        U[0][col] = 1  # A00
        V[b10][col] = 1  # B10
        V[b11][col] = -1  # -B11
        W[c01][col] = 1  # C01
        col += 1
        
        # M4 = A11(B01-B00)
        U[3][col] = 1  # A11
        V[b01][col] = 1  # B01
        V[b00][col] = -1  # -B00
        W[c00][col] = 1  # C00
        W[c10][col] = 1  # C10
        col += 1
        
        # M5 = (A00+A01)B11
        U[0][col] = 1  # A00
        U[1][col] = 1  # A01
        V[b11][col] = 1  # B11
        W[c00][col] = -1  # -C00
        W[c01][col] = 1  # C01
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[2][col] = 1  # A10
        U[0][col] = -1  # -A00
        V[b00][col] = 1  # B00
        V[b10][col] = 1  # B10
        W[c11][col] = 1  # C11
        col += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[1][col] = 1  # A01
        U[3][col] = -1  # -A11
        V[b01][col] = 1  # B01
        V[b11][col] = 1  # B11
        W[c00][col] = 1  # C00
        col += 1
    
    return {"U": U, "V": V, "W": W}
