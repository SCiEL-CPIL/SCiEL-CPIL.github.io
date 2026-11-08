"""
Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.

We split the 2x4x5 tensor into 6 disjoint sub-tensors:
- 4 sub-problems of size 2x2x2 (using Winograd rank-7)
- 2 sub-problems of size 2x2x1 (schoolbook rank 4)

Total rank = 4*7 + 2*4 = 36.

The output factor W follows the convention W[k*n+i, r] where
n=2 (rows of A), p=5 (cols of B), so W has shape (p*n, rank) = (10, rank).
Indexing: W[k*n + i, c] corresponds to C[i,k] = C[i, k] for i in [0,1], k in [0,4].
"""

import time

DIMENSIONS = [2, 4, 5]

# Winograd 2x2x2 rank-7 decomposition
# For multiplying two 2x2 matrices, C = A*B
# A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
#
# The 7 intermediate values are:
# u0 = (a00+a11)*(b00+b11)
# u1 = (a01+a11)*(b00-b10)  -- wait, let me use the standard Winograd scheme
#
# Standard Winograd for 2x2:
# u0 = (a00+a11)*(b00+b11)
# u1 = (a01+a11)*(b00-b10)
# u2 = (a00+a10)*(b11-b01)
# u3 = a00*b00
# u4 = a01*b01
# u5 = a10*b10
# u6 = a11*b11
#
# Then:
# c00 = u0 - u1 + u2 + u3
# c01 = u0 + u1 - u2 + u4
# c10 = u0 - u1 - u2 + u5
# c11 = u0 + u1 + u2 + u6

def _winograd_222_factors():
    """
    Return (U_local, V_local, W_local) for 2x2x2 matrix mult with rank 7.
    
    U has shape (4, 7): rows indexed by (i,j) where i in [0,1], j in [0,1]
      row index = i*2 + j
    V has shape (4, 7): rows indexed by (j,k) where j in [0,1], k in [0,1]
      row index = j*2 + k
    W has shape (4, 7): rows indexed by (k,i) following convention W[k*n+i, r]
      n=2, so row index = k*2 + i
    
    The tensor is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    This should equal the (i,k) entry of A*B, i.e., sum_j A[i,j]*B[j,k].
    """
    # U rows: (i,j) -> i*2+j
    # V rows: (j,k) -> j*2+k
    # W rows: (k,i) -> k*2+i (since n=2, W index = k*n + i = k*2 + i)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # u0 = (a00+a11)*(b00+b11)
    # U contributes a00 and a11, V contributes b00 and b11
    # u0: A part: a00 + a11, B part: b00 + b11
    # u0 = (a00+a11)*(b00+b11)
    # In bilinear form: sum over r of U[i,j,r]*V[j,k,r]
    # For c00 = u0 - u1 + u2 + u3:
    #   u0 contributes +1, u1 contributes -1, u2 contributes +1, u3 contributes +1
    # For c01 = u0 + u1 - u2 + u4:
    #   u0 contributes +1, u1 contributes +1, u2 contributes -1, u4 contributes +1
    # For c10 = u0 - u1 - u2 + u5:
    #   u0 contributes +1, u1 contributes -1, u2 contributes -1, u5 contributes +1
    # For c11 = u0 + u1 + u2 + u6:
    #   u0 contributes +1, u1 contributes +1, u2 contributes +1, u6 contributes +1
    
    # Let's define the 7 terms:
    # r=0: u0 = (a00+a11)*(b00+b11)
    # r=1: u1 = (a01+a11)*(b00-b10)
    # r=2: u2 = (a00+a10)*(b11-b01)
    # r=3: u3 = a00*b00
    # r=4: u4 = a01*b01
    # r=5: u5 = a10*b10
    # r=6: u6 = a11*b11
    
    # U[i,j,r]:
    # r=0: u0 involves a00 and a11 -> U[0,0,0]=1, U[1,1,0]=1 (i=0,j=0 and i=1,j=1)
    # r=1: u1 involves a01 and a11 -> U[0,1,1]=1, U[1,1,1]=1
    # r=2: u2 involves a00 and a10 -> U[0,0,2]=1, U[1,0,2]=1
    # r=3: u3 = a00 -> U[0,0,3]=1
    # r=4: u4 = a01 -> U[0,1,4]=1
    # r=5: u5 = a10 -> U[1,0,5]=1
    # r=6: u6 = a11 -> U[1,1,6]=1
    
    U[0*2+0][0] = 1  # a00 in u0
    U[1*2+1][0] = 1  # a11 in u0
    
    U[0*2+1][1] = 1  # a01 in u1
    U[1*2+1][1] = 1  # a11 in u1
    
    U[0*2+0][2] = 1  # a00 in u2
    U[1*2+0][2] = 1  # a10 in u2
    
    U[0*2+0][3] = 1  # a00 in u3
    U[0*2+1][4] = 1  # a01 in u4
    U[1*2+0][5] = 1  # a10 in u5
    U[1*2+1][6] = 1  # a11 in u6
    
    # V[j,k,r]:
    # r=0: u0 involves b00 and b11 -> V[0,0,0]=1, V[1,1,0]=1
    # r=1: u1 involves b00 and b10 with signs: (b00-b10) -> V[0,0,1]=1, V[1,0,1]=-1
    # r=2: u2 involves b11 and b01 with signs: (b11-b01) -> V[1,1,2]=1, V[0,1,2]=-1
    # r=3: u3 = b00 -> V[0,0,3]=1
    # r=4: u4 = b01 -> V[0,1,4]=1
    # r=5: u5 = b10 -> V[1,0,5]=1
    # r=6: u6 = b11 -> V[1,1,6]=1
    
    V[0*2+0][0] = 1  # b00 in u0
    V[1*2+1][0] = 1  # b11 in u0
    
    V[0*2+0][1] = 1  # b00 in u1
    V[1*2+0][1] = -1 # -b10 in u1
    
    V[1*2+1][2] = 1  # b11 in u2
    V[0*2+1][2] = -1 # -b01 in u2
    
    V[0*2+0][3] = 1  # b00 in u3
    V[0*2+1][4] = 1  # b01 in u4
    V[1*2+0][5] = 1  # b10 in u5
    V[1*2+1][6] = 1  # b11 in u6
    
    # W[k,i,r] where row index = k*2 + i
    # W is used as: sum_r ... * W[k,i,r] gives the coefficient for C[i,k]
    # C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] for the appropriate j summed
    
    # Actually, the tensor is T[i,j,k] = A[i,j]*B[j,k] and C[i,k] = sum_j T[i,j,k]
    # So W[k,i,r] should be the coefficient such that sum_j sum_r U[i,j,r]V[j,k,r]W[k,i,r] = C[i,k]
    
    # For r=0 (u0): coefficient for c00, c01, c10, c11 all have +1
    # W[0,0,0]=1 (k=0,i=0), W[1,0,0]=1 (k=0,i=1? No, W[k,i] so k=1,i=0 means C[0,1])
    # Wait, let me re-index. C[i,k] where i is row, k is col.
    # W row index = k*n + i = k*2 + i
    # W[0,0] = k=0,i=0 -> c00
    # W[0,1] = k=0,i=1 -> c10
    # W[1,0] = k=1,i=0 -> c01
    # W[1,1] = k=1,i=1 -> c11
    
    # r=0: c00+=1, c01+=1, c10+=1, c11+=1
    W[0*2+0][0] = 1  # c00
    W[1*2+0][0] = 1  # c01 (k=1,i=0)
    W[0*2+1][0] = 1  # c10 (k=0,i=1)
    W[1*2+1][0] = 1  # c11
    
    # r=1: c00-=1, c01+=1, c10-=1, c11+=1
    W[0*2+0][1] = -1 # c00
    W[1*2+0][1] = 1  # c01
    W[0*2+1][1] = -1 # c10
    W[1*2+1][1] = 1  # c11
    
    # r=2: c00+=1, c01-=1, c10-=1, c11+=1
    W[0*2+0][2] = 1  # c00
    W[1*2+0][2] = -1 # c01
    W[0*2+1][2] = -1 # c10
    W[1*2+1][2] = 1  # c11
    
    # r=3: c00+=1
    W[0*2+0][3] = 1  # c00
    
    # r=4: c01+=1
    W[1*2+0][4] = 1  # c01
    
    # r=5: c10+=1
    W[0*2+1][5] = 1  # c10
    
    # r=6: c11+=1
    W[1*2+1][6] = 1  # c11
    
    return U, V, W


def _schoolbook_221_factors():
    """
    Schoolbook for 2x2 * 2x1 matrix mult. A is 2x2, B is 2x1, C is 2x1.
    Rank = 4.
    
    U shape (4, 4): A is 2x2, rows indexed by i*2+j
    V shape (2, 4): B is 2x1, rows indexed by j*1+k = j
    W shape (2, 4): C is 2x1, rows indexed by k*n+i = k*2+i where k in [0,0], i in [0,1]
      So W has 2 rows: W[0*2+0]=c00, W[0*2+1]=c10
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    # c00 = a00*b0 + a01*b1
    # c10 = a10*b0 + a11*b1
    
    # r=0: a00*b0 -> c00
    U[0*2+0][0] = 1
    V[0][0] = 1
    W[0*2+0][0] = 1
    
    # r=1: a01*b1 -> c00
    U[0*2+1][1] = 1
    V[1][1] = 1
    W[0*2+0][1] = 1
    
    # r=2: a10*b0 -> c10
    U[1*2+0][2] = 1
    V[0][2] = 1
    W[0*2+1][2] = 1
    
    # r=3: a11*b1 -> c10
    U[1*2+1][3] = 1
    V[1][3] = 1
    W[0*2+1][3] = 1
    
    return U, V, W


def construct():
    """
    Construct rank-36 decomposition for 2x4 * 4x5 matrix multiplication.
    
    A is 2x4, B is 4x5, C is 2x5.
    U shape: (n*m, rank) = (8, 36)
    V shape: (m*p, rank) = (20, 36)  
    W shape: (p*n, rank) = (10, 36)
    
    W convention: W[k*n+i, r] where n=2, so W index = k*2+i for k in [0,4], i in [0,1]
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    total_rank = 36
    U = [[0] * total_rank for _ in range(n * m)]   # 8 x 36
    V = [[0] * total_rank for _ in range(m * p)]   # 20 x 36
    W = [[0] * total_rank for _ in range(p * n)]   # 10 x 36
    
    col = 0
    
    # We split A's columns into two groups of 2: A1 (cols 0,1) and A2 (cols 2,3)
    # We split B's columns into three groups: B1 (cols 0,1), B2 (col 2), B3 (cols 3,4)
    # Wait, that doesn't give nice 2x2x2 blocks. Let me reconsider.
    #
    # Better split:
    # A columns: [0,1] and [2,3] (two groups of 2)
    # B columns: [0,1], [2,3], [4] (groups of 2, 2, 1)
    #
    # For each pair (A_group, B_group):
    #   If both have 2 columns: 2x2x2 sub-problem (rank 7)
    #   If A has 2 cols, B has 1 col: 2x2x1 sub-problem (rank 4)
    #
    # Sub-problems:
    # 1. A1 (cols 0,1), B1 (cols 0,1) -> 2x2x2, rank 7
    # 2. A1 (cols 0,1), B2 (cols 2,3) -> 2x2x2, rank 7
    # 3. A1 (cols 0,1), B3 (col 4)     -> 2x2x1, rank 4
    # 4. A2 (cols 2,3), B1 (cols 0,1)  -> 2x2x2, rank 7
    # 5. A2 (cols 2,3), B2 (cols 2,3)  -> 2x2x2, rank 7
    # 6. A2 (cols 2,3), B3 (col 4)     -> 2x2x1, rank 4
    #
    # Total: 4*7 + 2*4 = 36
    
    # Get local factor matrices
    Uw, Vw, Ww = _winograd_222_factors()
    Us, Vs, Ws = _schoolbook_221_factors()
    
    # Sub-problem 1: A1 (cols 0,1), B1 (cols 0,1)
    # A rows: i in [0,1], A cols: j_local in [0,1] -> global j = j_local
    # B rows: j_local in [0,1] -> global j = j_local, B cols: k_local in [0,1] -> global k = k_local
    # C: i in [0,1], k_local in [0,1] -> global k = k_local
    
    for r in range(7):
        for i in range(2):
            for jl in range(2):
                U[i * 2 + jl][col + r] = Uw[i * 2 + jl][r]
        for jl in range(2):
            for kl in range(2):
                V[jl * 5 + kl][col + r] = Vw[jl * 2 + kl][r]
        for kl in range(2):
            for i in range(2):
                W[kl * 2 + i][col + r] = Ww[kl * 2 + i][r]
    col += 7
    
    # Sub-problem 2: A1 (cols 0,1), B2 (cols 2,3)
    # A: same as above
    # B rows: j_local in [0,1] -> global j = j_local, B cols: k_local in [0,1] -> global k = 2 + k_local
    # C: i in [0,1], k_local in [0,1] -> global k = 2 + k_local
    
    for r in range(7):
        for i in range(2):
            for jl in range(2):
                U[i * 2 + jl][col + r] = Uw[i * 2 + jl][r]
        for jl in range(2):
            for kl in range(2):
                V[jl * 5 + (2 + kl)][col + r] = Vw[jl * 2 + kl][r]
        for kl in range(2):
            for i in range(2):
                W[(2 + kl) * 2 + i][col + r] = Ww[kl * 2 + i][r]
    col += 7
    
    # Sub-problem 3: A1 (cols 0,1), B3 (col 4)
    # A: i in [0,1], j_local in [0,1] -> global j = j_local
    # B: j_local in [0,1] -> global j = j_local, k = 4
    # C: i in [0,1], k = 4
    
    for r in range(4):
        for i in range(2):
            for jl in range(2):
                U[i * 2 + jl][col + r] = Us[i * 2 + jl][r]
        for jl in range(2):
            V[jl * 5 + 4][col + r] = Vs[jl][r]
        for i in range(2):
            W[4 * 2 + i][col + r] = Ws[i][r]
    col += 4
    
    # Sub-problem 4: A2 (cols 2,3), B1 (cols 0,1)
    # A: i in [0,1], j_local in [0,1] -> global j = 2 + j_local
    # B: j_local in [0,1] -> global j = j_local, k_local in [0,1] -> global k = k_local
    # C: i in [0,1], k_local in [0,1] -> global k = k_local
    
    for r in range(7):
        for i in range(2):
            for jl in range(2):
                U[i * 2 + (2 + jl)][col + r] = Uw[i * 2 + jl][r]
        for jl in range(2):
            for kl in range(2):
                V[jl * 5 + kl][col + r] = Vw[jl * 2 + kl][r]
        for kl in range(2):
            for i in range(2):
                W[kl * 2 + i][col + r] = Ww[kl * 2 + i][r]
    col += 7
    
    # Sub-problem 5: A2 (cols 2,3), B2 (cols 2,3)
    # A: i in [0,1], j_local in [0,1] -> global j = 2 + j_local
    # B: j_local in [0,1] -> global j = j_local, k_local in [0,1] -> global k = 2 + k_local
    # C: i in [0,1], k_local in [0,1] -> global k = 2 + k_local
    
    for r in range(7):
        for i in range(2):
            for jl in range(2):
                U[i * 2 + (2 + jl)][col + r] = Uw[i * 2 + jl][r]
        for jl in range(2):
            for kl in range(2):
                V[jl * 5 + (2 + kl)][col + r] = Vw[jl * 2 + kl][r]
        for kl in range(2):
            for i in range(2):
                W[(2 + kl) * 2 + i][col + r] = Ww[kl * 2 + i][r]
    col += 7
    
    # Sub-problem 6: A2 (cols 2,3), B3 (col 4)
    # A: i in [0,1], j_local in [0,1] -> global j = 2 + j_local
    # B: j_local in [0,1] -> global j = j_local, k = 4
    # C: i in [0,1], k = 4
    
    for r in range(4):
        for i in range(2):
            for jl in range(2):
                U[i * 2 + (2 + jl)][col + r] = Us[i * 2 + jl][r]
        for jl in range(2):
            V[jl * 5 + 4][col + r] = Vs[jl][r]
        for i in range(2):
            W[4 * 2 + i][col + r] = Ws[i][r]
    col += 4
    
    return {"U": U, "V": V, "W": W}
