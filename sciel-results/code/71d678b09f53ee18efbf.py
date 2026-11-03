"""
Tensor decomposition for 2x4 matrix times 4x6 matrix.
Strategy: treat as 2x2 block multiplication with 2x2 blocks.
Each 2x2 block multiplication (A_block 2x2 times B_block 2x2) has rank 7 (Strassen).
Total rank = 7 blocks * 7 per block = 49.

But we can do better: the block structure gives us 2x2 blocks of size:
A is 2x4 -> split into two 2x2 blocks: A1 = A[:,0:2], A2 = A[:,2:4]
B is 4x6 -> split into three 2x2 blocks: B1 = B[0:2,:], B2 = B[2:4,:]  -- wait, B is 4x6, so 2 blocks of 2x3.

Actually, let me reconsider. The block structure should be:
A (2x4) = [A1 | A2] where A1, A2 are 2x2
B (4x6) = [B1; B2] where B1, B2 are 2x6

C = A1*B1 + A2*B2

Each of A1*B1 and A2*B2 is a (2x2)*(2x6) multiplication.
A 2x2 times 2x6 can be viewed as 2x2 times 2x2 for each pair of columns of B.
So A1*B1: B1 has 6 columns, we can group into 3 groups of 2 columns.
For each group, it's a 2x2 * 2x2 multiplication, rank 7 via Strassen.
So A1*B1 has rank 3*7 = 21, and A2*B2 has rank 3*7 = 21.
Total rank = 42.

This matches the observed improvement from 48 to 42.

Let me implement the Strassen-based rank-42 decomposition.

Strassen for 2x2 * 2x2:
Let A = [[a, b], [c, d]], B = [[w, x], [y, z]]
Then C = AB where:
P1 = (a+d)(w+z)
P2 = (c+d)w
P3 = a(x-z)
P4 = d(y-x)
P5 = (a+b)z
P6 = (c-a)(w+x)
P7 = (b-d)(y+z)

C[0,0] = P1 + P4 - P5 + P7
C[0,1] = P3 + P5
C[1,0] = P2 + P4
C[1,1] = P1 - P2 + P3 + P6

Let me build the full decomposition.

For the 2x4 * 4x6 case:
A has entries A[i,j] for i in {0,1}, j in {0,1,2,3}
B has entries B[j,k] for j in {0,1,2,3}, k in {0,1,2,3,4,5}
C has entries C[i,k] for i in {0,1}, k in {0,1,2,3,4,5}

C[i,k] = sum_j A[i,j] * B[j,k]
       = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
       = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])

The first part is A1[i,:] * B1[:,k] where A1 = A[:,0:2], B1 = B[0:2,:]
The second part is A2[i,:] * B2[:,k] where A2 = A[:,2:4], B2 = B[2:4,:]

For each of A1*B1 and A2*B2, we group B's columns into pairs:
B1 has columns 0-5, grouped as (0,1), (2,3), (4,5)
For group g with columns (2g, 2g+1):
  A1 * B1[:, 2g:2g+2] is a 2x2 * 2x2 multiplication
  Let B1g = [[B[0,2g], B[0,2g+1]], [B[1,2g], B[1,2g+1]]]
  Result R has entries R[i, 2g+0] and R[i, 2g+1]

Using Strassen for each such 2x2*2x2:
  a = A[i,0], b = A[i,1] for i=0; a = A[1,0], b = A[1,1] for i=1
  Wait, A1 is 2x2: A1[0,0], A1[0,1]; A1[1,0], A1[1,1]
  B1g is 2x2: B1g[0,0], B1g[0,1]; B1g[1,0], B1g[1,1]

Strassen products:
  P1 = (A1[0,0]+A1[1,1])*(B1g[0,0]+B1g[1,1])
  P2 = (A1[1,0]+A1[1,1])*B1g[0,0]
  P3 = A1[0,0]*(B1g[0,1]-B1g[1,1])
  P4 = A1[1,1]*(B1g[1,0]-B1g[0,0])
  P5 = (A1[0,0]+A1[0,1])*B1g[1,1]
  P6 = (A1[1,0]-A1[0,0])*(B1g[0,0]+B1g[0,1])
  P7 = (A1[0,1]-A1[1,1])*(B1g[1,0]+B1g[1,1])

  R[0,0] = P1 + P4 - P5 + P7
  R[0,1] = P3 + P5
  R[1,0] = P2 + P4
  R[1,1] = P1 - P2 + P3 + P6

Now I need to build U, V, W matrices.

U has shape (n*m, rank) = (8, 42)
V has shape (m*p, rank) = (24, 42)
W has shape (p*n, rank) = (12, 42)

For each of the 6 Strassen sub-multiplications (2 for A1, 3 groups each = 6 total... wait, 2 matrices A1 and A2, each with 3 column groups = 6 total 2x2*2x2 Strassen decompositions, each of rank 7, total 42).

Let me index:
- Part 1: A1 * B1[:, 0:2] (columns 0,1 of B)
- Part 2: A1 * B1[:, 2:4] (columns 2,3 of B)
- Part 3: A1 * B1[:, 4:6] (columns 4,5 of B)
- Part 4: A2 * B2[:, 0:2]
- Part 5: A2 * B2[:, 2:4]
- Part 6: A2 * B2[:, 4:6]

For each part, I have 7 Strassen terms.

A1 indices: A1[0,0] = A[0,0] = U idx 0*4+0=0; A1[0,1]=A[0,1]=U idx 1; A1[1,0]=A[1,0]=U idx 4; A1[1,1]=A[1,1]=U idx 5
A2 indices: A2[0,0]=A[0,2]=U idx 2; A2[0,1]=A[0,3]=U idx 3; A2[1,0]=A[1,2]=U idx 6; A2[1,1]=A[1,3]=U idx 7

B1[:,0:2]: B1[0,0]=B[0,0]=V idx 0*6+0=0; B1[0,1]=B[0,1]=V idx 1; B1[1,0]=B[1,0]=V idx 6; B1[1,1]=B[1,1]=V idx 7
B1[:,2:4]: B1[0,0]=B[0,2]=V idx 2; B1[0,1]=B[0,3]=V idx 3; B1[1,0]=B[1,2]=V idx 8; B1[1,1]=B[1,3]=V idx 9
B1[:,4:6]: B1[0,0]=B[0,4]=V idx 4; B1[0,1]=B[0,5]=V idx 5; B1[1,0]=B[1,4]=V idx 10; B1[1,1]=B[1,5]=V idx 11

B2[:,0:2]: B2[0,0]=B[2,0]=V idx 12; B2[0,1]=B[2,1]=V idx 13; B2[1,0]=B[3,0]=V idx 18; B2[1,1]=B[3,1]=V idx 19
B2[:,2:4]: B2[0,0]=B[2,2]=V idx 14; B2[0,1]=B[2,3]=V idx 15; B2[1,0]=B[3,2]=V idx 20; B2[1,1]=B[3,3]=V idx 21
B2[:,4:6]: B2[0,0]=B[2,4]=V idx 16; B2[0,1]=B[2,5]=V idx 17; B2[1,0]=B[3,4]=V idx 22; B2[1,1]=B[3,5]=V idx 23

C indices (W convention: W[k*n+i, r] where n=2):
C[0,0] = W[0*2+0] = W[0]
C[0,1] = W[1*2+0] = W[2]
C[0,2] = W[2*2+0] = W[4]
C[0,3] = W[3*2+0] = W[6]
C[0,4] = W[4*2+0] = W[8]
C[0,5] = W[5*2+0] = W[10]
C[1,0] = W[0*2+1] = W[1]
C[1,1] = W[1*2+1] = W[3]
C[1,2] = W[2*2+1] = W[5]
C[1,3] = W[3*2+1] = W[7]
C[1,4] = W[4*2+1] = W[9]
C[1,5] = W[5*2+1] = W[11]
"""

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    # Helper: A index for A1 (columns 0,1) or A2 (columns 2,3)
    # A[i, j] -> U[i*4 + j]
    # A1: a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[4], d=A[1,1]=U[5]
    # A2: a=A[0,2]=U[2], b=A[0,3]=U[3], c=A[1,2]=U[6], d=A[1,3]=U[7]
    
    def strassen(a, b, c, d, w, x, y, z, col_start, U, V, W):
        """
        Strassen decomposition for 2x2*2x2.
        a,b,c,d: U row indices for A matrix [[a,b],[c,d]]
        w,x,y,z: V row indices for B matrix [[w,x],[y,z]]
        col_start: starting column index for this block of 7
        W indices: for C = AB, C[0,0],C[0,1],C[1,0],C[1,1] are at specific W rows.
        But we need to map these to the correct W rows based on which output columns they correspond to.
        """
        cols = [col_start, col_start+1, col_start+2, col_start+3, col_start+4, col_start+5, col_start+6]
        c0, c1, c2, c3, c4, c5, c6 = cols
        
        # P1 = (a+d)(w+z)
        U[a][c0] += 1; U[d][c0] += 1
        V[w][c0] += 1; V[z][c0] += 1
        # P2 = (c+d)w
        U[c][c1] += 1; U[d][c1] += 1
        V[w][c1] += 1
        # P3 = a(x-z)
        U[a][c2] += 1
        V[x][c2] += 1; V[z][c2] -= 1
        # P4 = d(y-x)
        U[d][c3] += 1
        V[y][c3] += 1; V[x][c3] -= 1
        # P5 = (a+b)z
        U[a][c4] += 1; U[b][c4] += 1
        V[z][c4] += 1
        # P6 = (c-a)(w+x)
        U[c][c5] += 1; U[a][c5] -= 1
        V[w][c5] += 1; V[x][c5] += 1
        # P7 = (b-d)(y+z)
        U[b][c6] += 1; U[d][c6] -= 1
        V[y][c6] += 1; V[z][c6] += 1
        return cols
    
    # For each Strassen block, we need to know the W row indices for C[0,0], C[0,1], C[1,0], C[1,1]
    # These depend on which output columns the B block corresponds to.
    
    def strassen_with_W(a, b, c, d, w, x, y, z, col_start, U, V, W, wc00, wc01, wc10, wc11):
        """
        Same as strassen but also sets W entries.
        wc00, wc01, wc10, wc11: W row indices for C[0,0], C[0,1], C[1,0], C[1,1]
        """
        cols = [col_start, col_start+1, col_start+2, col_start+3, col_start+4, col_start+5, col_start+6]
        c0, c1, c2, c3, c4, c5, c6 = cols
        
        # P1 = (a+d)(w+z)
        U[a][c0] += 1; U[d][c0] += 1
        V[w][c0] += 1; V[z][c0] += 1
        # P2 = (c+d)w
        U[c][c1] += 1; U[d][c1] += 1
        V[w][c1] += 1
        # P3 = a(x-z)
        U[a][c2] += 1
        V[x][c2] += 1; V[z][c2] -= 1
        # P4 = d(y-x)
        U[d][c3] += 1
        V[y][c3] += 1; V[x][c3] -= 1
        # P5 = (a+b)z
        U[a][c4] += 1; U[b][c4] += 1
        V[z][c4] += 1
        # P6 = (c-a)(w+x)
        U[c][c5] += 1; U[a][c5] -= 1
        V[w][c5] += 1; V[x][c5] += 1
        # P7 = (b-d)(y+z)
        U[b][c6] += 1; U[d][c6] -= 1
        V[y][c6] += 1; V[z][c6] += 1
        
        # W contributions:
        # C[0,0] = P1 + P4 - P5 + P7
        W[wc00][c0] += 1; W[wc00][c3] += 1; W[wc00][c4] -= 1; W[wc00][c6] += 1
        # C[0,1] = P3 + P5
        W[wc01][c2] += 1; W[wc01][c4] += 1
        # C[1,0] = P2 + P4
        W[wc10][c1] += 1; W[wc10][c3] += 1
        # C[1,1] = P1 - P2 + P3 + P6
        W[wc11][c0] += 1; W[wc11][c1] -= 1; W[wc11][c2] += 1; W[wc11][c5] += 1
        
        return col_start + 7
    
    col = 0
    
    # Part 1: A1 * B1[:, 0:2]
    # A1: a=U[0], b=U[1], c=U[4], d=U[5]
    # B1[:,0:2]: w=V[0], x=V[1], y=V[6], z=V[7]
    # Output: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    col = strassen_with_W(0, 1, 4, 5, 0, 1, 6, 7, col, U, V, W, 0, 2, 1, 3)
    
    # Part 2: A1 * B1[:, 2:4]
    # B1[:,2:4]: w=V[2], x=V[3], y=V[8], z=V[9]
    # Output: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    col = strassen_with_W(0, 1, 4, 5, 2, 3, 8, 9, col, U, V, W, 4, 6, 5, 7)
    
    # Part 3: A1 * B1[:, 4:6]
    # B1[:,4:6]: w=V[4], x=V[5], y=V[10], z=V[11]
    # Output: C[0,4]=W[8], C[0,5]=W[10], C[1,4]=W[9], C[1,5]=W[11]
    col = strassen_with_W(0, 1, 4, 5, 4, 5, 10, 11, col, U, V, W, 8, 10, 9, 11)
    
    # Part 4: A2 * B2[:, 0:2]
    # A2: a=U[2], b=U[3], c=U[6], d=U[7]
    # B2[:,0:2]: w=V[12], x=V[13], y=V[18], z=V[19]
    # Output: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    col = strassen_with_W(2, 3, 6, 7, 12, 13, 18, 19, col, U, V, W, 0, 2, 1, 3)
    
    # Part 5: A2 * B2[:, 2:4]
    # B2[:,2:4]: w=V[14], x=V[15], y=V[20], z=V[21]
    # Output: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    col = strassen_with_W(2, 3, 6, 7, 14, 15, 20, 21, col, U, V, W, 4, 6, 5, 7)
    
    # Part 6: A2 * B2[:, 4:6]
    # B2[:,4:6]: w=V[16], x=V[17], y=V[22], z=V[23]
    # Output: C[0,4]=W[8], C[0,5]=W[10], C[1,4]=W[9], C[1,5]=W[11]
    col = strassen_with_W(2, 3, 6, 7, 16, 17, 22, 23, col, U, V, W, 8, 10, 9, 11)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
