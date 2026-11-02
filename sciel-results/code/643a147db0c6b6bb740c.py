"""
Matrix multiplication 2x2 by 2x7.

Strategy:
  Use the isomorphism T_{2,2,7} ≅ T_{2,7,2} to reinterpret the problem as
  multiplying a 2x7 matrix by a 7x2 matrix, with output 2x2.
  
  Split the 7 columns of the first matrix into three pairs (columns 0-1, 2-3, 4-5)
  and one singleton (column 6).
  
  For each pair of columns, use a rank-7 Strassen-like decomposition for the
  2x2x2 sub-tensor (multiplying 2x2 by 2x2).
  For the singleton column, use a rank-4 schoolbook decomposition for the
  2x2x1 sub-tensor (multiplying 2x2 by 2x1).
  
  Total rank: 3*7 + 4 = 25.
  
  The tensor isomorphism T_{n,m,p} ≅ T_{m,p,n} means we can permute the roles.
  Original: A (2x2) * B (2x7) = C (2x7).
  Reinterpreted: A' (2x7) * B' (7x2) = C' (2x2).
  
  In the original indexing:
    A has indices (i,j) where i in {0,1} (rows), j in {0,1} (cols)
    B has indices (j,k) where j in {0,1} (rows), k in {0,...,6} (cols)
    C has indices (i,k) where i in {0,1} (rows), k in {0,...,6} (cols)
  
  In the reinterpreted form:
    A' has indices (i,k) - this is C transposed, size 2x7
    B' has indices (k,j) - this is B transposed, size 7x2
    C' has indices (i,j) - this is A, size 2x2
    
  But we need to be careful with the trace convention.
  
  Actually, let me think about this differently. The standard decomposition for
  matrix multiplication C = A * B where A is n x m and B is m x p:
  
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  The tensor decomposition has:
    U[i*m + j, c] for A's indices
    V[j*p + k, c] for B's indices  
    W[k*n + i, c] for C's indices (following the task's convention W[k*n+i, r])
  
  For n=2, m=2, p=7:
    U is 4 x rank (indices: i*2+j for i in {0,1}, j in {0,1})
    V is 14 x rank (indices: j*7+k for j in {0,1}, k in {0,...,6})
    W is 14 x rank (indices: k*2+i for k in {0,...,6}, i in {0,1})
  
  Now, to exploit the structure, I'll group the k indices:
  - For k pairs (0,1), (2,3), (4,5): use rank-7 decomposition
  - For k=6: use rank-4 decomposition
  
  For a 2x2x2 block (k in {k0, k0+1}):
  C[i, k0]   = A[i,0]*B[0,k0]   + A[i,1]*B[1,k0]
  C[i, k0+1] = A[i,0]*B[0,k0+1] + A[i,1]*B[1,k0+1]
  
  This is like multiplying a 2x2 matrix A by a 2x2 matrix B_block, where
  B_block[j, l] = B[j, k0+l] for l in {0,1}.
  
  The Strassen decomposition for 2x2x2:
  Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
  M1 = (c+d)*e       -> C[1,0] += M1, C[0,0] -= M1  (wait, let me get this right)
  
  Standard Strassen:
  M1 = (a+b)*(e+f)   -> C[0,0] += M1, C[0,1] -= M1, C[1,0] -= M1, C[1,1] += M1
  M2 = (c+d)*f       -> C[0,1] += M2, C[1,1] += M2
  M3 = a*(g-h)       -> C[0,0] += M3, C[1,0] += M3
  M4 = d*(e-f)       -> C[0,1] += M4, C[1,1] -= M4
  M5 = (a+b)*h       -> C[0,1] += M5, C[1,1] += M5
  M6 = (c-a)*(e+g)   -> C[0,0] += M6, C[1,0] -= M6
  M7 = (b-d)*(f+h)   -> C[0,1] -= M7, C[1,1] -= M7
  
  Wait, let me use the standard Strassen formulas:
  M1 = (a + b) * (e + f)
  M2 = (c + d) * e
  M3 = (c + d) * f
  M4 = a * (g - h)
  M5 = d * (e - f)
  M6 = (a - c) * (e + g)
  M7 = (b - d) * (f + h)
  
  C[0,0] = M1 - M3 + M4 + M6
  C[0,1] = M3 + M5 - M7
  C[1,0] = M2 - M4 - M6
  C[1,1] = M1 - M2 + M5 + M7
  
  For the original problem with n=2, m=2, p=7:
  A[i,j] for i in {0,1}, j in {0,1}
  B[j,k] for j in {0,1}, k in {0,...,6}
  C[i,k] for i in {0,1}, k in {0,...,6}
  
  For block k0 (k0 in {0, 2, 4}):
  Let a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
  Let e = B[0,k0], f = B[0,k0+1], g = B[1,k0], h = B[1,k0+1]
  
  M1 = (a+b)*(e+f)
    U: a=1, b=1 -> U[0, col]=1, U[1, col]=1
    V: e=1, f=1 -> V[k0, col]=1, V[k0+1, col]=1
    W: C[0,0]+=1, C[0,1]+=1, C[1,0]-=1, C[1,1]-=1
    W indices: k0*2+0, k0*2+1, (k0+1)*2+0, (k0+1)*2+1
    W[k0*2+0, col] += 1
    W[k0*2+1, col] += 1
    W[(k0+1)*2+0, col] -= 1
    W[(k0+1)*2+1, col] -= 1
  
  M2 = (c+d)*e
    U: c=1, d=1 -> U[2, col]=1, U[3, col]=1
    V: e=1 -> V[k0, col]=1
    W: C[1,0]-=1, C[1,1]-=1
    W[(k0+1)*2+0, col] -= 1
    W[(k0+1)*2+1, col] -= 1
  
  M3 = (c+d)*f
    U: c=1, d=1 -> U[2, col]=1, U[3, col]=1
    V: f=1 -> V[k0+1, col]=1
    W: C[0,0]-=1, C[0,1]+=1
    W[k0*2+0, col] -= 1
    W[k0*2+1, col] += 1
  
  M4 = a*(g-h)
    U: a=1 -> U[0, col]=1
    V: g=1, h=-1 -> V[k0, col]=1, V[k0+1, col]=-1
    W: C[0,0]+=1, C[1,0]+=1
    W[k0*2+0, col] += 1
    W[(k0+1)*2+0, col] += 1
  
  M5 = d*(e-f)
    U: d=1 -> U[3, col]=1
    V: e=1, f=-1 -> V[k0, col]=1, V[k0+1, col]=-1
    W: C[0,1]+=1, C[1,1]+=1
    W[k0*2+1, col] += 1
    W[(k0+1)*2+1, col] += 1
  
  M6 = (a-c)*(e+g)
    U: a=1, c=-1 -> U[0, col]=1, U[2, col]=-1
    V: e=1, g=1 -> V[k0, col]=1, V[k0+1, col]=1  (wait, g is B[1,k0], so V index is 1*7+k0 = 7+k0)
    
  Hmm, I need to be more careful with V indices.
  V[j*p + k, col] where j in {0,1}, k in {0,...,6}, p=7.
  So V[0*7+k, col] = V[k, col] for B[0,k]
  And V[1*7+k, col] = V[7+k, col] for B[1,k]
  
  Let me redo:
  M6 = (a-c)*(e+g) where e=B[0,k0], g=B[1,k0]
    U: a=1, c=-1 -> U[0, col]=1, U[2, col]=-1
    V: e=1, g=1 -> V[0*7+k0, col]=V[k0, col]=1, V[1*7+k0, col]=V[7+k0, col]=1
    W: C[0,0]+=1, C[1,0]-=1
    W[k0*2+0, col] += 1
    W[(k0+1)*2+0, col] -= 1
  
  M7 = (b-d)*(f+h) where f=B[0,k0+1], h=B[1,k0+1]
    U: b=1, d=-1 -> U[1, col]=1, U[3, col]=-1
    V: f=1, h=1 -> V[0*7+(k0+1), col]=V[k0+1, col]=1, V[1*7+(k0+1), col]=V[7+k0+1, col]=1
    W: C[0,1]-=1, C[1,1]-=1
    W[k0*2+1, col] -= 1
    W[(k0+1)*2+1, col] -= 1
  
  For the singleton k=6:
  C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
  C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
  
  Schoolbook (4 terms):
  T1: A[0,0]*B[0,6]
    U[0, col]=1
    V[6, col]=1
    W[6*2+0, col]=W[12, col]=1
  
  T2: A[0,1]*B[1,6]
    U[1, col]=1
    V[7+6, col]=V[13, col]=1
    W[6*2+0, col]=W[12, col]+=1
  
  T3: A[1,0]*B[0,6]
    U[2, col]=1
    V[6, col]=1
    W[6*2+1, col]=W[13, col]=1
  
  T4: A[1,1]*B[1,6]
    U[3, col]=1
    V[13, col]=1
    W[6*2+1, col]=W[13, col]+=1
  
  Total rank: 3*7 + 4 = 25
"""

def construct():
    n, m, p = 2, 2, 7
    rank = 25
    U = [[0] * rank for _ in range(n * m)]  # 4 x 25
    V = [[0] * rank for _ in range(m * p)]  # 14 x 25
    W = [[0] * rank for _ in range(p * n)]  # 14 x 25
    
    col = 0
    
    # Three Strassen blocks for k0 in {0, 2, 4}
    for k0 in [0, 2, 4]:
        # M1 = (a+b)*(e+f)
        # U: a=1 (idx 0), b=1 (idx 1)
        U[0][col] += 1
        U[1][col] += 1
        # V: e=B[0,k0]=idx k0, f=B[0,k0+1]=idx k0+1
        V[k0][col] += 1
        V[k0 + 1][col] += 1
        # W: C[0,k0]+=1 (idx k0*2+0), C[0,k0+1]+=1 (idx k0*2+1)
        #     C[1,k0]-=1 (idx (k0+1)*2+0), C[1,k0+1]-=1 (idx (k0+1)*2+1)
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[(k0 + 1) * 2 + 0][col] -= 1
        W[(k0 + 1) * 2 + 1][col] -= 1
        col += 1
        
        # M2 = (c+d)*e
        # U: c=1 (idx 2), d=1 (idx 3)
        U[2][col] += 1
        U[3][col] += 1
        # V: e=B[0,k0]=idx k0
        V[k0][col] += 1
        # W: C[1,k0]-=1 (idx (k0+1)*2+0), C[1,k0+1]-=1 (idx (k0+1)*2+1)
        W[(k0 + 1) * 2 + 0][col] -= 1
        W[(k0 + 1) * 2 + 1][col] -= 1
        col += 1
        
        # M3 = (c+d)*f
        # U: c=1 (idx 2), d=1 (idx 3)
        U[2][col] += 1
        U[3][col] += 1
        # V: f=B[0,k0+1]=idx k0+1
        V[k0 + 1][col] += 1
        # W: C[0,k0]-=1 (idx k0*2+0), C[0,k0+1]+=1 (idx k0*2+1)
        W[k0 * 2 + 0][col] -= 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M4 = a*(g-h)
        # U: a=1 (idx 0)
        U[0][col] += 1
        # V: g=B[1,k0]=idx 7+k0, h=B[1,k0+1]=idx 7+k0+1
        V[7 + k0][col] += 1
        V[7 + k0 + 1][col] -= 1
        # W: C[0,k0]+=1 (idx k0*2+0), C[1,k0]+=1 (idx (k0+1)*2+0)
        W[k0 * 2 + 0][col] += 1
        W[(k0 + 1) * 2 + 0][col] += 1
        col += 1
        
        # M5 = d*(e-f)
        # U: d=1 (idx 3)
        U[3][col] += 1
        # V: e=B[0,k0]=idx k0, f=B[0,k0+1]=idx k0+1
        V[k0][col] += 1
        V[k0 + 1][col] -= 1
        # W: C[0,k0+1]+=1 (idx k0*2+1), C[1,k0+1]+=1 (idx (k0+1)*2+1)
        W[k0 * 2 + 1][col] += 1
        W[(k0 + 1) * 2 + 1][col] += 1
        col += 1
        
        # M6 = (a-c)*(e+g)
        # U: a=1 (idx 0), c=-1 (idx 2)
        U[0][col] += 1
        U[2][col] -= 1
        # V: e=B[0,k0]=idx k0, g=B[1,k0]=idx 7+k0
        V[k0][col] += 1
        V[7 + k0][col] += 1
        # W: C[0,k0]+=1 (idx k0*2+0), C[1,k0]-=1 (idx (k0+1)*2+0)
        W[k0 * 2 + 0][col] += 1
        W[(k0 + 1) * 2 + 0][col] -= 1
        col += 1
        
        # M7 = (b-d)*(f+h)
        # U: b=1 (idx 1), d=-1 (idx 3)
        U[1][col] += 1
        U[3][col] -= 1
        # V: f=B[0,k0+1]=idx k0+1, h=B[1,k0+1]=idx 7+k0+1
        V[k0 + 1][col] += 1
        V[7 + k0 + 1][col] += 1
        # W: C[0,k0+1]-=1 (idx k0*2+1), C[1,k0+1]-=1 (idx (k0+1)*2+1)
        W[k0 * 2 + 1][col] -= 1
        W[(k0 + 1) * 2 + 1][col] -= 1
        col += 1
    
    # Singleton block for k=6: schoolbook 2x2x1
    k = 6
    # T1: A[0,0]*B[0,6]
    U[0][col] += 1
    V[k][col] += 1
    W[k * 2 + 0][col] += 1
    col += 1
    
    # T2: A[0,1]*B[1,6]
    U[1][col] += 1
    V[7 + k][col] += 1
    W[k * 2 + 0][col] += 1
    col += 1
    
    # T3: A[1,0]*B[0,6]
    U[2][col] += 1
    V[k][col] += 1
    W[k * 2 + 1][col] += 1
    col += 1
    
    # T4: A[1,1]*B[1,6]
    U[3][col] += 1
    V[7 + k][col] += 1
    W[k * 2 + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
