import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 145.0

def construct():
    n, m, p = DIMENSIONS
    
    # Build the decomposition recursively
    # Returns (U, V, W) as lists of lists, and the rank
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
        T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
        """
        # Strassen's 7 products
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A10)(B10+B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # U[i,j,r]: coefficient for A[i,j] in product r
        # V[j,k,r]: coefficient for B[j,k] in product r
        # W[k,i,r]: coefficient for C[k,i] in product r (note: C row=k, col=i in our convention)
        
        # Wait, the convention is W[k*n+i, r] for C[i,k]... let me re-read.
        # "The output factor follows the official trace convention W[k*n+i,r]"
        # So W is indexed by (k, i) where k is the column of C and i is the row of C? 
        # No: W[k*n+i, r] means the row index is k*n+i.
        # In matrix mult C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is T[i,j,k] = 1
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # So W is indexed by (k, i) = (col of C, row of C)
        
        # For 2x2x2: n=2, m=2, p=2
        # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        # M1: A part: A00+A11, B part: B00+B11, C part: C00+C11
        # U row for M1: A00=1, A01=0, A10=0, A11=1 → [1,0,0,1]
        # V row for M1: B00=1, B01=0, B10=0, B11=1 → [1,0,0,1]
        # W row for M1: C00=1, C01=0, C10=0, C11=1 → but W[k*2+i], so:
        #   C00: k=0,i=0 → index 0
        #   C01: k=1,i=0 → index 1
        #   C10: k=0,i=1 → index 2
        #   C11: k=1,i=1 → index 3
        # W for M1: [1, 0, 0, 1]
        
        # M2: A10+A11, B00, C10+C11
        # U: [0,0,1,1], V: [1,0,0,0], W: C10+C11 → [0,0,1,1]
        
        # M3: A00, B10-B11, C01
        # U: [1,0,0,0], V: [0,0,1,-1], W: C01 → [0,1,0,0]
        
        # M4: A11, B01-B00, C00+C10
        # U: [0,0,0,1], V: [0,1,-1,0], W: C00+C10 → [1,0,1,0]
        
        # M5: A00+A01, B11, -C00+C01
        # U: [1,1,0,0], V: [0,0,0,1], W: -C00+C01 → [-1,1,0,0]
        
        # M6: A10-A00, B00+B01, C10
        # U: [0,0,1,-1], V: [1,1,0,0], W: C10 → [0,0,1,0]
        
        # M7: A01-A10, B10+B11, -C00+C11
        # U: [0,1,-1,0], V: [0,0,1,1], W: -C00+C11 → [-1,0,0,1]
        
        U = [
            [1, 0, 0, 1],  # M1
            [0, 0, 1, 1],  # M2
            [1, 0, 0, 0],  # M3
            [0, 0, 0, 1],  # M4
            [1, 1, 0, 0],  # M5
            [0, 0, 1, -1], # M6
            [0, 1, -1, 0], # M7
        ]
        V = [
            [1, 0, 0, 1],  # M1
            [1, 0, 0, 0],  # M2
            [0, 0, 1, -1], # M3
            [0, 1, -1, 0], # M4
            [0, 0, 0, 1],  # M5
            [1, 1, 0, 0],  # M6
            [0, 0, 1, 1],  # M7
        ]
        W = [
            [1, 0, 0, 1],  # M1: C00+C11
            [0, 0, 1, 1],  # M2: C10+C11
            [0, 1, 0, 0],  # M3: C01
            [1, 0, 1, 0],  # M4: C00+C10
            [-1, 1, 0, 0], # M5: -C00+C01
            [0, 0, 1, 0],  # M6: C10
            [-1, 0, 0, 1], # M7: -C00+C11
        ]
        
        # Verify: T[i,j,k] = sum_r U[i*2+j,r] * V[j*2+k,r] * W[k*2+i,r]
        # T[0,0,0] = sum_r U[0,r]*V[0,r]*W[0,r]
        # = 1*1*1 + 0*1*0 + 1*0*0 + 0*0*1 + 1*0*(-1) + 0*1*0 + 0*0*(-1)
        # = 1 + 0 + 0 + 0 + 0 + 0 + 0 = 1 ✓
        
        # T[0,0,1] = sum_r U[0,r]*V[1,r]*W[2,r]
        # = 1*0*0 + 0*0*0 + 1*1*0 + 0*(-1)*1 + 1*0*0 + 0*1*1 + 0*1*0
        # = 0 + 0 + 0 + 0 + 0 + 0 + 0 = 0 ✗
        
        # That's wrong. Let me recheck.
        # T[i,j,k] should be 1 for all i,j,k.
        # T[0,0,1]: i=0,j=0,k=1
        # U[0*2+0, r] = U[0, r]
        # V[0*2+1, r] = V[1, r]
        # W[1*2+0, r] = W[2, r]
        
        # M1: U[0]=1, V[1]=0, W[2]=0 → 0
        # M2: U[0]=0, V[1]=0, W[2]=1 → 0
        # M3: U[0]=1, V[1]=0, W[2]=0 → 0
        # M4: U[0]=0, V[1]=1, W[2]=1 → 0
        # M5: U[0]=1, V[1]=0, W[2]=0 → 0
        # M6: U[0]=0, V[1]=1, W[2]=1 → 0
        # M7: U[0]=0, V[1]=0, W[2]=0 → 0
        # Sum = 0, but should be 1. 
        
        # I think I have the W indices wrong. Let me reconsider.
        # W[k*n+i, r] where n=2.
        # W[0, r] corresponds to k=0, i=0 → C[0,0]
        # W[1, r] corresponds to k=0, i=1 → C[1,0]  (row 1, col 0)
        # W[2, r] corresponds to k=1, i=0 → C[0,1]  (row 0, col 1)
        # W[3, r] corresponds to k=1, i=1 → C[1,1]  (row 1, col 1)
        
        # So W index = k*2 + i, meaning:
        # index 0: (k=0, i=0) → C[0,0]
        # index 1: (k=0, i=1) → C[1,0]
        # index 2: (k=1, i=0) → C[0,1]
        # index 3: (k=1, i=1) → C[1,1]
        
        # Now C = A*B where C[i,k] = sum_j A[i,j] B[j,k]
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
        
        # Strassen:
        # M1 = (A00+A11)(B00+B11) → contributes to C00 and C11
        # M2 = (A10+A11)B00 → contributes to C10 and C11
        # M3 = A00(B10-B11) → contributes to C01
        # M4 = A11(B01-B00) → contributes to C00 and C10
        # M5 = (A00+A01)B11 → contributes to C01 and -C00
        # M6 = (A10-A00)(B00+B01) → contributes to C10
        # M7 = (A01-A10)(B10+B11) → contributes to C11 and -C00
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4 + M6
        # C11 = M1 - M2 + M3 + M7
        
        # Let me verify C01:
        # C01 = A00*B01 + A01*B11
        # M3 + M5 = A00*(B10-B11) + (A00+A01)*B11 = A00*B10 - A00*B11 + A00*B11 + A01*B11 = A00*B10 + A01*B11
        # But C01 = A00*B01 + A01*B11. These don't match unless B10=B01.
        
        # I think I have the Strassen formulas wrong. Let me look them up carefully.
        # Standard Strassen for 2x2:
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A10)(B10+B11)
        
        # C00 = M1 - M5 + M4 - M7
        # C01 = M5 + M3
        # C10 = M2 + M4
        # C11 = M1 - M2 + M7 + M3
        
        # Let me verify C01 = M5 + M3:
        # M5 + M3 = (A00+A01)B11 + A00(B10-B11) = A00*B11 + A01*B11 + A00*B10 - A00*B11 = A01*B11 + A00*B10
        # But C01 = A00*B01 + A01*B11. Still doesn't match.
        
        # I think the standard Strassen is:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A10)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Verify C00 = M1 + M4 - M5 - M7:
        # M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # M4 = A11*B01 - A11*B00
        # M5 = A00*B11 + A01*B11
        # M7 = A01*B10 + A01*B11 - A10*B10 - A10*B11
        
        # M1 + M4 - M5 - M7 = 
        # (A00*B00 + A00*B11 + A11*B00 + A11*B11) 
        # + (A11*B01 - A11*B00)
        # - (A00*B11 + A01*B11)
        # - (A01*B10 + A01*B11 - A10*B10 - A10*B11)
        # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B00 - A00*B11 - A01*B11 - A01*B10 - A01*B11 + A10*B10 + A10*B11
        # = A00*B00 + A11*B11 + A11*B01 - A01*B11 - A01*B10 + A10*B10 + A10*B11
        # That's not C00 = A00*B00 + A01*B10.
        
        # I'm clearly making errors. Let me just use a known correct Strassen decomposition.
        
        # The correct Strassen decomposition for the 2x2x2 tensor:
        # We need T[i,j,k] = 1 for all i,j,k in {0,1}
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # Known rank-7 decomposition (from literature):
        # Using the standard Strassen, the tensor entries are:
        
        # Let me just use a verified decomposition. I'll construct it from the bilinear algorithm.
        
        # The 7 bilinear forms are:
        # f1 = (a00+a11)(b00+b11)
        # f2 = (a10+a11)b00
        # f3 = a00(b10-b11)
        # f4 = a11(b01-b00)
        # f5 = (a00+a01)b11
        # f6 = (a10-a00)(b00+b01)
        # f7 = (a01-a10)(b10+b11)
        
        # And the output:
        # c00 = f1 + f4 - f5 - f7
        # c01 = f3 + f5
        # c10 = f2 + f4
        # c11 = f1 - f2 + f3 + f7
        
        # Let me verify c01 = f3 + f5:
        # f3 + f5 = a00(b10-b11) + (a00+a01)b11 = a00*b10 - a00*b11 + a00*b11 + a01*b11 = a00*b10 + a01*b11
        # c01 = a00*b01 + a01*b11
        # These are equal only if b10 = b01, which is not true in general.
        
        # So my Strassen formulas are wrong. Let me look this up properly.
        
        # Correct Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A10)(B10+B11)
        
        # C00 = M1 - M5 + M4 - M7
        # C01 = M5 + M3
        # C10 = M2 + M4
        # C11 = M1 - M2 + M7 + M3
        
        # Verify C00 = M1 - M5 + M4 - M7:
        # M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # M5 = A00*B11 + A01*B11
        # M4 = A11*B01 - A11*B00
        # M7 = A01*B10 + A01*B11 - A10*B10 - A10*B11
        
        # M1 - M5 + M4 - M7 =
        # A00*B00 + A00*B11 + A11*B00 + A11*B11
        # - A00*B11 - A01*B11
        # + A11*B01 - A11*B00
        # - A01*B10 - A01*B11 + A10*B10 + A10*B11
        # = A00*B00 + A11*B11 + A11*B01 - A01*B11 - A01*B10 + A10*B10 + A10*B11
        
        # C00 should be A00*B00 + A01*B10. Still not matching.
        
        # I think the issue is that I'm confusing the standard Strassen. Let me just use a different approach.
        
        # Actually, let me just verify with a simple case. For the tensor T[i,j,k]=1, I can use:
        # The naive rank-8 decomposition, or I can use a known rank-7.
        
        # Let me just use the naive rank-8 for now and focus on getting the overall structure right.
        # I can optimize the 2x2x2 case later.
        
        U_naive = [[0]*8 for _ in range(4)]
        V_naive = [[0]*8 for _ in range(4)]
        W_naive = [[0]*8 for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_naive[i*2+j][col] = 1
                    V_naive[j*2+k][col] = 1
                    W_naive[k*2+i][col] = 1
                    col += 1
        return U_naive, V_naive, W_naive
    
    # For now, use naive decomposition for everything
    # This gives rank = 3*4*8 = 96
    # We need to do better.
    
    # Let me implement the recursive decomposition properly.
    
    def decompose(n, m, p):
        """Return (U, V, W) for n x m x p matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        """
        if time.time() > DEADLINE:
            return None
        
        # Base cases
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        # Try to split dimensions
        # Prefer splitting the largest dimension
        dims = [(n, 'n'), (m, 'm'), (p, 'p')]
        dims.sort(key=lambda x: -x[0])
        
        for dim_val, dim_name in dims:
            if dim_val >= 2:
                half1 = dim_val // 2
                half2 = dim_val - half1
                
                if dim_name == 'n':
                    # Split n into half1 + half2
                    U1, V1, W1 = decompose(half1, m, p)
                    U2, V2, W2 = decompose(half2, m, p)
                    if U1 is None or U2 is None:
                        continue
                    rank1 = len(U1[0]) if U1 else 0
                    rank2 = len(U2[0]) if U2 else 0
                    total_rank = rank1 + rank2
                    
                    U = [[0]*total_rank for _ in range(n*m)]
                    V = [[0]*total_rank for _ in range(m*p)]
                    W = [[0]*total_rank for _ in range(p*n)]
                    
                    # Fill in U
                    for r in range(rank1):
                        for i in range(half1):
                            for j in range(m):
                                U[i*m+j][r] = U1[i*m+j][r]
                    for r in range(rank2):
                        for i in range(half2):
                            for j in range(m):
                                U[(half1+i)*m+j][rank1+r] = U2[i*m+j][r]
                    
                    # Fill in V
                    for r in range(rank1):
                        for j in range(m):
                            for k in range(p):
                                V[j*p+k][r] = V1[j*p+k][r]
                    for r in range(rank2):
                        for j in range(m):
                            for k in range(p):
                                V[j*p+k][rank1+r] = V2[j*p+k][r]
                    
                    # Fill in W
                    for r in range(rank1):
                        for k in range(p):
                            for i in range(half1):
                                W[k*n+i][r] = W1[k*half1+i][r]
                    for r in range(rank2):
                        for k in range(p):
                            for i in range(half2):
                                W[k*n+half1+i][rank1+r] = W2[k*half2+i][r]
                    
                    return U, V, W
                
                elif dim_name == 'm':
                    # Split m into half1 + half2
                    U1, V1, W1 = decompose(n, half1, p)
                    U2, V2, W2 = decompose(n, half2, p)
                    if U1 is None or U2 is None:
                        continue
                    rank1 = len(U1[0]) if U1 else 0
                    rank2 = len(U2[0]) if U2 else 0
                    total_rank = rank1 + rank2
                    
                    U = [[0]*total_rank for _ in range(n*m)]
                    V = [[0]*total_rank for _ in range(m*p)]
                    W = [[0]*total_rank for _ in range(p*n)]
                    
                    # Fill in U
                    for r in range(rank1):
                        for i in range(n):
                            for j in range(half1):
                                U[i*m+j][r] = U1[i*half1+j][r]
                    for r in range(rank2):
                        for i in range(n):
                            for j in range(half2):
                                U[i*m+half1+j][rank1+r] = U2[i*half2+j][r]
                    
                    # Fill in V
                    for r in range(rank1):
                        for j in range(half1):
                            for k in range(p):
                                V[j*p+k][r] = V1[j*p+k][r]
                    for r in range(rank2):
                        for j in range(half2):
                            for k in range(p):
                                V[(half1+j)*p+k][rank1+r] = V2[j*p+k][r]
                    
                    # Fill in W
                    for r in range(rank1):
                        for k in range(p):
                            for i in range(n):
                                W[k*n+i][r] = W1[k*n+i][r]
                    for r in range(rank2):
                        for k in range(p):
                            for i in range(n):
                                W[k*n+i][rank1+r] = W2[k*n+i][r]
                    
                    return U, V, W
                
                elif dim_name == 'p':
                    # Split p into half1 + half2
                    U1, V1, W1 = decompose(n, m, half1)
                    U2, V2, W2 = decompose(n, m, half2)
                    if U1 is None or U2 is None:
                        continue
                    rank1 = len(U1[0]) if U1 else 0
                    rank2 = len(U2[0]) if U2 else 0
                    total_rank = rank1 + rank2
                    
                    U = [[0]*total_rank for _ in range(n*m)]
                    V = [[0]*total_rank for _ in range(m*p)]
                    W = [[0]*total_rank for _ in range(p*n)]
                    
                    # Fill in U
                    for r in range(rank1):
                        for i in range(n):
                            for j in range(m):
                                U[i*m+j][r] = U1[i*m+j][r]
                    for r in range(rank2):
                        for i in range(n):
                            for j in range(m):
                                U[i*m+j][rank1+r] = U2[i*m+j][r]
                    
                    # Fill in V
                    for r in range(rank1):
                        for j in range(m):
                            for k in range(half1):
                                V[j*p+k][r] = V1[j*half1+k][r]
                    for r in range(rank2):
                        for j in range(m):
                            for k in range(half2):
                                V[j*p+half1+k][rank1+r] = V2[j*half2+k][r]
                    
                    # Fill in W
                    for r in range(rank1):
                        for k in range(half1):
                            for i in range(n):
                                W[k*n+i][r] = W1[k*n+i][r]
                    for r in range(rank2):
                        for k in range(half2):
                            for i in range(n):
                                W[(half1+k)*n+i][rank1+r] = W2[k*n+i][r]
                    
                    return U, V, W
        
        # Fallback: naive decomposition
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    result = decompose(n, m, p)
    if result is None:
        # Fallback to naive
        naive_rank = n * m * p
        U = [[0]*naive_rank for _ in range(n*m)]
        V = [[0]*naive_rank for _ in range(m*p)]
        W = [[0]*naive_rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    U, V, W = result
    return {"U": U, "V": V, "W": W}
