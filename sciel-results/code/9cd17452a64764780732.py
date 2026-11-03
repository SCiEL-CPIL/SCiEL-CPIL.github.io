import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    deadline = time.time() + 130  # safety margin
    
    def schoolbook(n, m, p):
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
        return U, V, W

    # Try to use Strassen decomposition for the 2x2x2 sub-blocks
    # The idea: partition the tensor into disjoint 2x2x2 blocks where possible
    # and use Strassen (rank 7 instead of 8) for each
    
    # For M_{3,5,7}:
    # A is 3x5, B is 5x7, C is 3x7
    # We can try to use Strassen on a 2x2x2 sub-block
    # e.g., A rows {0,1}, A cols {0,1}, B rows {0,1}, B cols {0,1}
    # This covers C entries C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Strassen for 2x2x2:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C[0,0] = a*e + b*g, C[0,1] = a*f + b*h
    # C[1,0] = c*e + d*g, C[1,1] = c*f + d*h
    # 
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-c)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    #
    # C[0,0] = M1 - M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 - M6
    
    # Let's build a hybrid: schoolbook for most terms, but use Strassen
    # for one 2x2x2 block to reduce rank by 1
    
    # Start with schoolbook
    U, V, W = schoolbook(n, m, p)
    rank = len(U[0])
    
    # Identify the schoolbook terms for the 2x2x2 block
    # i in {0,1}, j in {0,1}, k in {0,1}
    # col = i*m*p + j*p + k
    # i=0,j=0,k=0: col=0
    # i=0,j=0,k=1: col=1
    # i=0,j=1,k=0: col=2
    # i=0,j=1,k=1: col=3
    # i=1,j=0,k=0: col=4
    # i=1,j=0,k=1: col=5
    # i=1,j=1,k=0: col=6
    # i=1,j=1,k=1: col=7
    
    # Zero out these 8 columns
    for col in range(8):
        for row in range(len(U)):
            U[row][col] = 0
        for row in range(len(V)):
            V[row][col] = 0
        for row in range(len(W)):
            W[row][col] = 0
    
    # Add 7 new columns for Strassen terms
    # M1 = (a+d)(e+h): U[A(0,0)+A(1,1)], V[B(0,0)+B(1,1)], W[C(0,0)+C(0,1)+C(1,0)-C(1,1)]
    # Wait, let me be more careful.
    
    # In the tensor decomposition, each rank-1 term is U[*,r] * V[*,r] * W[*,r]
    # For schoolbook term (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    
    # For Strassen M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]):
    # U has 1 at positions 0*m+0=0 and 1*m+1=6 (i=0,j=0 and i=1,j=1)
    # V has 1 at positions 0*p+0=0 and 1*p+1=6 (j=0,k=0 and j=1,k=1)
    # W: M1 contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    # C[0,0]: k*n+i = 0*3+0 = 0
    # C[0,1]: k*n+i = 1*3+0 = 3
    # C[1,0]: k*n+i = 0*3+1 = 1
    # C[1,1]: k*n+i = 1*3+1 = 4
    # From the formulas:
    # C[0,0] = M1 - M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 - M6
    # So M1 contributes +1 to C[0,0], +1 to C[1,1], -1 to... wait
    # Actually M1 appears in C[0,0] with +1 and C[1,1] with +1
    # So W for M1: +1 at W[0] (C[0,0]), +1 at W[4] (C[1,1])
    
    # Let me redefine more carefully.
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # This contributes to:
    # C[0,0]: +1 (from M1 in C[0,0] = M1 - M4 - M5 + M7)
    # C[1,1]: +1 (from M1 in C[1,1] = M1 - M2 + M3 - M6)
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # C[1,0]: +1 (from M2 in C[1,0] = M2 + M4)
    
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # C[0,1]: +1 (from M3 in C[0,1] = M3 + M5)
    # C[1,1]: +1 (from M3 in C[1,1] = M1 - M2 + M3 - M6)
    
    # M4 = A[1,1] * (B[1,0]-A[1,0])... wait, M4 = d*(g-c) = A[1,1]*(B[1,0]-A[1,0])? 
    # No, M4 = d*(g-c) where c=A[1,0], d=A[1,1], g=B[1,0]
    # So M4 = A[1,1] * (B[1,0] - A[1,0])? That doesn't make sense dimensionally.
    
    # Let me re-derive Strassen properly.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 - M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 - M6
    
    # So:
    # M1: U=[a+d]=[1,0,0,1] (a=A[0,0], d=A[1,1]), V=[e+h]=[1,0,0,1] (e=B[0,0], h=B[1,1])
    #   W: C[0,0]+1, C[1,1]+1 => W[0]+=1, W[4]+=1
    #
    # M2: U=[c+d]=[0,0,1,1] (c=A[1,0], d=A[1,1]), V=[e]=[1,0,0,0] (e=B[0,0])
    #   W: C[1,0]+1 => W[1]+=1
    #
    # M3: U=[a]=[1,0,0,0] (a=A[0,0]), V=[f-h]=[0,1,0,-1] (f=B[0,1], h=B[1,1])
    #   W: C[0,1]+1, C[1,1]+1 => W[3]+=1, W[4]+=1
    #
    # M4: U=[d]=[0,0,0,1] (d=A[1,1]), V=[g-c]=[0,0,1,0]... wait g=B[1,0], c is not in B
    #   M4 = d*(g-c) where c=A[1,0], g=B[1,0]. This is problematic because it mixes A and B.
    
    # I see the issue - Strassen's M4 is d*(g-c) but c is from A, not B. 
    # This means M4 is not a simple rank-1 term in the tensor decomposition sense.
    # Actually, in the tensor decomposition, each term must be of the form
    # sum_{i,j,k} u_{i,j} v_{j,k} w_{k,i}
    # So each rank-1 term has separate U, V, W vectors.
    
    # Let me reconsider. The Strassen algorithm uses 7 multiplications, but
    # the terms are sums/differences of products, which can be represented
    # as linear combinations of rank-1 tensor terms.
    
    # Actually, for the tensor decomposition, I need to express the entire
    # multiplication as a sum of rank-1 terms. Strassen shows that 2x2x2
    # matrix multiplication has rank at most 7, meaning it can be written
    # as a sum of 7 rank-1 terms.
    
    # The 7 rank-1 terms for 2x2x2 are:
    # Term 1: U1=(1,0,0,0), V1=(0,1,0,0), W1=(0,1,0,0)  [a*f -> C[0,1]]
    # No wait, let me look up the standard rank-7 decomposition.
    
    # For 2x2x2 matrix multiplication tensor, one rank-7 decomposition is:
    # T1: U=(1,0,0,0), V=(1,0,0,0), W=(1,0,0,0)  [a*e -> C[0,0]]
    # T2: U=(0,1,0,0), V=(0,0,0,1), W=(0,0,0,1)  [b*h -> C[0,1]]
    # T3: U=(0,0,1,0), V=(0,0,1,0), W=(0,1,0,0)  [c*g -> C[1,0]]
    # T4: U=(0,0,0,1), V=(0,1,0,0), W=(0,0,1,0)  [d*f -> C[1,1]]
    # ... this is getting complicated. Let me just use schoolbook for now
    # since finding a valid lower-rank decomposition is non-trivial.
    
    # Actually, let me try a simpler approach: just use schoolbook.
    # The schoolbook decomposition is valid and has rank 105.
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
