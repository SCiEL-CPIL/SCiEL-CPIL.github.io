import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def verify(U, V, W, n=2, m=3, p=3):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Check against delta_{jk}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                expected = 1 if j == k else 0
                if abs(s - expected) > 1e-12:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Try rank 9
    rank = 9
    # Use the structure: for each j, 3 terms
    # U[i*m+j, 3j+s], V[j*p+k, 3j+s], W[k*n+i, 3j+s]
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # For each j, we need to handle all k
    # Let's try: V[j,k,3j+s] = delta_{k,j} for all s (diagonal only)
    # Then we need: for k=j: sum_s U[i,j,3j+s]*1*W[i,j,3j+s] = 1
    # For k!=j: automatically 0
    
    # This only works if we can make the sum = 1 for each i
    # With 3 terms per j, and 2 i-values:
    # sum_s U[i,j,3j+s]*W[i,j,3j+s] = 1
    
    # Simple: U[0,j,3j+0]=1, W[0,j,3j+0]=1, others 0 → sum=1 for i=0
    # U[1,j,3j+1]=1, W[1,j,3j+1]=1, others 0 → sum=1 for i=1
    # But this only uses 2 of 3 terms per j, and s=2 is wasted
    
    # Actually this gives rank 6 effectively. But does it work?
    # C[i,k] = sum_j sum_r U[i,j,r]*V[j,k,r]*W[i,k,r]
    # For k=j: U[i,j,3j+0]*V[j,j,3j+0]*W[i,j,3j+0] = (i==0?1:0)*1*(i==0?1:0) = 1 if i=0
    #           U[i,j,3j+1]*V[j,j,3j+1]*W[i,j,3j+1] = (i==1?1:0)*1*(i==1?1:0) = 1 if i=1
    # So C[0,j] = A[0,j]*B[j,j], C[1,j] = A[1,j]*B[j,j]
    # This is WRONG - we need C[i,k] = sum_j A[i,j]*B[j,k], not just B[j,j]
    
    # The issue: V[j,k,r] = delta_{k,j} means we only pick up B[j,j], not B[j,k] for k!=j
    # We need V to extract B[j,k] for all k, not just k=j
    
    # So V[j,k,r] must be nonzero for k!=j too, but arranged so that
    # sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = 0 for j!=k
    
    # This is a system of equations. Let me try a different approach.
    
    # For 2x3x3, use the following known rank-9 decomposition:
    # Based on the idea that we can use 3 "columns" of B and 2 "rows" of A
    
    # Let me try: for each k (column of B, 3 values), and each pair of rows...
    # Actually, let me just do a numerical search.
    
    start_time = time.time()
    deadline = start_time + 100  # Leave margin for 150s limit
    
    # Try to find rank-9 solution via random initialization + verification
    # The structure: T[i,j,k] = delta_{jk}
    # We need sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = delta_{jk}
    
    # For rank 9, let's parameterize:
    # U is 6x9, V is 9x9, W is 6x9
    # Total 6*9 + 9*9 + 6*9 = 54+81+54 = 189 variables
    # Constraints: 2*3*3 = 18 equations (but many are trivially satisfied by structure)
    
    # Actually, 18 equations, 189 unknowns - very underdetermined!
    # We can fix many variables and solve for others.
    
    # Let's use the symmetry: the problem is symmetric in j,k (both range over 3)
    # and the i dimension has 2 values.
    
    # Try: set V[j,k,r] = delta_{r, j*p+k} (identity-like) for r < 9
    # Then V is 9x9 identity.
    # Then: sum_r U[i,j,r]*delta_{r,j*p+k}*W[i,k,r] = U[i,j,j*p+k]*W[i,k,j*p+k] = delta_{jk}
    # So we need: U[i,j,j*p+k]*W[i,k,j*p+k] = delta_{jk}
    # For j=k: U[i,j,j*p+j]*W[i,j,j*p+j] = 1
    # For j!=k: U[i,j,j*p+k]*W[i,k,j*p+k] = 0
    
    # With V = I, the product U[i,j,r]*W[i,k,r] must give delta_{jk} when r=j*p+k
    # This means: for each (i,j,k) with j!=k, either U[i,j,j*p+k]=0 or W[i,k,j*p+k]=0
    # And for each (i,j) with j=k: U[i,j,j*p+j]*W[i,j,j*p+j] = 1
    
    # Simple solution: U[i,j,j*p+j] = 1, W[i,j,j*p+j] = 1 for all i,j
    # U[i,j,j*p+k] = 0 for k!=j, W[i,k,j*p+k] = 0 for k!=j (already 0)
    # But W[i,k,j*p+k] for j!=k: we need U[i,j,j*p+k]*W[i,k,j*p+k] = 0
    # If U[i,j,j*p+k] = 0 for k!=j, then it's automatically 0.
    
    # So: U[i*m+j, j*p+k] = 1 if k==j else 0
    #     V[j*p+k, r] = 1 if r==j*p+k else 0 (identity)
    #     W[k*n+i, j*p+k] = 1 if k==j else 0... wait, W is indexed by (k,i)
    
    # W[k*n+i, r] where r = j*p+k. We need W[k*n+i, j*p+k] = 1 if j==k else 0.
    # So W[k*n+i, j*p+k] = 1 if j==k else 0.
    
    # Let's verify:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #        = sum_j U[i*m+j, j*p+k] * 1 * W[k*n+i, j*p+k]  (since V is identity)
    #        = sum_j (1 if k==j else 0) * (1 if j==k else 0)
    #        = sum_j delta_{jk} * delta_{jk}
    #        = sum_j delta_{jk}
    #        = 1 if there exists j with j=k, which is always true for k in 0..2
    #        = 1
    
    # Wait, that gives C[i,k] = 1 for all i,k, not C[i,k] = sum_j A[i,j]*B[j,k]
    
    # I'm confusing the tensor decomposition with the actual multiplication.
    # The tensor T[i,j,k] = delta_{jk} represents the structure of matrix multiplication.
    # The decomposition T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] means:
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_j sum_r U[i,j,r]*V[j,k,r]*W[i,k,r]*A[i,j]*B[j,k]... no.
    
    # Actually, the standard way: if T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r],
    # then C[i,k] = sum_j A[i,j]*B[j,k] = sum_j sum_r U[i,j,r]*V[j,k,r]*W[i,k,r]*A[i,j]*B[j,k]
    # That's not right either.
    
    # The correct interpretation: U, V, W are "extractors" and "combiners".
    # U[i,j,r] multiplies A[i,j], V[j,k,r] multiplies B[j,k], W[i,k,r] is the coefficient.
    # The rank-r decomposition means:
    # C[i,k] = sum_r W[i,k,r] * (sum_{i',j'} U[i',j',r] * A[i',j']) * (sum_{j',k'} V[j',k',r] * B[j',k'])
    # For this to equal sum_j A[i,j]*B[j,k], we need the tensor identity.
    
    # Let me just verify numerically with the schoolbook and try to find a lower rank.
    
    # For now, return schoolbook rank 18 as fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
