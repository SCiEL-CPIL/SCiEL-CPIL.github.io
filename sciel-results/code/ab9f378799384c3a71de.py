import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    R = 11
    
    # We need to find U (n*m x R), V (m*p x R), W (p*n x R)
    # such that for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try a structured approach
    # Start with a known rank-11 decomposition if we can find one
    
    # Approach: Use symmetry and try to find a solution
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
    
    # Let's try to construct this using a mix of terms
    # Think of it as: for each j, we have a 2x3 subproblem
    
    # Actually, let me try a direct search with small values
    
    # Candidate values: -1, -0.5, 0, 0.5, 1
    # But that's too many combinations. Let me be smarter.
    
    # Try a structured decomposition:
    # Use 2 terms for the "main" diagonal and 9 for corrections
    
    # Let me try:
    # Term 0: u=[1,0,0,1], v=[1,0,0,1,0,0], w=[1,0,0,1,0,0] -> contributes to (0,0,0) and (1,1,1)
    # Wait, w has shape (p*n, R) = (3*2, R) = (6, R), indexed by k*n+i
    
    # Let me be very explicit:
    # U indices: i*m+j, so (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # V indices: j*p+k, so (0,0)->0, (0,1)->1, (0,2)->2, (1,0)->3, (1,1)->4, (1,2)->5
    # W indices: k*n+i, so (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3, (2,0)->4, (2,1)->5
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let me try a rank-11 solution:
    # Terms 0-7: Handle the 2x2x2 part (k=0,1) with Strassen-like approach
    # Terms 8-10: Handle k=2
    
    # For k=0,1 (2x2x2 subproblem), use 7 terms (Strassen)
    # For k=2, we need 4 more terms for the 2x2 part, but we can share some
    
    # Actually, for k=2:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # This is a 2x2x1 multiplication, which needs rank 4 in general
    # But we can share with the k=0,1 terms
    
    # Let me try:
    # 7 terms for k in {0,1} using Strassen
    # 4 terms for k=2, but maybe we can do it in 3?
    
    # Total: 7 + 4 = 11
    
    # Strassen for 2x2x2 (k=0,1):
    # M1 = (A00+A11)(B00+B11) -> u=[1,0,1,1], v=[1,0,1,1,0,0], w=[1,0,1,0,0,0]
    # Wait, I need to be careful with the W indexing.
    
    # Let me define the 11 terms:
    
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # A part: A[0,0] and A[1,1] -> u[0]=1, u[3]=1
    # B part: B[0,0] and B[1,1] -> v[0]=1, v[4]=1
    # C part: contributes to C[0,0] and C[1,1] -> w[0]=1 (k=0,i=0), w[3]=1 (k=1,i=1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    # A part: A[1,0] and A[1,1] -> u[2]=1, u[3]=1
    # B part: B[0,0] -> v[0]=1
    # C part: contributes to C[1,0] -> w[0]=1 (k=0,i=1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = 1
    
    # Term 2: M3 = A00(B10-B11)
    # A part: A[0,0] -> u[0]=1
    # B part: B[1,0] and B[1,1] -> v[3]=1, v[4]=-1
    # C part: contributes to C[0,1] -> w[3]=1 (k=1,i=0)
    U[0][2] = 1
    V[3][2] = 1; V[4][2] = -1
    W[3][2] = 1
    
    # Term 3: M4 = A11(B01-B00)
    # A part: A[1,1] -> u[3]=1
    # B part: B[0,1] and B[0,0] -> v[1]=1, v[0]=-1
    # C part: contributes to C[1,1] -> w[3]=1 (k=1,i=1)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[3][3] = 1
    
    # Term 4: M5 = (A00+A10)B11
    # A part: A[0,0] and A[1,0] -> u[0]=1, u[2]=1
    # B part: B[1,1] -> v[4]=1
    # C part: contributes to C[0,1] -> w[3]=1 (k=1,i=0)
    U[0][4] = 1; U[2][4] = 1
    V[4][4] = 1
    W[3][4] = 1
    
    # Term 5: M6 = (A01-A00)(B00+B10)
    # A part: A[0,1] and A[0,0] -> u[1]=1, u[0]=-1
    # B part: B[0,0] and B[1,0] -> v[0]=1, v[3]=1
    # C part: contributes to C[0,0] -> w[0]=1 (k=0,i=0)
    U[1][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[3][5] = 1
    W[0][5] = 1
    
    # Term 6: M7 = (A10-A11)(B01+B11)
    # A part: A[1,0] and A[1,1] -> u[2]=1, u[3]=-1
    # B part: B[0,1] and B[1,1] -> v[1]=1, v[4]=1
    # C part: contributes to C[1,1] -> w[3]=1 (k=1,i=1)
    U[2][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[4][6] = 1
    W[3][6] = 1
    
    # Now for k=2, we need:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Term 7: A[0,0]*B[0,2]
    U[0][7] = 1
    V[2][7] = 1  # j=0, k=2
    W[4][7] = 1  # k=2, i=0
    
    # Term 8: A[0,1]*B[1,2]
    U[1][8] = 1
    V[5][8] = 1  # j=1, k=2
    W[4][8] = 1  # k=2, i=0
    
    # Term 9: A[1,0]*B[0,2]
    U[2][9] = 1
    V[2][9] = 1  # j=0, k=2
    W[5][9] = 1  # k=2, i=1
    
    # Term 10: A[1,1]*B[1,2]
    U[3][10] = 1
    V[5][10] = 1  # j=1, k=2
    W[5][10] = 1  # k=2, i=1
    
    # Verify the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check all 12 entries
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(R):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    valid = False
                    print(f"Error: T[{i},{j},{k}] = {total}, expected 1")
    
    if not valid:
        # Fall back to rank-12
        R = 12
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
    
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
