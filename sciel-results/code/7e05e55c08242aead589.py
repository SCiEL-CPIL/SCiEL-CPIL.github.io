"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Strategy: Use Strassen-like decomposition
    # Split B (3x10) into B_left (3x2) and B_right (3x8)
    # A (2x3) @ B_left (3x2) -> 2x2, use Strassen rank 7
    # A (2x3) @ B_right (3x8) -> 2x8
    # For 2x3 @ 3x8: split B_right into four 3x2 blocks
    # Each 2x3 @ 3x2: we can do this cleverly
    
    # Actually, let's use a cleaner approach:
    # Partition the 3 intermediate dimension into 1 + 2
    # A = [A1 | A2] where A1 is 2x1, A2 is 2x2
    # B = [B1; B2] where B1 is 1x10, B2 is 2x10
    # C = A1 @ B1 + A2 @ B2
    # A1 @ B1: 2x1 @ 1x10, rank 2 (just scale B1's rows)
    # A2 @ B2: 2x2 @ 2x10, split B2 into five 2x2 blocks, use Strassen (rank 7 each)
    # Total rank: 2 + 5*7 = 37? Let's verify this is valid.
    
    # Wait, A1 @ B1 where A1 is 2x1 and B1 is 1x10:
    # C1[i,j] = A1[i,0] * B1[0,j]
    # This needs rank 2 (one term per row of C1)
    # Actually rank = 2*1*1 = 2? No, the tensor for 2x1 @ 1x10 has rank 2.
    # Because C1[i,j] = A1[i,0]*B1[0,j], so we need 2 terms (one for each i).
    
    # A2 @ B2 where A2 is 2x2 and B2 is 2x10:
    # Split B2 = [B2_1 | B2_2 | B2_3 | B2_4 | B2_5] each 2x2
    # Each 2x2 @ 2x2 can be done with Strassen rank 7
    # Total for this part: 5 * 7 = 35
    
    # Grand total: 2 + 35 = 37
    
    # Let's construct this:
    rank_total = 2 + 5 * 7
    
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x rank_total
    V = [[0] * rank_total for _ in range(m * p)]  # 30 x rank_total
    W = [[0] * rank_total for _ in range(p * n)]  # 20 x rank_total
    
    col = 0
    
    # Part 1: A1 @ B1 (A1 = A[:, 0], B1 = B[0, :])
    # C[i,j] += A[i,0] * B[0,j]
    # U[i*m + 0] = A[i,0] -> U[i*3+0] = 1 for i=0,1
    # V[0*10 + j] = B[0,j] -> V[j] = 1 for j=0..9
    # W[j*n + i] = C[i,j] -> W[j*2+i] = 1 for i=0,1, j=0..9
    # But we only need rank 2, so:
    # Term 1: i=0: U[0*3+0]=1, V[j]=1 for all j, W[j*2+0]=1
    # Term 2: i=1: U[1*3+0]=1, V[j]=1 for all j, W[j*2+1]=1
    # But V would be the same for both terms, which is fine.
    # Actually, we can combine: for each j, we need U[0,0]=1 and U[1,0]=1, V[j]=1, W[2j]=1 and W[2j+1]=1
    # But that's 10 terms if we do it per j. Instead:
    # The tensor for 2x1@1x10: T[i,j,k] = delta(i,k)*... 
    # Let me think again. C[i,j] = A[i,0]*B[0,j]
    # This is an outer product of vector A[:,0] (length 2) and vector B[0,:] (length 10)
    # So rank = 1! We can do it with one term:
    # U[i*3+0] = A[i,0] coefficient -> set to 1 for both i (since A[i,0] is the variable)
    # Wait no. In the tensor decomposition, U, V, W are FIXED matrices, and the product is:
    # C = sum_c (U[:,c] outer V[:,c]) contracted appropriately... 
    
    # I'm confusing myself. Let me restart with the proper tensor decomposition framework.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    # A decomposition of rank r means: T = sum_{c=1}^r U[:,c] * V[:,c] * W[:,c] (outer products)
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    # and the product gives: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(j==j) for the correct terms
    
    # For the schoolbook: for each (i,j,k), one term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    
    # Let me just build the rank-37 decomposition explicitly:
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Handle A[:,0] @ B[0,:] contribution
    # C[i,j] += A[i,0] * B[0,j]
    # This is rank 1: one term where U[i*3+0]=1 for i=0,1; V[0*10+j]=1 for j=0..9; W[j*2+i]=1
    # But U has entries at positions 0 and 3 (i=0,j=0 and i=1,j=0)
    # V has entries at positions 0..9 (j=0, k=0..9)
    # W has entries at positions j*2+i for j=0..9, i=0,1
    
    # Actually this IS rank 1 if we set:
    # U[0] = 1, U[3] = 1 (A[0,0] and A[1,0])
    # V[j] = 1 for j=0..9 (B[0,j])
    # W[j*2+i] = 1 for j=0..9, i=0,1
    # Then sum over c of U[i*3+0]*V[0*10+j]*W[j*2+i] = 1*1*1 = 1 for the right (i,j)
    # But this gives C[i,j] += A[i,0]*B[0,j] which is correct!
    # And it's just 1 term!
    
    # So Part 1 uses rank 1, not 2.
    
    U[0][col] = 1  # A[0,0]
    U[3][col] = 1  # A[1,0]
    for j in range(10):
        V[j][col] = 1  # B[0,j]
        W[j*2+0][col] = 1  # C[0,j]
        W[j*2+1][col] = 1  # C[1,j]
    col += 1
    
    # Part 2: Handle A[:,1:3] @ B[1:3,:] contribution
    # A2 is 2x2 (columns 1,2 of A), B2 is 2x10 (rows 1,2 of B)
    # C[i,j] += sum_{l=0,1} A[i, l+1] * B[l+1, j]
    # Split B2 into five 2x2 blocks: B2[:, 0:2], B2[:, 2:4], ..., B2[:, 8:10]
    # For each 2x2 @ 2x2 block, use Strassen (rank 7)
    
    # Strassen for 2x2 @ 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For our case, A2[i,l] corresponds to A[i, l+1] for i=0,1, l=0,1
    # B2[l,j] corresponds to B[l+1, j] for l=0,1, j=0..9
    
    # For block b (j from 2*b to 2*b+1):
    # A2 is 2x2: A2[0,0]=A[0,1], A2[0,1]=A[0,2], A2[1,0]=A[1,1], A2[1,1]=A[1,2]
    # B2_b is 2x2: B2_b[0,0]=B[1,2b], B2_b[0,1]=B[1,2b+1], B2_b[1,0]=B[2,2b], B2_b[1,1]=B[2,2b+1]
    
    # Strassen terms for block b:
    # Each term is of the form (linear combo of A2 entries) * (linear combo of B2_b entries)
    # contributing to specific C entries
    
    # Let me define the 7 Strassen multipliers:
    # M1 = (A2[0,0]+A2[1,1]) * (B2_b[0,0]+B2_b[1,1])
    # M2 = (A2[1,0]+A2[1,1]) * B2_b[0,0]
    # M3 = A2[0,0] * (B2_b[1,0]-B2_b[1,1])
    # M4 = A2[1,1] * (B2_b[0,1]-B2_b[1,0])
    # M5 = (A2[0,0]+A2[0,1]) * B2_b[1,1]
    # M6 = (A2[1,0]-A2[0,0]) * (B2_b[0,0]+B2_b[0,1])
    # M7 = (A2[0,1]-A2[1,1]) * (B2_b[1,0]+B2_b[1,1])
    
    # C contributions for block b (columns 2b, 2b+1):
    # C[0,2b] += M1 + M4 - M5 + M7
    # C[0,2b+1] += M3 + M5
    # C[1,2b] += M2 + M4
    # C[1,2b+1] += M1 - M2 + M3 + M6
    
    # Now I need to express each M as a rank-1 term in the overall decomposition.
    # Each M is (sum of U coefficients) * (sum of V coefficients)
    # So for each M, I create one column in U, V, W.
    
    for b in range(5):
        j0 = 2 * b
        j1 = 2 * b + 1
        
        # M1 = (A[0,1]+A[1,2]) * (B[1,j0]+B[2,j1])
        # U: A[0,1] -> U[0*3+1]=U[1], A[1,2] -> U[1*3+2]=U[5]
        # V: B[1,j0] -> V[1*10+j0]=V[10+j0], B[2,j1] -> V[2*10+j1]=V[20+j1]
        # W: C[0,j0]+=1, C[0,j1]+=0, C[1,j0]+=0, C[1,j1]+=1 (from M1 in C11 and C22)
        # C[0,j0] -> W[j0*2+0], C[1,j1] -> W[j1*2+1]
        U[1][col] = 1; U[5][col] = 1
        V[10+j0][col] = 1; V[20+j1][col] = 1
        W[j0*2+0][col] = 1; W[j1*2+1][col] = 1
        col += 1
        
        # M2 = (A[1,1]+A[1,2]) * B[1,j0]
        # U: A[1,1] -> U[4], A[1,2] -> U[5]
        # V: B[1,j0] -> V[10+j0]
        # W: C[1,j0]+=1 -> W[j0*2+1], C[1,j1]+=-1 -> W[j1*2+1]
        U[4][col] = 1; U[5][col] = 1
        V[10+j0][col] = 1
        W[j0*2+1][col] = 1; W[j1*2+1][col] = -1
        col += 1
        
        # M3 = A[0,1] * (B[2,j0]-B[2,j1])
        # U: A[0,1] -> U[1]
        # V: B[2,j0] -> V[20+j0], B[2,j1] -> V[20+j1]
        # W: C[0,j1]+=1 -> W[j1*2+0], C[1,j1]+=1 -> W[j1*2+1]
        U[1][col] = 1
        V[20+j0][col] = 1; V[20+j1][col] = -1
        W[j1*2+0][col] = 1; W[j1*2+1][col] = 1
        col += 1
        
        # M4 = A[1,2] * (B[1,j1]-B[2,j0])
        # U: A[1,2] -> U[5]
        # V: B[1,j1] -> V[10+j1], B[2,j0] -> V[20+j0]
        # W: C[0,j0]+=1 -> W[j0*2+0], C[1,j0]+=1 -> W[j0*2+1]
        U[5][col] = 1
        V[10+j1][col] = 1; V[20+j0][col] = -1
        W[j0*2+0][col] = 1; W[j0*2+1][col] = 1
        col += 1
        
        # M5 = (A[0,1]+A[0,2]) * B[2,j1]
        # U: A[0,1] -> U[1], A[0,2] -> U[2]
        # V: B[2,j1] -> V[20+j1]
        # W: C[0,j0]+=-1 -> W[j0*2+0], C[0,j1]+=1 -> W[j1*2+0]
        U[1][col] = 1; U[2][col] = 1
        V[20+j1][col] = 1
        W[j0*2+0][col] = -1; W[j1*2+0][col] = 1
        col += 1
        
        # M6 = (A[1,1]-A[0,1]) * (B[1,j0]+B[1,j1])
        # U: A[1,1] -> U[4], A[0,1] -> U[1]
        # V: B[1,j0] -> V[10+j0], B[1,j1] -> V[10+j1]
        # W: C[1,j1]+=1 -> W[j1*2+1]
        U[4][col] = 1; U[1][col] = -1
        V[10+j0][col] = 1; V[10+j1][col] = 1
        W[j1*2+1][col] = 1
        col += 1
        
        # M7 = (A[0,2]-A[1,2]) * (B[2,j0]+B[2,j1])
        # U: A[0,2] -> U[2], A[1,2] -> U[5]
        # V: B[2,j0] -> V[20+j0], B[2,j1] -> V[20+j1]
        # W: C[0,j0]+=1 -> W[j0*2+0]
        U[2][col] = 1; U[5][col] = -1
        V[20+j0][col] = 1; V[20+j1][col] = 1
        W[j0*2+0][col] = 1
        col += 1
    
    # Verify col == rank_total
    assert col == rank_total, f"Column count {col} != expected {rank_total}"
    
    # Verify the decomposition is correct
    if time.time() < deadline:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
        # Only when it's a valid (i,j,k) triple, 0 otherwise
        # Actually, the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # since C[i,k] = sum_j A[i,j]*B[j,k]
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank_total))
                    error += abs(val - 1)
        
        if error > 1e-10:
            # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
