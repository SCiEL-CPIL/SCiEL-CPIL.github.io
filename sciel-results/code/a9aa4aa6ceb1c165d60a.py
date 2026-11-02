import time
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Row-column block embedding: partition 4x4x6 into four disjoint 2x2x3 subproblems
    # Split rows into {0,1} and {2,3}, columns of B into {0,1} and {2,3}, 
    # and output columns into {0,1}, {2,3}, {4,5}
    
    # For each combination of row-block and col-block, we have a 2x2x3 subproblem
    # We use a known rank-5 decomposition for 2x2x3
    
    # First, let's verify the 2x2x3 rank-5 decomposition
    # A is 2x2: indices (i,j) -> i*2+j for i,j in {0,1}
    # B is 2x3: indices (j,k) -> j*3+k for j in {0,1}, k in {0,1,2}
    # C is 3x2: indices (k,i) -> k*2+i for k in {0,1,2}, i in {0,1}
    
    # The multiplication: C[k,i] = sum_j A[i,j] * B[j,k]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,0] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[2,0] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[2,1] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Rank-5 decomposition for 2x2x3:
    # Term 1: A[0,0] * B[0,:] -> contributes to C[0,0], C[1,0], C[2,0]
    # Term 2: A[0,1] * B[1,:] -> contributes to C[0,1], C[1,1], C[2,1]
    # Term 3: A[1,0] * B[0,:] -> contributes to C[0,0], C[1,0], C[2,0]
    # Term 4: A[1,1] * B[1,:] -> contributes to C[0,1], C[1,1], C[2,1]
    # Term 5: (A[0,0]+A[1,0]) * (B[0,0]+B[1,0]) - A[0,0]*B[0,0] - A[1,0]*B[1,0]
    #         This gives cross terms that help combine
    
    # Actually, let's use a simpler approach: just use schoolbook for 2x2x3 (rank 6)
    # and see if we can do better. For now, rank 6 per subproblem gives 4*6=24.
    
    # But the directive says rank 5 is known. Let me construct it properly.
    
    # 2x2x3 rank-5 decomposition:
    # M1 = A[0,0] * (B[0,0] + B[0,1] + B[0,2])
    # M2 = A[0,1] * (B[1,0] + B[1,1] + B[1,2])
    # M3 = A[1,0] * (B[0,0] + B[0,1] + B[0,2])
    # M4 = A[1,1] * (B[1,0] + B[1,1] + B[1,2])
    # M5 = (A[0,0]+A[0,1]+A[1,0]+A[1,1]) * (B[0,0]+B[0,1]+B[0,2]+B[1,0]+B[1,1]+B[1,2])
    #      - M1 - M2 - M3 - M4
    
    # This doesn't work directly. Let me use a different known decomposition.
    
    # Actually, for 2x2x3, the optimal rank is 6 (schoolbook). Let me verify.
    # The tensor has 2*2*3 = 12 entries, but due to bilinear structure, 
    # the rank is at most 6 (one per (i,j,k) triple where j is the shared index).
    
    # Wait, schoolbook for 2x2x3: for each i in {0,1}, j in {0,1}, k in {0,1,2}:
    # C[k,i] += A[i,j] * B[j,k]
    # That's 2*2*3 = 12 terms. But we can group by j:
    # For j=0: A[:,0] * B[0,:] gives 2*3 = 6 products
    # For j=1: A[:,1] * B[1,:] gives 2*3 = 6 products
    # Total: 12 terms in naive schoolbook.
    
    # But we can do better. The rank of 2x2x3 tensor is known to be 6.
    # Let me construct it:
    
    # Term 1: U=[1,0,0,0], V=[1,0,0,0,0,0], W=[1,0,1,0,1,0]
    #   A[0,0] * (B[0,0]+B[0,1]+B[0,2]) -> C[0,0]+=1, C[1,0]+=1, C[2,0]+=1
    # Term 2: U=[0,1,0,0], V=[0,0,1,0,0,0], W=[0,1,0,1,0,1]
    #   A[0,1] * (B[1,0]+B[1,1]+B[1,2]) -> C[0,1]+=1, C[1,1]+=1, C[2,1]+=1
    # Term 3: U=[0,0,1,0], V=[1,0,0,0,0,0], W=[1,0,1,0,1,0]
    #   A[1,0] * (B[0,0]+B[0,1]+B[0,2]) -> C[0,0]+=1, C[1,0]+=1, C[2,0]+=1
    # Term 4: U=[0,0,0,1], V=[0,0,1,0,0,0], W=[0,1,0,1,0,1]
    #   A[1,1] * (B[1,0]+B[1,1]+B[1,2]) -> C[0,1]+=1, C[1,1]+=1, C[2,1]+=1
    
    # Wait, this gives:
    # C[0,0] = A[0,0]*(B[0,0]+B[0,1]+B[0,2]) + A[1,0]*(B[0,0]+B[0,1]+B[0,2])
    #        = (A[0,0]+A[1,0])*(B[0,0]+B[0,1]+B[0,2])
    # But we need C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    
    # This is wrong. Let me reconsider.
    
    # Correct schoolbook: C[k,i] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # So:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,0] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[2,0] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[2,1] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Schoolbook uses 6 terms (one per (j,k) pair, with A[i,j] and B[j,k]):
    # Actually, schoolbook is: for each j in {0,1}, k in {0,1,2}:
    #   C[k,:] += A[:,j] * B[j,k]
    # That's 2*3 = 6 terms, each being an outer product of a 2-vector and a scalar.
    
    # So rank 6 is the schoolbook rank for 2x2x3.
    
    # The directive claims rank 5. Let me try to find it.
    
    # One known rank-5 decomposition for 2x2x3:
    # M1 = A[0,0] * B[0,0]
    # M2 = A[0,1] * B[1,0]
    # M3 = A[1,0] * B[0,1]
    # M4 = A[1,1] * B[1,1]
    # M5 = (A[0,0]+A[0,1]+A[1,0]+A[1,1]) * (B[0,2]+B[1,2])
    #      - A[0,0]*B[0,2] - A[0,1]*B[1,2] - A[1,0]*B[0,2] - A[1,1]*B[1,2]
    #      = (A[0,0]+A[0,1]+A[1,0]+A[1,1])*(B[0,2]+B[1,2]) - M1' - M2' - M3' - M4'
    #      where M1'=A[0,0]*B[0,2], etc.
    # This doesn't work because it introduces extra terms.
    
    # Let me just use rank 6 for now, giving total rank 4*6 = 24.
    # This is already better than 96.
    
    # Build the full 4x4x6 decomposition using 4 independent 2x2x3 subproblems
    
    rank = 24  # 4 subproblems * 6 terms each
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 16 x 24
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 24 x 24
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 24 x 24
    
    # Subproblem (bi, cj) where bi in {0,1} (row block of A/C), cj in {0,1,2} (col block of B/C)
    # Wait, we need to be more careful.
    
    # A is 4x4: rows 0-3, cols 0-3
    # B is 4x6: rows 0-3, cols 0-5
    # C = A*B is 4x6: rows 0-3, cols 0-5
    
    # C[k,i] for k in 0..5, i in 0..3 (using trace convention W[k*n+i])
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Let's partition:
    # A rows: {0,1} and {2,3}
    # A cols: {0,1} and {2,3}
    # B rows: {0,1} and {2,3}
    # B cols: {0,1}, {2,3}, {4,5}
    # C rows: {0,1}, {2,3}, {4,5}
    # C cols: {0,1} and {2,3}
    
    # Subproblem (ri, cj) where ri in {0,1} (row block), cj in {0,1,2} (col block)
    # For subproblem (ri, cj):
    #   A block: rows ri*2..ri*2+1, cols 0..3 (full A cols, since we're not splitting A cols)
    #   Wait, this doesn't decompose cleanly.
    
    # Let me reconsider. The multiplication C = A*B where A is 4x4, B is 4x6.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # If we split i into {0,1} and {2,3}, and k into {0,1}, {2,3}, {4,5},
    # then for a fixed (i_block, k_block), C[i,k] = sum_j A[i,j] * B[j,k]
    # where j ranges over all 4 columns.
    
    # This doesn't decompose into independent subproblems because j is shared.
    
    # Alternative: split j as well. Let's split:
    # i: {0,1}, {2,3}
    # j: {0,1}, {2,3}
    # k: {0,1}, {2,3}, {4,5}
    
    # Then C[i,k] = sum_{j in {0,1}} A[i,j] * B[j,k] + sum_{j in {2,3}} A[i,j] * B[j,k]
    
    # For each combination of (i_block, j_block, k_block), we have a 2x2x? subproblem.
    # But k_block has size 2 or 3.
    
    # This is getting complex. Let me just use schoolbook with rank 96 for now,
    # but structure it cleanly.
    
    # Actually, let me try a different approach: use the fact that we can write
    # the multiplication as a sum of rank-1 tensors.
    
    # Schoolbook: rank = n*m*p = 4*4*6 = 96
    # Term (i,j,k): U has 1 at (i,j), V has 1 at (j,k), W has 1 at (k,i)
    # in the appropriate index conventions.
    
    # U index: i*m + j = i*4 + j
    # V index: j*p + k = j*6 + k
    # W index: k*n + i = k*4 + i
    
    U = [[Fraction(0)] * (n*m*p) for _ in range(n*m)]
    V = [[Fraction(0)] * (n*m*p) for _ in range(m*p)]
    W = [[Fraction(0)] * (n*m*p) for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = Fraction(1)
                V[j*p + k][col] = Fraction(1)
                W[k*n + i][col] = Fraction(1)
                col += 1
    
    # Convert to numeric
    def to_numeric(mat):
        out = []
        for row in mat:
            new_row = []
            for x in row:
                if x == int(x):
                    new_row.append(int(x))
                else:
                    new_row.append(float(x))
            out.append(new_row)
        return out
    
    U_out = to_numeric(U)
    V_out = to_numeric(V)
    W_out = to_numeric(W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
