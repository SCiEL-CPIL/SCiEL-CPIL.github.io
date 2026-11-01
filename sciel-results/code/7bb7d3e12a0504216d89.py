import numpy as np
import time
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """
    Verify exact decomposition using Fraction arithmetic.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Indices:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    """
    Uf = [[Fraction(int(round(x*2))/2) for x in row] for row in U]
    Vf = [[Fraction(int(round(x*2))/2) for x in row] for row in V]
    Wf = [[Fraction(int(round(x*2))/2) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = 2, 2, 5
    
    # Strategy: Direct sum of two 2x2x2 Strassen blocks and one 2x2x1 schoolbook block.
    # Total Rank = 7 (Strassen) + 7 (Strassen) + 2 (Schoolbook) = 16.
    
    # The multiplication is C = A @ B where A is 2x2 and B is 2x5.
    # We can split B into B1 (2x2) and B2 (2x2) and B3 (2x1)? 
    # Actually, B is 2x5. We can view B as [B_0, B_1, B_2, B_3, B_4] where B_k is 2x1.
    # Or better, split the 5 columns into 2+2+1.
    # C = A @ [B_0, B_1, B_2, B_3, B_4] = [A@B_0, A@B_1, A@B_2, A@B_3, A@B_4]
    
    # Let's define the blocks:
    # Block 1: Multiplies A (2x2) by B_0 (2x2) -> C_0 (2x2). Use Strassen.
    # Block 2: Multiplies A (2x2) by B_1 (2x2) -> C_1 (2x2). Use Strassen.
    # Block 3: Multiplies A (2x2) by B_2 (2x1) -> C_2 (2x1). Use Schoolbook (rank 2).
    # Wait, 2+2+1 = 5 columns.
    # B_0 is cols 0,1. B_1 is cols 2,3. B_2 is col 4.
    
    # Indices for U, V, W:
    # U has rows indexed by (i, j) from A. i in 0..1, j in 0..1. Total 4 rows.
    # V has rows indexed by (j, k) from B. j in 0..1, k in 0..4. Total 10 rows.
    # W has rows indexed by (k, i) from C. k in 0..4, i in 0..1. Total 10 rows.
    
    # Rank 16.
    # Columns 0-6: Strassen for B_0 (cols 0,1)
    # Columns 7-13: Strassen for B_1 (cols 2,3)
    # Columns 14-15: Schoolbook for B_2 (col 4)
    
    U = np.zeros((n*m, 16))
    V = np.zeros((m*p, 16))
    W = np.zeros((p*n, 16))
    
    # Strassen 2x2 Decomposition:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's map the Strassen terms to the tensor indices.
    # For a 2x2x2 block multiplying A (2x2) and B (2x2) to get C (2x2):
    # U indices: i*m+j. Here local i,j in 0..1.
    # V indices: j*p+k. Here local j in 0..1, local k in 0..1 (within the 2x2 block).
    # W indices: k*n+i. Here local k in 0..1, local i in 0..1.
    
    # We need to be careful with global indices.
    # Global A index: (i, j) -> row i*m+j
    # Global B index: (j, k) -> row j*p+k
    # Global C index: (k, i) -> row k*n+i
    
    # Strassen Block 1: B cols 0,1.
    # Local B k indices: 0, 1.
    # Global B k indices: 0, 1.
    
    # Strassen Block 2: B cols 2,3.
    # Local B k indices: 0, 1.
    # Global B k indices: 2, 3.
    
    # Schoolbook Block 3: B col 4.
    # Global B k index: 4.
    
    def add_strassen_block(offset, k_start):
        """
        Adds a Strassen block for B columns k_start, k_start+1.
        offset: column offset in U, V, W.
        k_start: global k index for the first column of this 2x2 block.
        """
        # M1 = (A11 + A22)(B11 + B22)
        # A11 is (i=0, j=0). A22 is (i=1, j=1).
        # B11 is (j=0, k=k_start). B22 is (j=1, k=k_start+1).
        
        # Term 1: (A11 + A22) * (B11 + B22)
        # U[0, offset] = 1 (A11)
        # U[3, offset] = 1 (A22)
        # V[0*p + k_start, offset] = 1 (B11)
        # V[1*p + (k_start+1), offset] = 1 (B22)
        # W contribution to C11 (k=k_start, i=0): +1
        # W contribution to C22 (k=k_start+1, i=1): +1
        U[0, offset] += 1
        U[3, offset] += 1
        V[0*p + k_start, offset] += 1
        V[1*p + (k_start+1), offset] += 1
        W[k_start*n + 0, offset] += 1  # C11
        W[(k_start+1)*n + 1, offset] += 1 # C22
        
        # Term 2: (A21 + A22) * B11
        # A21 is (i=1, j=0). A22 is (i=1, j=1).
        # B11 is (j=0, k=k_start).
        U[2, offset+1] += 1 # A21
        U[3, offset+1] += 1 # A22
        V[0*p + k_start, offset+1] += 1 # B11
        # W contribution to C21 (k=k_start, i=1): +1
        W[k_start*n + 1, offset+1] += 1
        
        # Term 3: A11 * (B21 - B22)
        # A11 is (i=0, j=0).
        # B21 is (j=1, k=k_start). B22 is (j=1, k=k_start+1).
        U[0, offset+2] += 1
        V[1*p + k_start, offset+2] += 1
        V[1*p + (k_start+1), offset+2] -= 1
        # W contribution to C12 (k=k_start+1, i=0): +1
        # W contribution to C21 (k=k_start, i=1): +1
        W[(k_start+1)*n + 0, offset+2] += 1
        W[k_start*n + 1, offset+2] += 1
        
        # Term 4: A22 * (B12 - B11)
        # A22 is (i=1, j=1).
        # B12 is (j=0, k=k_start+1). B11 is (j=0, k=k_start).
        U[3, offset+3] += 1
        V[0*p + (k_start+1), offset+3] += 1
        V[0*p + k_start, offset+3] -= 1
        # W contribution to C11 (k=k_start, i=0): +1
        # W contribution to C21 (k=k_start, i=1): +1
        W[k_start*n + 0, offset+3] += 1
        W[k_start*n + 1, offset+3] += 1
        
        # Term 5: (A11 + A12) * B22
        # A11 is (i=0, j=0). A12 is (i=0, j=1).
        # B22 is (j=1, k=k_start+1).
        U[0, offset+4] += 1
        U[1, offset+4] += 1
        V[1*p + (k_start+1), offset+4] += 1
        # W contribution to C11 (k=k_start, i=0): -1
        # W contribution to C12 (k=k_start+1, i=0): +1
        W[k_start*n + 0, offset+4] -= 1
        W[(k_start+1)*n + 0, offset+4] += 1
        
        # Term 6: (A21 - A11) * (B11 + B12)
        # A21 is (i=1, j=0). A11 is (i=0, j=0).
        # B11 is (j=0, k=k_start). B12 is (j=0, k=k_start+1).
        U[2, offset+5] += 1
        U[0, offset+5] -= 1
        V[0*p + k_start, offset+5] += 1
        V[0*p + (k_start+1), offset+5] += 1
        # W contribution to C22 (k=k_start+1, i=1): +1
        W[(k_start+1)*n + 1, offset+5] += 1
        
        # Term 7: (A12 - A22) * (B21 + B22)
        # A12 is (i=0, j=1). A22 is (i=1, j=1).
        # B21 is (j=1, k=k_start). B22 is (j=1, k=k_start+1).
        U[1, offset+6] += 1
        U[3, offset+6] -= 1
        V[1*p + k_start, offset+6] += 1
        V[1*p + (k_start+1), offset+6] += 1
        # W contribution to C11 (k=k_start, i=0): +1
        # W contribution to C22 (k=k_start+1, i=1): +1
        W[k_start*n + 0, offset+6] += 1
        W[(k_start+1)*n + 1, offset+6] += 1

    # Add Strassen Block 1 for B cols 0,1
    add_strassen_block(0, 0)
    
    # Add Strassen Block 2 for B cols 2,3
    add_strassen_block(7, 2)
    
    # Add Schoolbook Block for B col 4
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Terms:
    # 1. A[0,0] * B[0,4] -> C[0,4]
    # 2. A[0,1] * B[1,4] -> C[0,4]
    # 3. A[1,0] * B[0,4] -> C[1,4]
    # 4. A[1,1] * B[1,4] -> C[1,4]
    
    # Rank 16 means we have 2 slots left? No, 7+7=14. 16-14=2 slots.
    # Wait, schoolbook for 2x2x1 has rank 2?
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
    # This is a tensor of size 2x2x1.
    # T[i,j,k] where k is fixed to 4.
    # T[0,0,4]=1, T[0,1,4]=1, T[1,0,4]=1, T[1,1,4]=1.
    # Can we decompose this in rank 2?
    # Term 1: A[0,0] B[0,4] C[0,4]
    # Term 2: A[1,1] B[1,4] C[1,4]
    # This misses A[0,1]B[1,4] and A[1,0]B[0,4].
    # Actually, 2x2x1 matrix multiplication is equivalent to two independent dot products? No.
    # C[0,4] depends on A[0,0], A[0,1].
    # C[1,4] depends on A[1,0], A[1,1].
    # The tensor is block diagonal in some sense?
    # Let's check rank of 2x2x1 matmul tensor.
    # It is the sum of 4 rank 1 tensors.
    # Is it reducible?
    # T = e_00 e_04 e_04^T + e_01 e_14 e_04^T + e_10 e_04 e_14^T + e_11 e_14 e_14^T
    # This looks like it has rank 4?
    # Wait, C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
    # Let's try rank 2.
    # M1: A[0,0]B[0,4] -> C[0,4]
    # M2: A[1,1]B[1,4] -> C[1,4]
    # Missing: A[0,1]B[1,4] -> C[0,4] and A[1,0]B[0,4] -> C[1,4].
    # If we use:
    # M1: (A[0,0]+A[0,1]) B[0,4] ... no B indices are mixed.
    # Actually, for 2x2x1, the rank is 4?
    # Let's check lower bound.
    # If we fix B, it's linear.
    # Let's just use rank 4 for the last part?
    # Total rank 7+7+4 = 18.
    # The inspiration says "optimized 2x3" allows rank 16 or 17.
    # Maybe I should split B into 2x2 and 2x3?
    # 2x2 Strassen (7) + 2x3 optimized?
    # Rank of 2x2x3 matmul?
    # C (2x3) = A (2x2) B (2x3).
    # This is equivalent to 3 independent 2x2x1 multiplications? No, A is shared.
    # C[:, 0] = A B[:, 0]
    # C[:, 1] = A B[:, 1]
    # C[:, 2] = A B[:, 2]
    # This is a tensor of size 2x2x3.
    # Rank?
    # If we use Strassen for the first 2 columns (rank 7), we still have the 3rd column.
    # C[:, 2] = A B[:, 2].
    # This is 2x2x1.
    # If 2x2x1 has rank 4, total is 7+4=11 for 2x2x3?
    # Then total 7+11 = 18.
    
    # Is there a better decomposition for 2x2x3?
    # Maybe rank 10?
    # If 2x2x3 has rank 10, then 7+10 = 17.
    # If 2x2x3 has rank 9, then 7+9 = 16.
    
    # Let's try to construct 2x2x3 with rank 9 or 10.
    # Or just stick to the 16 rank target from the prompt's strategy hint:
    # "two 2x2x2 Strassen blocks (rank 7 each) and a 2x2x1 schoolbook block (rank 2)"
    # The prompt claims 2x2x1 schoolbook block has rank 2.
    # Let's re-evaluate 2x2x1 rank.
    # T[i,j,k] with k fixed.
    # T[0,0]=1, T[0,1]=1, T[1,0]=1, T[1,1]=1.
    # This is the matrix J (all ones) as a tensor?
    # C[0] = A[0,0]B[0] + A[0,1]B[1]
    # C[1] = A[1,0]B[0] + A[1,1]B[1]
    # Let's try to find rank 2 decomposition.
    # U (2x2), V (2x2), W (2x2).
    # Sum_c U[:,c] V[:,c] W[:,c] = T.
    # T is 2x2 (flattened A, B) x 2 (flattened C).
    # Actually T is 2x2x1.
    # Let's try:
    # c=0: U=[1,0,0,1]^T (A00, A11), V=[1,0,1,0]^T (B0, B1), W=[1,1]^T (C0, C1)
    #   Term 0: A00 B0 C0 + A00 B0 C1 + A11 B1 C0 + A11 B1 C1
    #   This gives A00 B0 in C0 and C1. We only want it in C0.
    #   So W must be [1,0] for A00 B0 term.
    #   If W=[1,0], then C1 gets nothing from this term.
    #   We need C1 to get A10 B0 + A11 B1.
    #   So we need a term for A10 B0 -> C1.
    #   And a term for A11 B1 -> C1.
    #   And a term for A01 B1 -> C0.
    #   And a term for A00 B0 -> C0.
    #   These are 4 distinct "paths".
    #   Can we combine them?
    #   Term 1: A00 B0 -> C0.
    #   Term 2: A01 B1 -> C0.
    #   Term 3: A10 B0 -> C1.
    #   Term 4: A11 B1 -> C1.
    #   If we use:
    #   M1: (A00 + A01) B0 ... no B indices differ.
    #   M1: (A00 + A10) B0 ... A indices differ, B same.
    #   (A00 + A10) B0 -> C0 + C1?
    #   If W=[1,1], then (A00+A10)B0 contributes to C0 and C1.
    #   We want A00 B0 in C0 only.
    #   So we need to cancel it in C1?
    #   If we have another term with -A00 B0 in C1?
    #   This suggests rank might be higher.
    
    # However, the prompt explicitly suggests rank 16 is achievable with 7+7+2.
    # I will trust the prompt's structural hint and implement the 2x2x1 part with rank 2, 
    # but if it fails verification, I will fall back to rank 4 for that part (total 18) 
    # or try to find a valid rank 2 decomposition.
    
    # Let's try to force rank 2 for 2x2x1.
    # Maybe:
    # M1: (A00 + A01) * (B0 + B1) ...
    # Let's just implement the 18 rank version first to be safe, then try to optimize?
    # No, the goal is to beat 20. 18 is better than 20.
    # But 16 is better.
    
    # Let's check if 2x2x1 rank is actually 2.
    # Reference: "The rank of the matrix multiplication tensor <m,n,p> is ..."
    # For <2,2,1>, the rank is 4?
    # Wait, C[0] = A[0,0]B[0] + A[0,1]B[1]
    # C[1] = A[1,0]B[0] + A[1,1]B[1]
    # This is just two independent linear combinations.
    # The tensor is T = e_00 e_0 e_0 + e_01 e_1 e_0 + e_10 e_0 e_1 + e_11 e_1 e_1
    # (using local indices for A: 00,01,10,11; B: 0,1; C: 0,1)
    # This is a sum of 4 rank 1 tensors.
    # Can it be done in 3?
    # If rank is 4, total is 7+7+4 = 18.
    # 18 is still an improvement over 20.
    # Score 18 vs 20. Seed was 20. 20/18 = 1.11.
    
    # Let's implement the 18 rank version (7+7+4).
    # And then see if I can shave off 2 more.
    
    # Actually, let's look at the "2x2x1 schoolbook block (rank 2)" claim again.
    # Is it possible the prompt meant something else?
    # "2x2x1 schoolbook block"
    # Maybe it means multiplying a 2x2 matrix by a 2x1 matrix?
    # Yes.
    # If the rank is 4, then 18 is the safe bet.
    
    # Let's implement 18.
    
    U = np.zeros((n*m, 18))
    V = np.zeros((m*p, 18))
    W = np.zeros((p*n, 18))
    
    # Strassen Block 1 (cols 0,1)
    add_strassen_block(0, 0)
    
    # Strassen Block 2 (cols 2,3)
    add_strassen_block(7, 2)
    
    # Schoolbook Block 3 (col 4) - Rank 4
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
    
    # Term 14: A[0,0] B[0,4] -> C[0,4]
    U[0, 14] = 1
    V[0*p + 4, 14] = 1
    W[4*n + 0, 14] = 1
    
    # Term 15: A[0,1] B[1,4] -> C[0,4]
    U[1, 15] = 1
    V[1*p + 4, 15] = 1
    W[4*n + 0, 15] = 1
    
    # Term 16: A[1,0] B[0,4] -> C[1,4]
    U[2, 16] = 1
    V[0*p + 4, 16] = 1
    W[4*n + 1, 16] = 1
    
    # Term 17: A[1,1] B[1,4] -> C[1,4]
    U[3, 17] = 1
    V[1*p + 4, 17] = 1
    W[4*n + 1, 17] = 1
    
    # Verify
    if not _verify(U, V, W, n, m, p, 18):
        # Fallback to schoolbook 20 if something went wrong
        U = np.zeros((n*m, 20))
        V = np.zeros((m*p, 20))
        W = np.zeros((p*n, 20))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
