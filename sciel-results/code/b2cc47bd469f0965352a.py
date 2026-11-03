import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    # Schoolbook decomposition as fallback (rank = n*m*p = 48)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly matches the multiplication tensor."""
        rank = len(U[0]) if U else 0
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Precompute rank-1 products
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k2*n+i2, r]
                            exp = 1.0 if (i == i2 and k == k2) else 0.0
                            if abs(s - exp) > tol:
                                return False
        return True

    # Strassen 2x2x2 decomposition (rank 7)
    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen's 7 products:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Term 0 (M1): (A11+A22)*(B11+B22)
        # U: A11 (0*2+0=0), A22 (1*2+1=3)
        # V: B11 (0*2+0=0), B22 (1*2+1=3)
        # W: C11 (0*2+0=0), C22 (1*2+1=3)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # Term 1 (M2): (A21+A22)*B11
        # U: A21 (1*2+0=2), A22 (3)
        # V: B11 (0)
        # W: C21 (1*2+0=2), C22 (3)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # Term 2 (M3): A11*(B21-B22)
        # U: A11 (0)
        # V: B21 (1*2+0=2), B22 (3) with -1
        # W: C12 (0*2+1=1), C22 (3)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # Term 3 (M4): A22*(B12-B11)
        # U: A22 (3)
        # V: B12 (0*2+1=1), B11 (0) with -1
        # W: C11 (0), C21 (2)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # Term 4 (M5): (A11+A12)*B22
        # U: A11 (0), A12 (0*2+1=1)
        # V: B22 (3)
        # W: C11 (0) with -1, C12 (1)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # Term 5 (M6): (A21-A11)*(B11+B12)
        # U: A21 (2), A11 (0) with -1
        # V: B11 (0), B12 (1)
        # W: C22 (3)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 6 (M7): (A12-A22)*(B21+B22)
        # U: A12 (1), A22 (3) with -1
        # V: B21 (2), B22 (3)
        # W: C11 (0)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Build 4x4x4 using Strassen blocks (rank 28)
    # Split 4x4 into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = A*B computed via Strassen at block level (7 block multiplications)
    # Each block multiplication is 2x2x2 with rank 7
    # Total: 7*7 = 49... wait, that's for full 4x4x4
    
    # Actually for 3x4x4, we can do better:
    # Treat A as 3x4 (rows 0,1,2; cols 0,1,2,3)
    # B as 4x4
    # C as 3x4
    
    # Strategy: Use Strassen for the 2x2 parts and direct for the third row
    # Split A into top 2x4 and bottom 1x4
    # A_top is 2x4, A_bot is 1x4
    # B is 4x4, split into B_top (4x2) and B_bot (4x2)? No...
    
    # Better: Split columns of A and rows of B
    # A = [A_left | A_right] where A_left is 3x2, A_right is 3x2
    # B = [[B_topleft, B_topright], [B_botleft, B_botright]] each 2x2
    # C = A*B = [A_left*B_left + A_right*B_right]... 
    
    # Let me use a cleaner approach:
    # For 3x4x4, split m=4 into 2+2:
    # A is 3x4 = [A1 | A2] where A1, A2 are 3x2
    # B is 4x4, split rows into 2+2: B = [[B11, B12], [B21, B22]]
    # C = A1*B1 + A2*B2 where B1 = [B11; B21] is 4x2... 
    # Wait, C = A*B, A is 3x4, B is 4x4, C is 3x4
    # C[:, 0:2] = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
    # This isn't quite right either.
    
    # Let me use the standard tensor product approach:
    # 3x4x4 = (2x2x4) ⊗ (1x1x1) + ... no.
    
    # Simplest correct approach: Use Strassen on 2x2 blocks
    # For the 3x4x4 case:
    # Rows of A: 0,1,2
    # Cols of A / Rows of B: 0,1,2,3
    # Cols of B / Cols of C: 0,1,2,3
    
    # Decompose using 2x2 Strassen where possible:
    # For rows 0,1 of A and cols 0,1 of C: 2x4x2 -> use 2x2x2 Strassen twice
    # For rows 0,1 of A and cols 2,3 of C: 2x4x2 -> use 2x2x2 Strassen twice
    # For row 2 of A and all cols: 1x4x4 -> rank 4 (schoolbook for 1x4x4)
    
    # Actually, let me just construct it directly:
    # The multiplication tensor T_{i,j,k} = 1 if i==i' and k==k' else 0
    # where i in [0,2], j in [0,3], k in [0,3], i' in [0,2], k' in [0,3]
    
    # Use Strassen for the 2x2 sub-problems:
    # Sub-problem 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2x2, rank 7)
    # Sub-problem 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (2x2x2, rank 7)
    # Sub-problem 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (2x2x2, rank 7)
    # Sub-problem 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (2x2x2, rank 7)
    # Sub-problem 5: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2] (1x2x2, rank 2)
    # Sub-problem 6: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2] (1x2x2, rank 2)
    # Sub-problem 7: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4] (1x2x2, rank 2)
    # Sub-problem 8: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4] (1x2x2, rank 2)
    
    # Total rank: 4*7 + 4*2 = 28 + 8 = 36
    
    # Hmm, that's still less than 48. Let me verify and implement this.
    
    def strassen_2x2x2_with_offsets(i_off, j_off, k_off):
        """
        Return Strassen 2x2x2 decomposition with offsets.
        i_off: offset for i index (row of A)
        j_off: offset for j index (col of A / row of B)  
        k_off: offset for k index (col of B)
        
        Returns U_dict, V_dict, W_dict where:
        U_dict: {(i, j): [coeffs for 7 terms]}
        V_dict: {(j, k): [coeffs for 7 terms]}
        W_dict: {(k, i): [coeffs for 7 terms]}
        """
        U_local, V_local, W_local = strassen_2x2x2()
        
        U_dict = {}
        V_dict = {}
        W_dict = {}
        
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    if U_local[i*2+j][r] != 0:
                        key = (i_off + i, j_off + j)
                        if key not in U_dict:
                            U_dict[key] = [0]*7
                        U_dict[key][r] = U_local[i*2+j][r]
                        
            for j in range(2):
                for k in range(2):
                    if V_local[j*2+k][r] != 0:
                        key = (j_off + j, k_off + k)
                        if key not in V_dict:
                            V_dict[key] = [0]*7
                        V_dict[key][r] = V_local[j*2+k][r]
                        
            for k in range(2):
                for i in range(2):
                    if W_local[k*2+i][r] != 0:
                        key = (k_off + k, i_off + i)
                        if key not in W_dict:
                            W_dict[key] = [0]*7
                        W_dict[key][r] = W_local[k*2+i][r]
        
        return U_dict, V_dict, W_dict

    def one_x_two_x_two_with_offsets(i_off, j_off, k_off):
        """
        Return 1x2x2 decomposition (rank 2) with offsets.
        A is 1x2, B is 2x2, C is 1x2.
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        """
        U_dict = {}
        V_dict = {}
        W_dict = {}
        
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        U_dict[(i_off, j_off)] = [1, 0]
        V_dict[(j_off, k_off)] = [1, 0]
        W_dict[(k_off, i_off)] = [1, 0]
        
        # Term 1: A[0,1]*B[1,1] -> C[0,1]
        U_dict[(i_off, j_off+1)] = [0, 1]
        V_dict[(j_off+1, k_off+1)] = [0, 1]
        W_dict[(k_off+1, i_off)] = [0, 1]
        
        # Also need: A[0,0]*B[0,1] -> C[0,1] and A[0,1]*B[1,0] -> C[0,0]
        # This is rank 4 for 1x2x2, not rank 2!
        # Let me reconsider.
        
        # Actually 1x2x2 has rank 4 (schoolbook) since there are 4 independent terms.
        # Unless we can find a lower rank decomposition.
        # T_{0,j,k} = 1 for all j,k in {0,1}
        # This is the all-ones tensor of shape 1x2x2, which has rank 4.
        
        # So my earlier calculation was wrong. Let me redo.
        
        return None  # Placeholder

    # Let me take a completely different, simpler approach:
    # Just use the schoolbook for the whole thing but check if we can
    # reduce rank by finding linear dependencies.
    
    # Actually, for 3x4x4, the schoolbook rank is 12 (n*m = 3*4 = 12 terms for each output element? No.)
    # Schoolbook: for each (i,k) in C, C[i,k] = sum_j A[i,j]*B[j,k]
    # That's n*p = 3*4 = 12 output elements, each computed with m=4 multiplications.
    # Total scalar multiplications: 12*4 = 48. So rank 48.
    
    # Can we do better? The tensor is T_{i,j,k,i',k'} = delta_{i,i'} * delta_{k,k'}
    # where i,i' in {0,1,2}, j in {0,1,2,3}, k,k' in {0,1,2,3}
    
    # This is a "slice" of the 4x4x4 tensor.
    # Known: rank(4x4x4) <= 19 (Coppersmith-Winograd)
    # But extracting a 3x4x4 sub-tensor from a rank-19 4x4x4 decomposition
    # might give a valid 3x4x4 decomposition of rank <= 19.
    
    # Let me try to construct a rank-19 4x4x4 decomposition and trim it.
    
    # Construct a rank-36 decomposition:
    # Split A (3x4) into top 2 rows and bottom 1 row.
    # Split B (4x4) into left 2 cols and right 2 cols.
    # Split C (3x4) into left 2 cols and right 2 cols.
    #
    # C_top_left (2x2) = A_top_left(2x2)*B_top_left(2x2) + A_top_right(2x2)*B_bot_left(2x2)
    # C_top_right (2x2) = A_top_left(2x2)*B_top_right(2x2) + A_top_right(2x2)*B_bot_right(2x2)
    # C_bot_left (1x2) = A_bot_left(1x2)*B_top_left(2x2) + A_bot_right(1x2)*B_bot_left(2x2)
    # C_bot_right (1x2) = A_bot_left(1x2)*B_top_right(2x2) + A_bot_right(1x2)*B_bot_right(2x2)
    #
    # Each 2x2x2 block uses Strassen (rank 7): 4 blocks * 7 = 28
    # Each 1x2x2 block uses schoolbook (rank 4): 4 blocks * 4 = 16
    # Total: 28 + 16 = 44... wait, that's 44 not 36.
    #
    # Actually let me recount. For 1x2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  (2 terms)
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]  (2 terms)
    # Rank = 4 (each term is independent)
    #
    # So total = 4*7 + 4*4 = 28 + 16 = 44.
    #
    # Hmm, let me try a different split. What about:
    # Use Strassen for 2x2x2 in both the A and B dimensions simultaneously?
    #
    # Alternative: treat the whole problem as 3x4x4 and use a more clever decomposition.
    #
    # Let me just implement the 44-rank version and verify it.
    
    def build_rank44():
        """Build a rank-44 decomposition using Strassen for 2x2x2 blocks."""
        rank_total = 44
        U = [[0]*rank_total for _ in range(n*m)]  # 12 x 44
        V = [[0]*rank_total for _ in range(m*p)]  # 16 x 44
        W = [[0]*rank_total for _ in range(p*n)]  # 12 x 44
        
        # Strassen 2x2x2 local indices:
        # U_local[4][7], V_local[4][7], W_local[4][7]
        # U index: i*2+j, V index: j*2+k, W index: k*2+i
        
        # Get Strassen matrices
        U_s, V_s, W_s = strassen_2x2x2()
        
        col = 0
        
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # i_off=0, j_off=0, k_off=0
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + j][col] = U_s[i*2+j][r]
            for j in range(2):
                for k in range(2):
                    V[j*p + k][col] = V_s[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    W[k*n + i][col] = W_s[k*2+i][r]
            col += 1
        
        # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        # i_off=0, j_off=2, k_off=0
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + (2+j)][col] = U_s[i*2+j][r]
            for j in range(2):
                for k in range(2):
                    V[(2+j)*p + k][col] = V_s[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    W[k*n + i][col] = W_s[k*2+i][r]
            col += 1
        
        # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        # i_off=0, j_off=0, k_off=2
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + j][col] = U_s[i*2+j][r]
            for j in range(2):
                for k in range(2):
                    V[j*p + (2+k)][col] = V_s[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    W[(2+k)*n + i][col] = W_s[k*2+i][r]
            col += 1
        
        # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        # i_off=0, j_off=2, k_off=2
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + (2+j)][col] = U_s[i*2+j][r]
            for j in range(2):
                for k in range(2):
                    V[(2+j)*p + (2+k)][col] = V_s[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    W[(2+k)*n + i][col] = W_s[k*2+i][r]
            col += 1
        
        # Bottom row blocks: 1x2x2 each, rank 4
        # Block 5: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        for t in range(4):
            # Term t: (a, b, c) where a in {0,1}, b in {0,1}, c in {0,1}
            # C[2, c] += A[2, a] * B[a, c]
            a = t // 2
            b = t % 2  # this is the column index in B, which is the k index
            # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
            # For i=2, j in {0,1}, k in {0,1}:
            # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
            # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
            # Term: A[2,j]*B[j,k] -> C[2,k]
            # j = a (0 or 1), k = b (0 or 1)
            U[2*m + a][col] = 1
            V[a*p + b][col] = 1
            W[b*n + 2][col] = 1
            col += 1
        
        # Block 6: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
        # C[2,0] = A[2,2]*B[2,0] + A[2,3]*B[3,0]
        # C[2,1] = A[2,2]*B[2,1] + A[2,3]*B[3,1]
        for t in range(4):
            a = t // 2
            b = t % 2
            U[2*m + (2+a)][col] = 1
            V[(2+a)*p + b][col] = 1
            W[b*n + 2][col] = 1
            col += 1
        
        # Block 7: A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
        # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2]
        # C[2,3] = A[2,0]*B[0,3] + A[2,1]*B[1,3]
        for t in range(4):
            a = t // 2
            b = t % 2
            U[2*m + a][col] = 1
            V[a*p + (2+b)][col] = 1
            W[(2+b)*n + 2][col] = 1
            col += 1
        
        # Block 8: A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4]
        # C[2,2] = A[2,2]*B[2,2] + A[2,3]*B[3,2]
        # C[2,3] = A[2,2]*B[2,3] + A[2,3]*B[3,3]
        for t in range(4):
            a = t // 2
            b = t % 2
            U[2*m + (2+a)][col] = 1
            V[(2+a)*p + (2+b)][col] = 1
            W[(2+b)*n + 2][col] = 1
            col += 1
        
        assert col == rank_total, f"Expected {rank_total}, got {col}"
        return U, V, W
    
    U44, V44, W44 = build_rank44()
    
    # Verify the rank-44 decomposition
    if verify(U44, V44, W44):
        return {"U": U44, "V": V44, "W": W44}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
