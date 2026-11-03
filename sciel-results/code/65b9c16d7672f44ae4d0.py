"""Matrix multiplication 3x4x7 rank decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    deadline = time.time() + 140
    
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
        return {"U": U, "V": V, "W": W}
    
    # Rank-7 decomposition for 3x3x3 matrix multiplication
    # X(3x3) * Y(3x3) = Z(3x3)
    # Z[i,k] = sum_j X[i,j] * Y[j,k]
    #
    # Using the known rank-7 decomposition (from literature):
    # We need to find 7 triples (u_r, v_r, w_r) such that
    # sum_r u_r[i,j] * v_r[j,k] = w_r[i,k] for all i,j,k
    #
    # One valid rank-7 decomposition for 3x3x3:
    # Based on the structure, let me construct it explicitly.
    
    def rank7_3x3x3():
        """
        Returns U, V, W for 3x3x3 matrix mult with rank 7.
        U shape: (9, 7), V shape: (9, 7), W shape: (9, 7)
        Indexing: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
        """
        U = [[0]*7 for _ in range(9)]
        V = [[0]*7 for _ in range(9)]
        W = [[0]*7 for _ in range(9)]
        
        # Term 1: X[0,0] * Y[0,0] -> Z[0,0]
        U[0*3+0][0] = 1
        V[0*3+0][0] = 1
        W[0*3+0][0] = 1
        
        # Term 2: X[0,1] * Y[1,0] -> Z[0,0]
        U[0*3+1][1] = 1
        V[1*3+0][1] = 1
        W[0*3+0][1] = 1
        
        # Term 3: X[0,2] * Y[2,0] -> Z[0,0]
        U[0*3+2][2] = 1
        V[2*3+0][2] = 1
        W[0*3+0][2] = 1
        
        # Term 4: X[1,0] * Y[0,1] -> Z[1,1]
        U[1*3+0][3] = 1
        V[0*3+1][3] = 1
        W[1*3+1][3] = 1
        
        # Term 5: X[1,1] * Y[1,1] -> Z[1,1]
        U[1*3+1][4] = 1
        V[1*3+1][4] = 1
        W[1*3+1][4] = 1
        
        # Term 6: X[1,2] * Y[2,1] -> Z[1,1]
        U[1*3+2][5] = 1
        V[2*3+1][5] = 1
        W[1*3+1][5] = 1
        
        # Term 7: X[2,0] * Y[0,2] -> Z[2,2]
        U[2*3+0][6] = 1
        V[0*3+2][6] = 1
        W[2*3+2][6] = 1
        
        # But this is just schoolbook with 7 terms, not a true rank-7 decomposition!
        # We need a proper rank-7 decomposition that uses sums.
        
        # Let me use a verified rank-7 decomposition for 3x3x3.
        # The key insight: we can combine terms using linearity.
        
        # Actually, the minimal rank for 3x3x3 is 7, and here's one decomposition:
        # Based on the work by Landsberg and others, here's a concrete one:
        
        # Let me construct it from scratch using the property that
        # we need sum_r u_r ⊗ v_r ⊗ w_r = multiplication tensor
        
        # A known rank-7 decomposition:
        # r=0: (X00+X22)(Y00+Y22) contributes to Z00, Z02, Z20, Z22
        # This is getting complex. Let me use a simpler verified approach.
        
        # Actually, for correctness, let me just use schoolbook for 3x3x3 (rank 9)
        # and build up from there. The total will still be much better than 84.
        
        # Schoolbook 3x3x3: rank 9
        U = [[0]*9 for _ in range(9)]
        V = [[0]*9 for _ in range(9)]
        W = [[0]*9 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W
    
    # Let me build the full decomposition using schoolbook subcomponents
    # but structured to minimize total rank.
    
    # Strategy: 3x4x7 = (3x4x4) + (3x4x3)
    # 3x4x4: A(3x4)*B(4x4) = C(3x4)
    #   Split A = [A1(3x3) | a(3x1)], B = [B1(3x4); b(1x4)]
    #   C = A1*B1 + a*b
    #   A1*B1: 3x3x4, schoolbook rank 36... no, 3*3*4=36
    #   That's too much.
    
    # Better: Use the structure more carefully.
    # 3x3x4: A(3x3)*B(3x4) = C(3x4)
    #   Split B = [B1(3x3) | b(3x1)]
    #   C = A*B1 + A*b
    #   A*B1: 3x3x3, rank 9 (schoolbook)
    #   A*b: 3x3x1, rank 9 (each of 3 rows of A dotted with b, gives 3 values, but we need 3*3=9 terms for the full 3x3 matrix... 
    #   Wait, A*b where A is 3x3 and b is 3x1 gives a 3x1 vector. That's only 3 outputs.
    #   So A*b contributes to column 3 of C. Rank 3.
    #   Total 3x3x4: 9 + 3 = 12
    
    # 3x4x4: A(3x4)*B(4x4) = C(3x4)
    #   Split A = [A1(3x3) | a(3x1)], B = [B1(3x4); b(1x4)]
    #   C = A1*B1 + a*b
    #   A1*B1: 3x3x4, rank 12 (from above)
    #   a*b: a is 3x1, b is 1x4, outer product gives 3x4 matrix, rank 12
    #   Total 3x4x4: 12 + 12 = 24
    
    # 3x4x3: A(3x4)*B(4x3) = C(3x3)
    #   Split A = [A1(3x3) | a(3x1)], B = [B1(3x3); b(1x3)]
    #   C = A1*B1 + a*b
    #   A1*B1: 3x3x3, rank 9
    #   a*b: a is 3x1, b is 1x3, outer product gives 3x3, rank 9
    #   Total 3x4x3: 9 + 9 = 18
    
    # Grand total: 24 + 18 = 42
    
    # Let me construct this explicitly.
    
    U_list = []  # Will collect (n*m, rank) entries
    V_list = []  # Will collect (m*p, rank) entries
    W_list = []  # Will collect (p*n, rank) entries
    
    # We'll build column by column (each column is one rank-1 term)
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (28, rank)
    # W has shape (p*n, rank) = (21, rank)
    
    # Let me build the full U, V, W matrices directly.
    # Total rank = 42
    
    total_rank = 42
    U = [[0]*total_rank for _ in range(n*m)]  # 12 x 42
    V = [[0]*total_rank for _ in range(m*p)]  # 28 x 42
    W = [[0]*total_rank for _ in range(p*n)]  # 21 x 42
    
    col = 0
    
    # PART 1: 3x4x4 part (columns 0-3 of B, i.e., k=0,1,2,3)
    # A(3x4) * B(4x4) = C(3x4)
    # A = [A1(3x3) | a(3x1)] where A1 = A[:,0:3], a = A[:,3]
    # B = [B1(3x4); b(1x4)] where B1 = B[0:3,:], b = B[3,:]
    # C = A1*B1 + a*b
    
    # Sub-part 1a: A1*B1 where A1 is 3x3, B1 is 3x4
    # B1 = [B11(3x3) | B12(3x1)]
    # A1*B1 = A1*B11 + A1*B12
    # A1*B11: 3x3x3, schoolbook rank 9
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # U index: i*m + j = i*4 + j (since A is 3x4, rows i, cols j)
                # But A1 is columns 0,1,2 of A, so j=0,1,2
                # V index: j*p + k = j*7 + k (since B is 4x7, but we're using B1 which is 3x4, rows j=0,1,2)
                # Wait, B is 4x7. B1 is the top 3 rows of B, so B1[j,k] for j=0,1,2 and k=0,1,2,3
                # In the full V, B[j,k] is at index j*7+k
                # W index: k*n + i = k*3 + i (C is 3x4, but we're computing C[:,0:3] which is 3x3)
                # In the full W, C[k,i]... wait, the convention is W[k*n+i, r]
                # C[i,k] is the output. The trace convention says W[k*n+i, r]
                # So C[i,k] corresponds to W[k*3+i, r]
                
                U[i*4 + j][col] = 1  # A1[i,j] = A[i,j] for j=0,1,2
                V[j*7 + k][col] = 1  # B1[j,k] = B[j,k] for j=0,1,2
                W[k*3 + i][col] = 1  # C[i,k]
                col += 1
    
    # A1*B12: A1 is 3x3, B12 is 3x1 (column 3 of B1, i.e., B[0:3,3])
    # Result is 3x1 vector, contributes to C[:,3]
    for i in range(3):
        for j in range(3):
            # A1[i,j] * B[ j, 3 ] contributes to C[i,3]
            U[i*4 + j][col] = 1
            V[j*7 + 3][col] = 1
            W[3*3 + i][col] = 1  # C[i,3] -> W[3*3+i]
            col += 1
    
    # Sub-part 1b: a*b where a = A[:,3] (3x1), b = B[3,:] (1x4)
    # a*b is 3x4 outer product, contributes to all of C
    for i in range(3):
        for k in range(4):
            # A[i,3] * B[3,k] contributes to C[i,k]
            U[i*4 + 3][col] = 1
            V[3*7 + k][col] = 1
            W[k*3 + i][col] = 1
            col += 1
    
    # PART 2: 3x4x3 part (columns 4-6 of B, i.e., k=4,5,6)
    # A(3x4) * B(4x3) = C(3x3)
    # A = [A1(3x3) | a(3x1)], B = [B1(3x3); b(1x3)]
    # B1 = B[0:3, 4:7], b = B[3, 4:7]
    # C = A1*B1 + a*b
    
    # Sub-part 2a: A1*B1 where A1 is 3x3, B1 is 3x3
    # Schoolbook rank 9
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A1[i,j] * B[j, 4+k] contributes to C[i,k]
                U[i*4 + j][col] = 1
                V[j*7 + (4+k)][col] = 1
                W[(4+k)*3 + i][col] = 1  # C[i,k] for the 3x3x3 part, k=0,1,2 -> output columns 4,5,6
                col += 1
    
    # Sub-part 2b: a*b where a = A[:,3] (3x1), b = B[3, 4:7] (1x3)
    # a*b is 3x3 outer product
    for i in range(3):
        for k in range(3):
            # A[i,3] * B[3, 4+k] contributes to C[i,k]
            U[i*4 + 3][col] = 1
            V[3*7 + (4+k)][col] = 1
            W[(4+k)*3 + i][col] = 1
            col += 1
    
    # Verify col == total_rank
    # Part 1a: 3*3*3 = 27
    # Part 1b: 3*3 = 9
    # Part 1c: 3*4 = 12
    # Part 2a: 3*3*3 = 27
    # Part 2b: 3*3 = 9
    # Total: 27+9+12+27+9 = 84... that's just schoolbook again!
    
    # I made an error. Let me recount.
    # The issue is I'm using schoolbook for the 3x3x3 parts.
    # 3x3x3 schoolbook is rank 9, not 7.
    
    # Let me verify: 27 + 9 + 12 + 27 + 9 = 84. Yes, that's schoolbook.
    
    # I need to use rank-7 for the 3x3x3 parts to get improvement.
    # But I need a valid rank-7 decomposition.
    
    # Let me just return schoolbook for now since I can't easily verify a rank-7 decomposition.
    return schoolbook(n, m, p)
# EVOLVE-BLOCK-END

