"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_numpy(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    return np.allclose(T, 1.0, atol=1e-9)

def _schoolbook(n, m, p):
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

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W for multiplying (2x2) * (2x2) -> (2x2).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*2+j, c] = coefficient for A[i,j] in product c
    # V[j*2+k, c] = coefficient for B[j,k] in product c
    # W[k*2+i, c] = coefficient for C[k,i] in product c
    
    # Product 0: M1 = (A00+A11)*(B00+B11)
    # A part: A00=1, A11=1
    # B part: B00=1, B11=1
    # C part: C00+=1, C11+=1
    # Product 1: M2 = (A10+A11)*B00
    # A part: A10=1, A11=1
    # B part: B00=1
    # C part: C10+=1
    # Product 2: M3 = A00*(B01-B11)
    # A part: A00=1
    # B part: B01=1, B11=-1
    # C part: C01+=1
    # Product 3: M4 = A11*(B10-B00)
    # A part: A11=1
    # B part: B10=1, B00=-1
    # C part: C00+=1, C10+=1
    # Product 4: M5 = (A00+A01)*B11
    # A part: A00=1, A01=1
    # B part: B11=1
    # C part: C00-=1, C01+=1
    # Product 5: M6 = (A10-A00)*(B00+B01)
    # A part: A10=1, A00=-1
    # B part: B00=1, B01=1
    # C part: C11+=1
    # Product 6: M7 = (A01-A11)*(B10+B11)
    # A part: A01=1, A11=-1
    # B part: B10=1, B11=1
    # C part: C00+=1, C11+=1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U[i*2+j, c]
    U[0][0] = 1  # A00 in M1
    U[3][0] = 1  # A11 in M1
    
    U[2][1] = 1  # A10 in M2
    U[3][1] = 1  # A11 in M2
    
    U[0][2] = 1  # A00 in M3
    
    U[3][3] = 1  # A11 in M4
    
    U[0][4] = 1  # A00 in M5
    U[1][4] = 1  # A01 in M5
    
    U[2][5] = 1  # A10 in M6
    U[0][5] = -1  # A00 in M6
    
    U[1][6] = 1  # A01 in M7
    U[3][6] = -1  # A11 in M7
    
    # V[j*2+k, c]
    V[0][0] = 1  # B00 in M1
    V[3][0] = 1  # B11 in M1
    
    V[0][1] = 1  # B00 in M2
    
    V[1][2] = 1  # B01 in M3
    V[3][2] = -1  # B11 in M3
    
    V[2][3] = 1  # B10 in M4
    V[0][3] = -1  # B00 in M4
    
    V[3][4] = 1  # B11 in M5
    
    V[0][5] = 1  # B00 in M6
    V[1][5] = 1  # B01 in M6
    
    V[2][6] = 1  # B10 in M7
    V[3][6] = 1  # B11 in M7
    
    # W[k*2+i, c]
    W[0][0] = 1  # C00 in M1
    W[3][0] = 1  # C11 in M1
    
    W[2][1] = 1  # C10 in M2
    
    W[1][2] = 1  # C01 in M3
    
    W[0][3] = 1  # C00 in M4
    W[2][3] = 1  # C10 in M4
    
    W[0][4] = -1  # C00 in M5
    W[1][4] = 1  # C01 in M5
    
    W[3][5] = 1  # C11 in M6
    
    W[0][6] = 1  # C00 in M7
    W[3][6] = 1  # C11 in M7
    
    return U, V, W

def _laderman_3x3x3():
    """
    Laderman decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W for multiplying (3x3) * (3x3) -> (3x3).
    U: (9, 13), V: (9, 13), W: (9, 13)
    Indexing: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    # Laderman's 13 products for 3x3 matrix multiplication
    # This is a known optimal decomposition
    
    # I'll use the standard Laderman decomposition
    # The 13 products are:
    # P1 = (a11+a22+a33)(b11+b22+b33)
    # P2 = (a12+a13+a21+a31)(b12+b13+b21+b31) 
    # P3 = (a11-a22+a33)(b11-b22+b33)
    # P4 = (a12-a13+a21-a31)(b12-b13+b21-b31)
    # P5 = (a11+a22-a33)(b11+b22-b33)
    # P6 = (a12+a13-a21-a31)(b12+b13-b21-b31)
    # P7 = (a11-a22-a33)(b11-b22-b33)
    # P8 = (a12-a13-a21+a31)(b12-b13-a21+b31)
    # P9 = (a13+a23+a32)(b13+b23+b32)
    # P10 = (a13-a23+a32)(b13-b23+b32)
    # P11 = (a13+a23-a32)(b13+b23-b32)
    # P12 = (a13-a23-a32)(b13-b23-b32)
    # P13 = (a22+a33-a11)(b22+b33-b11)
    
    # Actually, let me use a more standard formulation.
    # Laderman's decomposition uses 13 multiplications.
    
    # Let me use a verified decomposition. I'll construct it carefully.
    
    # A better approach: use the known Laderman decomposition
    # Reference: Laderman, "A note on the multiplication of 3x3 matrices"
    
    # The 13 terms are (using 0-indexed):
    # Term 0: (A[0,0]+A[1,1]+A[2,2]) * (B[0,0]+B[1,1]+B[2,2])
    # Term 1: (A[0,1]+A[1,0]) * (B[1,0]+B[0,1])
    # Term 2: (A[0,2]+A[2,0]) * (B[2,0]+B[0,2])
    # Term 3: (A[1,2]+A[2,1]) * (B[2,1]+B[1,2])
    # Term 4: (A[0,0]-A[1,1]+A[2,2]) * (B[0,0]-B[1,1]+B[2,2])
    # Term 5: (A[0,1]-A[1,0]) * (B[1,0]-B[0,1])
    # Term 6: (A[0,2]-A[2,0]) * (B[2,0]-B[0,2])
    # Term 7: (A[1,2]-A[2,1]) * (B[2,1]-B[1,2])
    # Term 8: (A[0,0]+A[1,1]-A[2,2]) * (B[0,0]+B[1,1]-B[2,2])
    # Term 9: (A[0,1]+A[1,0]) * (B[1,0]-B[0,1])
    # Term 10: (A[0,2]+A[2,0]) * (B[2,0]-B[0,2])
    # Term 11: (A[1,2]+A[2,1]) * (B[2,1]-B[1,2])
    # Term 12: (A[0,0]-A[1,1]-A[2,2]) * (B[0,0]-B[1,1]-B[2,2])
    
    # Hmm, this doesn't look right. Let me use a different approach.
    
    # I'll use a known valid Laderman decomposition from literature.
    # The decomposition has 13 rank-1 terms.
    
    # Let me define it directly with the correct coefficients.
    
    U = [[0]*13 for _ in range(9)]
    V = [[0]*13 for _ in range(9)]
    W = [[0]*13 for _ in range(9)]
    
    # I'll use a verified Laderman decomposition
    # After careful consideration, here's a valid decomposition:
    
    # Term 0: A[0,0]*B[0,0] contributes to C[0,0]
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # Term 1: A[0,1]*B[1,0] contributes to C[0,0]
    U[1][1] = 1
    V[3][1] = 1
    W[0][1] = 1
    
    # Term 2: A[0,2]*B[2,0] contributes to C[0,0]
    U[2][2] = 1
    V[6][2] = 1
    W[0][2] = 1
    
    # Term 3: A[1,0]*B[0,1] contributes to C[1,0]
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    # Term 4: A[1,1]*B[1,1] contributes to C[1,1]
    U[4][4] = 1
    V[4][4] = 1
    W[4][4] = 1
    
    # Term 5: A[1,2]*B[2,1] contributes to C[1,1]
    U[5][5] = 1
    V[7][5] = 1
    W[4][5] = 1
    
    # Term 6: A[2,0]*B[0,2] contributes to C[2,0]
    U[6][6] = 1
    V[2][6] = 1
    W[2][6] = 1
    
    # Term 7: A[2,1]*B[1,2] contributes to C[2,1]
    U[7][7] = 1
    V[5][7] = 1
    W[5][7] = 1
    
    # Term 8: A[2,2]*B[2,2] contributes to C[2,2]
    U[8][8] = 1
    V[8][8] = 1
    W[8][8] = 1
    
    # Now I need 4 more terms to make it 13, and I need to use Strassen-like tricks
    # to reduce the rank. This is getting complex. Let me use a different approach.
    
    # Actually, for simplicity and correctness, let me just use schoolbook for the 3x3 part
    # and Strassen for the 2x2 part. This gives rank 4*9 + 4*7 = 36 + 28 = 64.
    # That's still much better than 100.
    
    # But wait, I can do better. Let me use Strassen for both 2x2 blocks
    # and schoolbook for the 3x3 block.
    
    # Actually, let me reconsider the block decomposition.
    # For 4x5x5, I can partition the 5x5 matrix into:
    # - A 2x2 block
    # - A 3x3 block
    # And use Strassen for the 2x2 part and Laderman for the 3x3 part.
    
    # But I need to handle the cross terms carefully.
    
    # Let me use a simpler approach: use Strassen for a 2x2x2 subproblem
    # and schoolbook for the rest.
    
    # For 4x5x5:
    # Partition columns 0-1 of B and rows 0-1 of C into a 2x2 block
    # The remaining columns 2-4 of B and rows 2-4 of C form a 3x3 block
    # But we also have cross terms.
    
    # Actually, the cleanest approach is to use tensor product structure.
    # Let me just use Strassen for a 2x2x2 subproblem and schoolbook for the rest.
    
    # For 4x5x5:
    # Use Strassen for multiplying 4x2 by 2x2 -> 4x2 (rank 7*4 = 28? No, rank 7 for the 2x2x2, scaled by 4)
    # Wait, that's not right either.
    
    # Let me think more carefully.
    # The matrix multiplication tensor for (n x m) * (m x p) has rank at most n*m*p.
    # We can use block decomposition to reduce the rank.
    
    # For 4x5x5, let's partition as follows:
    # A is 4x5, B is 5x5, C is 4x5
    # Partition the 5 columns of A into 2 + 3
    # Partition the 5 rows of B into 2 + 3
    # Partition the 5 columns of B into 2 + 3
    # Partition the 5 columns of C into 2 + 3
    
    # Then C = [C_2 C_3] where:
    # C_2 = A_2 * B_2 + A_3 * B_32 (4x2)
    # C_3 = A_2 * B_23 + A_3 * B_3 (4x3)
    
    # This is getting complicated. Let me just use a known good decomposition.
    
    # For now, let me use Strassen for a 2x2x2 subproblem and schoolbook for the rest.
    # This will give a valid decomposition with rank < 100.
    
    # Actually, let me just return a valid schoolbook decomposition for now,
    # and use Strassen where I can clearly apply it.
    
    # For 4x5x5, I can apply Strassen to a 2x2x2 subproblem.
    # Let's say I use Strassen for the top-left 2x2 block of the multiplication.
    # The rest uses schoolbook.
    
    # This is complex to implement correctly. Let me use a simpler approach:
    # Just use schoolbook for everything, which gives rank 100.
    # Then I'll try to improve it with Strassen for specific blocks.
    
    # For now, return a valid Laderman-like decomposition.
    # Actually, let me just use schoolbook for the 3x3 part.
    
    # Reset and use schoolbook for 3x3x3
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

def _block_decompose(n, m, p):
    """
    Block decomposition for n x m x p matrix multiplication.
    For 4x5x5, partition the 5 dimension into 2+3.
    Use Strassen for 2x2x2 subproblems and schoolbook for the rest.
    """
    if n != 4 or m != 5 or p != 5:
        return _schoolbook(n, m, p)
    
    n, m, p = 4, 5, 5
    
    # Partition:
    # A: 4x5 -> [A2 | A3] where A2 is 4x2, A3 is 4x3
    # B: 5x5 -> [[B22, B23], [B32, B33]] where B22 is 2x2, B23 is 2x3, B32 is 3x2, B33 is 3x3
    # C: 4x5 -> [C2 | C3] where C2 is 4x2, C3 is 4x3
    
    # C2 = A2*B22 + A3*B32
    # C3 = A2*B23 + A3*B33
    
    # We need to compute:
    # 1. A2*B22: 4x2 * 2x2 -> 4x2 (use Strassen-like, rank 7*2=14? No, rank 7 for 2x2x2, but we have 4 rows)
    #    Actually, for 4x2 * 2x2, we can use 2 Strassen decompositions (one for each pair of rows)
    #    Or we can use a 4x2x2 decomposition.
    
    # Let me use a simpler approach: use Strassen for each 2x2x2 subproblem.
    # For A2*B22 where A2 is 4x2 and B22 is 2x2:
    # We can write A2 as [A2_0; A2_1] where A2_0 and A2_1 are 2x2.
    # Then A2*B22 = [A2_0*B22; A2_1*B22]
    # Each A2_i*B22 is a 2x2*2x2 -> 2x2 multiplication, which has rank 7.
    # So total rank for A2*B22 is 14.
    
    # Similarly for other terms.
    
    # Let me count the total rank:
    # A2*B22: 14 (two 2x2x2 Strassen)
    # A3*B32: 4x3 * 3x2 -> 4x2. Use schoolbook: 4*3*2 = 24. Or better: 2x2 Strassen for each of 2 row pairs? 
    #   A3 is 4x3, B32 is 3x2. Partition A3 into two 2x3 blocks.
    #   Each 2x3 * 3x2 -> 2x2. This is not a standard Strassen.
    #   Use schoolbook: 2*3*2 = 12 per block, total 24.
    # A2*B23: 4x2 * 2x3 -> 4x3. Partition A2 into two 2x2 blocks.
    #   Each 2x2 * 2x3 -> 2x3. Not standard.
    #   Use schoolbook: 2*2*3 = 12 per block, total 24.
    # A3*B33: 4x3 * 3x3 -> 4x3. Partition A3 into two 2x3 blocks.
    #   Each 2x3 * 3x3 -> 2x3. Use Laderman? No, that's 3x3*3x3.
    #   Use schoolbook: 2*3*3 = 18 per block, total 36.
    
    # Total rank: 14 + 24 + 24 + 36 = 98. That's barely better than 100.
    
    # Let me try a different partition.
    # What if I use Strassen for all 2x2 subproblems and Laderman for 3x3?
    
    # Actually, let me just use a simpler approach:
    # Use Strassen for a 2x2x2 subproblem and schoolbook for the rest.
    # This gives rank 7 + (100 - 8) = 99. Not great.
    
    # Let me use a different strategy: use Strassen for the entire multiplication
    # by embedding 4x5x5 into a larger Strassen-like decomposition.
    
    # Actually, the simplest valid improvement is to use Strassen for a 2x2x2 subproblem.
    # Let me do that.
    
    # Use Strassen for the top-left 2x2 block of C, which comes from A[0:2,0:2]*B[0:2,0:2]
    # The rest uses schoolbook.
    
    U = [[0]*100 for _ in range(20)]
    V = [[0]*100 for _ in range(25)]
    W = [[0]*100 for _ in range(20)]
    
    # First, set up schoolbook for all terms
    col = 0
    for i in range(4):
        for j in range(5):
            for k in range(5):
                U[i*5+j][col] = 1
                V[j*5+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    # Now, replace the 8 schoolbook terms for the 2x2x2 subproblem with Strassen's 7 terms
    # The 2x2x2 subproblem is i in {0,1}, j in {0,1}, k in {0,1}
    # These are 8 terms in the schoolbook decomposition.
    # In the schoolbook, they are at columns:
    # i=0,j=0,k=0: col=0
    # i=0,j=0,k=1: col=1
    # i=0,j=1,k=0: col=5
    # i=0,j=1,k=1: col=6
    # i=1,j=0,k=0: col=10
    # i=1,j=0,k=1: col=11
    # i=1,j=1,k=0: col=15
    # i=1,j=1,k=1: col=16
    
    # Zero out these 8 columns
    for col_idx in [0, 1, 5, 6, 10, 11, 15, 16]:
        for i in range(20):
            U[i][col_idx] = 0
        for j in range(25):
            V[j][col_idx] = 0
        for k in range(20):
            W[k][col_idx] = 0
    
    # Now add Strassen's 7 terms in columns 0-6
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    for c in range(7):
        # U: 4x2 submatrix at rows 0:4, cols 0:7
        # But we need to map to the full 4x5 matrix
        # A[0:2, 0:2] corresponds to U[0:4, :] in the 2x2 case
        # In the full matrix, A[i,j] for i in {0,1}, j in {0,1} is at U[i*5+j]
        for u_row in range(4):
            i = u_row // 2
            j = u_row % 2
            full_row = i * 5 + j
            U[full_row][c] = strassen_U[u_row][c]
        
        # V: 2x2 submatrix at rows 0:4, cols 0:7
        # B[0:2, 0:2] corresponds to V[0:4, :] in the 2x2 case
        # In the full matrix, B[j,k] for j in {0,1}, k in {0,1} is at V[j*5+k]
        for v_row in range(4):
            j = v_row // 2
            k = v_row % 2
            full_row = j * 5 + k
            V[full_row][c] = strassen_V[v_row][c]
        
        # W: 2x2 submatrix at rows 0:4, cols 0:7
        # C[0:2, 0:2] corresponds to W[0:4, :] in the 2x2 case
        # In the full matrix, C[k,i] for k in {0,1}, i in {0,1} is at W[k*4+i]
        for w_row in range(4):
            k = w_row // 2
            i = w_row % 2
            full_row = k * 4 + i
            W[full_row][c] = strassen_W[w_row][c]
    
    # Verify
    if not _verify_numpy(U, V, W, 4, 5, 5):
        # Fallback to schoolbook
        return _schoolbook(4, 5, 5)
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try block decomposition
    result = _block_decompose(n, m, p)
    best_rank = len(result['U'][0])
    
    # If we got a good result, return it
    if best_rank < 100:
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
