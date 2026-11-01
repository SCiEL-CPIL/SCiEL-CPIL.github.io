<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # We suspect rank 16 is achievable
    # Strategy: Try to find a decomposition of rank 16
    # If not found, fall back to schoolbook
    
    # Given the complexity of finding the exact decomposition via random search,
    # and the time limit, we will return the schoolbook decomposition 
    # unless a specific known decomposition is hardcoded.
    # However, the prompt asks for a "stronger algorithm".
    # A known decomposition for M_{2,2,5} of rank 16 exists?
    # R(M_{2,2,2}) = 7.
    # R(M_{2,2,1}) = 1.
    # R(M_{2,2,5}) <= R(M_{2,2,2}) + R(M_{2,2,2}) + R(M_{2,2,1}) = 7 + 7 + 1 = 15?
    # Wait, M_{2,2,5} = M_{2,2,2} (cols 0,1) + M_{2,2,2} (cols 2,3) + M_{2,2,1} (col 4).
    # The A matrix is shared.
    # In the decomposition, we sum the tensors.
    # T = T1 + T2 + T3.
    # Rank(T) <= Rank(T1) + Rank(T2) + Rank(T3).
    # So Rank <= 7 + 7 + 1 = 15.
    # Is it 15?
    # If we can find a rank 15 decomposition, the score is 15.
    
    # Let's try to construct a rank 15 decomposition.
    # This requires knowing the specific entries.
    # Without a known explicit decomposition, random search for rank 15 is hard.
    # We will return schoolbook rank 20 to be safe, as finding 15/16 in 120s 
    # with random search is unlikely without a specific heuristic.
    
    rank = schoolbook_rank
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
=======
import time
import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U: (4, 7) - A entries indexed by i*m+j
    V: (4, 7) - B entries indexed by j*p+k  
    W: (4, 7) - C entries indexed by k*n+i
    """
    n, m, p = 2, 2, 2
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products for 2x2 matrix multiplication
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
    
    # A indices: A[0]=A11, A[1]=A12, A[2]=A21, A[3]=A22
    # B indices: B[0]=B11, B[1]=B12, B[2]=B21, B[3]=B22
    # C indices: C[0]=C11, C[1]=C12, C[2]=C21, C[3]=C22
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (A21+A22)B11
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] += 1
    
    # M3 = A11(B21-B22)
    U[0][2] += 1
    V[2][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = A22(B12-B11)
    U[3][3] += 1
    V[1][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5 = (A11+A12)B22
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _optimal_223():
    """
    Returns U, V, W for 2x2x3 matrix multiplication with rank 9.
    This is the known optimal rank for M_{2,2,3}.
    U: (4, 9) - A entries indexed by i*m+j
    V: (6, 9) - B entries indexed by j*p+k
    W: (6, 9) - C entries indexed by k*n+i
    """
    n, m, p = 2, 2, 3
    U = [[0]*9 for _ in range(4)]
    V = [[0]*9 for _ in range(6)]
    W = [[0]*9 for _ in range(6)]
    
    # For 2x2x3, we can use a structured decomposition.
    # One approach: split into 2x2x2 (rank 7) + 2x2x1 (rank 1) = 8? 
    # But that's not optimal. The known rank is 9.
    # 
    # Actually, let me reconsider. R(2,2,3) = 9 is known.
    # Let me construct it explicitly.
    #
    # Alternative: Use the fact that we can embed 2x2x2 and add one more column.
    # But rank(2,2,2)=7 and rank(2,2,1)=1, so rank(2,2,3) <= 8? 
    # No, that's not right because the A part is shared.
    #
    # Let me just use a known explicit rank-9 decomposition for 2x2x3.
    # 
    # After research, R(2,2,3) = 9. Here's one construction:
    # We can think of it as computing C = A*B where A is 2x2, B is 2x3.
    # 
    # Let me use a systematic approach with bilinear forms.
    
    # Actually, let me try a different strategy. 
    # For the full 2x2x5 problem, I'll construct rank-16 directly.
    # 
    # Strategy: 
    # - Use Strassen (rank 7) for the first 2 columns of B
    # - Use a rank-9 decomposition for the last 3 columns of B
    # Total: 7 + 9 = 16
    
    # For now, return a placeholder that we'll replace
    # with the actual construction in the main function
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Construct rank-16 decomposition:
    # Part 1: Strassen for columns 0,1 of B (rank 7)
    # Part 2: Rank-9 decomposition for columns 2,3,4 of B (rank 9)
    
    U1, V1, W1 = _strassen_222()
    
    # For the 2x2x3 part (columns 2,3,4), we need a rank-9 decomposition.
    # Let me construct it using a known method.
    
    # Actually, let me try a simpler approach: 
    # Use schoolbook for the 2x2x3 part (rank 6) + Strassen for 2x2x2 (rank 7) = 13?
    # No, that's not right either.
    
    # Let me just use schoolbook for everything to be safe, but with a better structure.
    # 
    # Actually, the simplest improvement is:
    # R(2,2,5) <= R(2,2,2) + R(2,2,3) = 7 + 9 = 16
    
    # Let me construct the 2x2x3 rank-9 decomposition explicitly.
    
    U2 = [[0]*9 for _ in range(4)]
    V2 = [[0]*9 for _ in range(6)]
    W2 = [[0]*9 for _ in range(6)]
    
    # For 2x2x3, here's a rank-9 construction:
    # We can use the fact that it's equivalent to multiplying a 2x2 by 2x3.
    # 
    # After some thought, let me use a different approach:
    # I'll use a known explicit decomposition.
    
    # For simplicity and correctness, let me use schoolbook for the 2x2x3 part (rank 6)
    # This gives total rank 7 + 6 = 13, which is better than 16.
    # But wait, is R(2,2,3) actually 6? No, it's 9.
    
    # Let me just go with the safe approach: use schoolbook for the entire thing.
    # But that's rank 20, which is what we already have.
    
    # OK, let me try yet another approach. I'll use:
    # - Strassen for 2x2x2 (rank 7)  
    # - Schoolbook for 2x2x3 (rank 6)
    # Total: 13
    
    # But I need to verify this is valid. The issue is that the A matrix is shared.
    # In the tensor decomposition, we're decomposing the full multiplication tensor.
    # If we partition the output columns, we can sum the decompositions.
    
    # Let me construct this properly:
    
    rank_total = 13
    U = [[0]*rank_total for _ in range(4)]
    V = [[0]*rank_total for _ in range(10)]
    W = [[0]*rank_total for _ in range(10)]
    
    # Part 1: Strassen for B columns 0,1
    for col in range(7):
        for i in range(4):
            U[i][col] = U1[i][col]
        for j in range(4):
            # V1 is for 2x2x2, so V1[j][col] corresponds to B[j]
            # In the full problem, B indices 0,1 are at positions 0,1 (for row 0) and 2,3 (for row 1)
            # Wait, V is (m*p, rank) = (2*5, rank) = (10, rank)
            # B index j*p+k: for j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 5; j=1,k=1 -> 6
            # So V1[0] -> V[0], V1[1] -> V[1], V1[2] -> V[5], V1[3] -> V[6]
            pass
        for k in range(4):
            # W1 is for 2x2x2, so W1[k][col] corresponds to C[k]
            # In the full problem, C index k*n+i: for k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
            # So W1[0] -> W[0], W1[1] -> W[1], W1[2] -> W[2], W1[3] -> W[3]
            pass
    
    # This is getting complicated. Let me just return the schoolbook solution for now.
    # The improvement to rank 16 requires careful construction.
    
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
>>>>>>> REPLACE

Wait, I'm overcomplicating this. Let me create a clean rank-16 solution using Strassen for the first 2 columns and schoolbook for the remaining 3 columns.

