"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 x 3x10 matrix multiplication.
    
    Strategy: Use a Strassen-like decomposition for the 2x3 block, then extend to 3x10.
    The key insight is that we can decompose the 2x3 multiplication using rank 5 (instead of 6),
    and then use this structure to reduce the total rank.
    
    For 2x3 x 3x10:
    - Schoolbook: rank 2*3*10 = 60
    - Using Strassen for the 2x2 part: we can get rank 54
    
    The approach:
    1. Split the 3 columns into a 2-column part and a 1-column part
    2. Use Strassen's algorithm (rank 7) for the 2x2 x 2x2 part
    3. Handle the remaining 2x1 x 1x10 part directly
    4. Combine to get total rank 7*10/2 + 2*10/2 = 35 + 10 = 45? No, let me reconsider.
    
    Actually, let me use a more systematic approach:
    
    For multiplying A (2x3) by B (3x10):
    We can write this as:
    - Split A into A1 (2x2) and A2 (2x1)
    - Split B into B1 (2x10) and B2 (1x10)
    - Result = A1*B1 + A2*B2
    
    For A1*B1: use Strassen with rank 7, giving 7*10/2 = 35 terms? No.
    
    Let me think differently. The tensor for 2x3 x 3x10 has rank at most 60.
    
    A better approach: Use the fact that 3 = 2+1, and apply a block decomposition.
    
    Actually, the known result is that R(2,3,10) <= 54 using a Strassen-like method.
    
    Here's the construction:
    - View the 3x3 middle as composed of blocks
    - Use the identity that for 2x3 x 3x10, we can achieve rank 54
    
    Let me use a direct Strassen-based construction:
    
    Split the 3 input columns into groups of sizes (2,1) and the 3 output rows similarly.
    
    For the 2x2 x 2x2 part, Strassen gives rank 7.
    For the cross terms, we handle them directly.
    
    Total rank = 7 (for 2x2 core) + additional terms for the 1-column/row extensions.
    
    Let me compute:
    - 2x2 x 2x2: rank 7
    - 2x1 x 1x2: rank 2 (direct)
    - 1x2 x 2x1: rank 2 (direct)  
    - 1x1 x 1x1: rank 1 (direct)
    
    But this doesn't directly apply to 2x3 x 3x10.
    
    Alternative: Use the formula R(n,m,p) <= R(n,m1,p1) + R(n,m2,p2) + ... 
    when we can partition appropriately.
    
    For 2x3 x 3x10, split the 3 into 2+1:
    R(2,3,10) <= R(2,2,10) + R(2,1,10) + R(1,2,10) + R(1,1,10)
    
    R(2,2,10): Use Strassen for 2x2, then multiply by 10 columns. Rank = 7*10/2? No.
    Actually R(2,2,p) = 7*ceil(p/2) for even p? No, that's not right either.
    
    Let me use a simpler bound: R(2,2,10) <= 7 * 5 = 35 (using Strassen on pairs of columns)
    R(2,1,10) = 2*1*10 = 20 (schoolbook, no Strassen possible)
    R(1,2,10) = 1*2*10 = 20
    R(1,1,10) = 1*1*10 = 10
    
    Total: 35 + 20 + 20 + 10 = 85. That's worse than 60.
    
    This approach doesn't work well. Let me try a different decomposition.
    
    Better approach: Use the fact that we can apply Strassen to the 2x2 part of the 3x3.
    
    Let A be 2x3, B be 3x10.
    Write A = [A1 | a] where A1 is 2x2, a is 2x1.
    Write B = [b; B1] where b is 1x10, B1 is 2x10.
    
    Then AB = A1*B1 + a*b
    
    For A1*B1 (2x2 x 2x10):
    We can use Strassen-like methods. R(2,2,10) can be done in rank 35.
    
    For a*b (2x1 x 1x10):
    This is an outer product, rank = 2*10 = 20? No, it's just element-wise, rank 10.
    Actually, a is 2x1 and b is 1x10, so a*b is 2x10. This requires 2*10 = 20 scalar multiplications.
    
    Total: 35 + 20 = 55.
    
    So we can achieve rank 55!
    
    Let me construct this:
    
    For R(2,2,10) = 35:
    Split the 10 columns into 5 pairs. For each pair, use Strassen (rank 7).
    Total: 5 * 7 = 35.
    
    For the a*b part: 20 terms.
    
    Total rank: 55.
    """
    
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # We'll construct rank-55 decomposition
    # Split: A (2x3) = [A1 (2x2) | a (2x1)]
    #         B (3x10) = [b (1x10); B1 (2x10)]
    # AB = A1*B1 + a*b
    #
    # A1*B1: 2x2 times 2x10, rank 35 (5 Strassen blocks of rank 7)
    # a*b: 2x1 times 1x10, rank 20
    #
    # Total rank: 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 3*10 = 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 10*2 = 20 rows
    
    col = 0
    
    # Part 1: A1*B1 where A1 is 2x2 (indices 0,1 in A's columns), B1 is 2x10 (indices 0,1 in B's rows)
    # A1*B1: (2x2) * (2x10)
    # Split 10 columns into 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    # For each pair, use Strassen with rank 7
    
    # Strassen for 2x2 x 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x2 x 2x2k, we can do it in 7k rank.
    # For k=1 (2x2 x 2x2), rank 7.
    # For 2x2 x 2x10, split into 5 blocks of 2x2 x 2x2, each rank 7, total 35.
    
    for block in range(5):
        # Columns 2*block and 2*block+1 in the 10-column space
        c0 = 2 * block
        c1 = 2 * block + 1
        
        # Strassen factors for this block
        # M1 = (A11+A22)*(B11+B22)  [using 1-indexed for A1 rows, B1 cols]
        # In our indexing: A1 has rows 0,1 (of original A), cols 0,1 (of original A's 3 cols)
        # B1 has rows 0,1 (of original B's 3 rows), cols c0,c1 (of original B's 10 cols)
        
        # U indices: i*m+j where i in {0,1} (A's rows), j in {0,1} (A's first 2 cols)
        # V indices: j*p+k where j in {0,1} (B's first 2 rows), k in {c0,c1}
        # W indices: k*n+i where k in {c0,c1} (B's cols), i in {0,1} (A's rows)
        
        # M1: (A[0,0]+A[1,1]) * (B[0,c0]+B[1,c1])
        U[0*3+0][col] = 1  # A[0,0]
        U[1*3+1][col] = 1  # A[1,1]
        V[0*10+c0][col] = 1  # B[0,c0]
        V[1*10+c1][col] = 1  # B[1,c1]
        W[c0*2+0][col] = 1  # C[0,c0]
        W[c1*2+1][col] = 1  # C[1,c1]
        col += 1
        
        # M2: (A[1,0]+A[1,1]) * B[0,c0]
        U[1*3+0][col] = 1  # A[1,0]
        U[1*3+1][col] = 1  # A[1,1]
        V[0*10+c0][col] = 1  # B[0,c0]
        W[c0*2+1][col] = 1  # C[1,c0]
        col += 1
        
        # M3: A[0,0] * (B[0,c1]-B[1,c1])
        U[0*3+0][col] = 1  # A[0,0]
        V[0*10+c1][col] = 1  # B[0,c1]
        V[1*10+c1][col] = -1  # -B[1,c1]
        W[c1*2+0][col] = 1  # C[0,c1]
        col += 1
        
        # M4: A[1,1] * (B[1,c0]-B[0,c0])
        U[1*3+1][col] = 1  # A[1,1]
        V[1*10+c0][col] = 1  # B[1,c0]
        V[0*10+c0][col] = -1  # -B[0,c0]
        W[c0*2+1][col] = 1  # C[1,c0]
        col += 1
        
        # M5: (A[0,0]+A[0,1]) * B[1,c1]
        U[0*3+0][col] = 1  # A[0,0]
        U[0*3+1][col] = 1  # A[0,1]
        V[1*10+c1][col] = 1  # B[1,c1]
        W[c1*2+1][col] = 1  # C[1,c1]
        col += 1
        
        # M6: (A[1,0]-A[0,0]) * (B[0,c0]+B[0,c1])
        U[1*3+0][col] = 1  # A[1,0]
        U[0*3+0][col] = -1  # -A[0,0]
        V[0*10+c0][col] = 1  # B[0,c0]
        V[0*10+c1][col] = 1  # B[0,c1]
        W[c0*2+1][col] = 1  # C[1,c0]
        W[c1*2+1][col] = 1  # C[1,c1]
        col += 1
        
        # M7: (A[0,1]-A[1,1]) * (B[1,c0]+B[1,c1])
        U[0*3+1][col] = 1  # A[0,1]
        U[1*3+1][col] = -1  # -A[1,1]
        V[1*10+c0][col] = 1  # B[1,c0]
        V[1*10+c1][col] = 1  # B[1,c1]
        W[c0*2+0][col] = 1  # C[0,c0]
        W[c1*2+0][col] = 1  # C[0,c1]
        col += 1
    
    # Now we need to verify the Strassen reconstruction:
    # C[0,c0] = M1 + M4 - M5 + M7
    # C[0,c1] = M3 + M5
    # C[1,c0] = M2 + M4
    # C[1,c1] = M1 - M2 + M3 + M6
    
    # Let me check if my W assignments are correct:
    # For block with cols c0, c1:
    # C[0,c0] should get contributions from M1, M4, -M5, M7
    #   M1: W[c0*2+0] += 1  (done)
    #   M4: W[c0*2+1] += 1  (wrong! M4 contributes to C[1,c0], not C[0,c0])
    
    # I made an error. Let me redo this more carefully.
    
    # Actually, let me restart with a cleaner approach. I'll verify the Strassen formulas.
    
    # For 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A*B = [[c00, c01], [c10, c11]]
    #
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    #
    # Strassen:
    # M1 = (a00+a11)*(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # M2 = (a10+a11)*b00 = a10*b00 + a11*b00
    # M3 = a00*(b01-b11) = a00*b01 - a00*b11
    # M4 = a11*(b10-b00) = a11*b10 - a11*b00
    # M5 = (a00+a01)*b11 = a00*b11 + a01*b11
    # M6 = (a10-a00)*(b00+b01) = a10*b00 + a10*b01 - a00*b00 - a00*b01
    # M7 = (a01-a11)*(b10+b11) = a01*b10 + a01*b11 - a11*b10 - a11*b11
    #
    # c00 = M1 + M4 - M5 + M7
    #   = (a00*b00 + a00*b11 + a11*b00 + a11*b11) 
    #   + (a11*b10 - a11*b00) 
    #   - (a00*b11 + a01*b11) 
    #   + (a01*b10 + a01*b11 - a11*b10 - a11*b11)
    #   = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b10 - a11*b00 
    #   - a00*b11 - a01*b11 + a01*b10 + a01*b11 - a11*b10 - a11*b11
    #   = a00*b00 + a01*b10  ✓
    #
    # c01 = M3 + M5
    #   = (a00*b01 - a00*b11) + (a00*b11 + a01*b11)
    #   = a00*b01 + a01*b11  ✓
    #
    # c10 = M2 + M4
    #   = (a10*b00 + a11*b00) + (a11*b10 - a11*b00)
    #   = a10*b00 + a11*b10  ✓
    #
    # c11 = M1 - M2 + M3 + M6
    #   = (a00*b00 + a00*b11 + a11*b00 + a11*b11) 
    #   - (a10*b00 + a11*b00) 
    #   + (a00*b01 - a00*b11) 
    #   + (a10*b00 + a10*b01 - a00*b00 - a00*b01)
    #   = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a10*b00 - a11*b00 
    #   + a00*b01 - a00*b11 + a10*b00 + a10*b01 - a00*b00 - a00*b01
    #   = a10*b01 + a11*b11  ✓
    
    # Good, the Strassen formulas are correct. Now let me fix my W assignments.
    
    # I need to reset and redo.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: A1*B1 (2x2 * 2x10) using 5 Strassen blocks
    for block in range(5):
        c0 = 2 * block
        c1 = 2 * block + 1
        
        # M1: (a00+a11) * (b00+b11)
        # U: A[0,0]=1, A[1,1]=1
        # V: B[0,c0]=1, B[1,c1]=1
        # W: contributes to C[0,c0], C[1,c1]
        # C[0,c0] += M1, C[1,c1] += M1
        U[0*3+0][col] = 1   # a00
        U[1*3+1][col] = 1   # a11
        V[0*10+c0][col] = 1 # b00
        V[1*10+c1][col] = 1 # b11
        W[c0*2+0][col] = 1  # c00 += M1
        W[c1*2+1][col] = 1  # c11 += M1
        col += 1
        
        # M2: (a10+a11) * b00
        # U: A[1,0]=1, A[1,1]=1
        # V: B[0,c0]=1
        # W: C[1,c0] += M2
        U[1*3+0][col] = 1   # a10
        U[1*3+1][col] = 1   # a11
        V[0*10+c0][col] = 1 # b00
        W[c0*2+1][col] = 1  # c10 += M2
        col += 1
        
        # M3: a00 * (b01-b11)
        # U: A[0,0]=1
        # V: B[0,c1]=1, B[1,c1]=-1
        # W: C[0,c1] += M3
        U[0*3+0][col] = 1   # a00
        V[0*10+c1][col] = 1 # b01
        V[1*10+c1][col] = -1 # -b11
        W[c1*2+0][col] = 1  # c01 += M3
        col += 1
        
        # M4: a11 * (b10-b00)
        # U: A[1,1]=1
        # V: B[1,c0]=1, B[0,c0]=-1
        # W: C[1,c0] += M4
        U[1*3+1][col] = 1   # a11
        V[1*10+c0][col] = 1 # b10
        V[0*10+c0][col] = -1 # -b00
        W[c0*2+1][col] = 1  # c10 += M4
        col += 1
        
        # M5: (a00+a01) * b11
        # U: A[0,0]=1, A[0,1]=1
        # V: B[1,c1]=1
        # W: C[1,c1] += M5, but with negative sign in c00 formula
        # c00 -= M5, c11 += M5
        U[0*3+0][col] = 1   # a00
        U[0*3+1][col] = 1   # a01
        V[1*10+c1][col] = 1 # b11
        W[c0*2+0][col] = -1 # c00 -= M5
        W[c1*2+1][col] = 1  # c11 += M5
        col += 1
        
        # M6: (a10-a00) * (b00+b01)
        # U: A[1,0]=1, A[0,0]=-1
        # V: B[0,c0]=1, B[0,c1]=1
        # W: C[1,c0] += M6, C[1,c1] += M6
        U[1*3+0][col] = 1   # a10
        U[0*3+0][col] = -1  # -a00
        V[0*10+c0][col] = 1 # b00
        V[0*10+c1][col] = 1 # b01
        W[c0*2+1][col] = 1  # c10 += M6
        W[c1*2+1][col] = 1  # c11 += M6
        col += 1
        
        # M7: (a01-a11) * (b10+b11)
        # U: A[0,1]=1, A[1,1]=-1
        # V: B[1,c0]=1, B[1,c1]=1
        # W: C[0,c0] += M7, C[0,c1] += M7
        U[0*3+1][col] = 1   # a01
        U[1*3+1][col] = -1  # -a11
        V[1*10+c0][col] = 1 # b10
        V[1*10+c1][col] = 1 # b11
        W[c0*2+0][col] = 1  # c00 += M7
        W[c1*2+0][col] = 1  # c01 += M7
        col += 1
    
    # Part 2: a*b where a is A[*,2] (2x1) and b is B[2,*] (1x10)
    # a*b: outer product, requires 2*10 = 20 terms
    # C[i,k] += a[i]*b[k] for i in {0,1}, k in {0,...,9}
    for i in range(2):
        for k in range(10):
            U[i*3+2][col] = 1   # a[i] = A[i,2]
            V[2*10+k][col] = 1  # b[k] = B[2,k]
            W[k*2+i][col] = 1   # C[i,k] += a[i]*b[k]
            col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
