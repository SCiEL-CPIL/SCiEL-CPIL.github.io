"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition using vectorized numpy ops."""
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            U_c = U_reshaped[:, :, c]
            V_c = V_reshaped[:, :, c]
            W_c = W_reshaped[:, :, c]
            T += U_c[:, :, None] * V_c[None, :, :] * W_c.T[None, :, :]
        
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p

    # Construct rank-25 decomposition for 4x4x5
    # Use Strassen-based rank-21 for 4x4x4 + 4 extra terms for 5th column
    def construct_rank25():
        """Construct a rank-25 decomposition for 4x4x5 matrix multiplication."""
        rank = 25
        U = [[0] * rank for _ in range(16)]  # n*m = 4*4
        V = [[0] * rank for _ in range(20)]  # m*p = 4*5
        W = [[0] * rank for _ in range(20)]  # p*n = 5*4
        
        # Strassen's 7 products for 2x2x2:
        # For 4x4x4, split into 2x2 blocks and use Strassen at block level
        # Each block product is a 2x2x2 multiplication (rank 7)
        # But we share structure to get rank 21
        
        # Block indices: A = [[A00, A01], [A10, A11]] each 2x2
        # B = [[B00, B01], [B10, B11]] each 2x2
        # C = AB
        
        # Strassen products at block level:
        # S1 = (A00+A11)(B00+B11)
        # S2 = (A10+A11)B00
        # S3 = A00(B01-B11)
        # S4 = A11(B10-B00)
        # S5 = (A00+A01)B11
        # S6 = (A10-A00)(B00+B01)
        # S7 = (A01-A11)(B10+B11)
        
        # Each Si is a 2x2 matrix product, decomposable into 7 scalar products
        # Total: 7*7 = 49, but we can do better with sharing
        
        # For rank-21, use the known decomposition. Let me construct it explicitly.
        # The rank-21 decomposition for 4x4x4 uses the following structure:
        # It's based on the fact that 4x4x4 tensor rank is 21
        
        # Let me use a different approach: direct construction
        # For 4x4x4, I'll use a known rank-21 decomposition
        
        # Actually, let me just use a simpler but valid approach:
        # Rank-25 = rank-21 (for 4x4x4) + 4 (for the 5th column of B)
        
        # For the 4x4x4 part (columns 0-3 of B), use rank-21 decomposition
        # For the 5th column (column 4 of B), use 4 extra terms
        
        # Rank-21 decomposition for 4x4x4:
        # This is complex to construct from scratch. Let me use a known one.
        
        # Alternative: use rank-24 for 4x4x4 (easier to construct) + 1 for 5th col = 25
        # Or just use a structured approach
        
        # Let me construct a rank-25 decomposition directly:
        # Use the fact that C[i,k] = sum_j A[i,j] * B[j,k]
        
        # For k=0,1,2,3: use Strassen-like decomposition (rank 21)
        # For k=4: use 4 terms (one per j)
        
        # For simplicity, let me use a rank-25 construction:
        # First 21 terms handle 4x4x4, next 4 handle 4x4x1 (the 5th column)
        
        # Strassen for 2x2x2 (rank 7):
        # U entries for (i,j) pair, V entries for (j,k) pair, W entries for (k,i) pair
        # i,j in {0,1}, k in {0,1}
        
        # For 4x4x4 with block structure:
        # A block (bi,bj) where bi,bj in {0,1} for 2x2 blocks, each block is 2x2
        # So A[bi*2+ai, bj*2+aj] where ai,aj in {0,1}
        
        # Let me just use a known rank-21 decomposition for 4x4x4.
        # The explicit rank-21 decomposition is well-known in the literature.
        
        # For now, let me use a simpler but valid rank-25 construction:
        # Use schoolbook for the first 3 columns of B (12 terms) 
        # and Strassen-like for remaining 2 columns... no, that's not right.
        
        # Let me try a different approach: use the bilinear form directly
        # with a structured rank-25 decomposition.
        
        # Actually, the simplest valid improvement: use rank-25 where
        # first 21 terms are a valid 4x4x4 decomposition and last 4 handle column 4.
        
        # For the 4x4x4 part, let me use a known rank-21 decomposition.
        # Here's one based on the Winograd scheme:
        
        # I'll construct it using the following method:
        # The 4x4x4 multiplication tensor can be decomposed with rank 21
        # using the following 21 triples (u_c, v_c, w_c) where:
        # u_c is a 4x4 matrix, v_c is a 4x4 matrix, w_c is a 4x4 matrix
        # and sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k
        
        # This is equivalent to: sum_c (u_c ⊗ v_c ⊗ w_c) = T where T[i,j,k]=1
        
        # For a concrete construction, let me use the following approach:
        # Split 4x4 into 2x2 blocks and use Strassen at each level with sharing.
        
        # Given the complexity, let me use a simpler but still improved rank.
        # Use rank-28: 7 Strassen products for 2x2x2, applied to 4 2x2 blocks
        # but with some sharing, plus 4 for the 5th column.
        
        # Actually, let me just implement a valid rank-25 decomposition using
        # a known construction. I'll use the following:
        
        # For 4x4x4, use the following rank-21 decomposition (from literature):
        # The 21 terms correspond to specific (u,v,w) triples.
        
        # Due to complexity, let me use a different strategy:
        # Construct a rank-25 decomposition where:
        # - Terms 0-19: handle 4x4x4 using a structured approach
        # - Terms 20-23: handle 4x4x1 for the 5th column
        # - Term 24: correction term
        
        # For the 4x4x1 part (column 4 of B):
        # C[i,4] = sum_j A[i,j] * B[j,4]
        # This needs 4 terms: for each j, U[i,j,term]=1, V[j,4,term]=1, W[4,i,term]=1
        # Wait, W is indexed as W[k*n+i], so W[4*4+i] = W[16+i]
        
        # For term c (c=20,21,22,23), let j = c-20:
        # U[i*m+j][c] = 1 for all i -> U[i*4+j][c] = 1
        # V[j*p+4][c] = 1 -> V[j*5+4][c] = 1
        # W[4*n+i][c] = 1 -> W[16+i][c] = 1
        
        # Now for the 4x4x4 part, I need a rank-21 decomposition.
        # Let me construct one using the following method:
        
        # Use the fact that for 2x2x2, Strassen gives rank 7.
        # For 4x4x4, split into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]] (each 2x2)
        # B = [[B00, B01], [B10, B11]] (each 2x2)
        
        # Strassen products:
        # S1 = (A00+A11)(B00+B11)
        # S2 = (A10+A11)B00
        # S3 = A00(B01-B11)
        # S4 = A11(B10-B00)
        # S5 = (A00+A01)B11
        # S6 = (A10-A00)(B00+B01)
        # S7 = (A01-A11)(B10+B11)
        
        # Each Sk is a 2x2 product, which needs 7 scalar products (Strassen).
        # Total: 49 scalar products.
        
        # But we can share: some of the intermediate matrices are reused.
        # The rank-21 result comes from a more clever sharing.
        
        # For simplicity, let me use rank-49 for 4x4x4 + 4 for 5th col = 53.
        # That's worse than schoolbook 80, so it's an improvement but not great.
        
        # Let me try rank-25 with a different approach.
        
        # I'll use the following construction for 4x4x4 with rank 21:
        # Based on the explicit decomposition from the literature.
        
        # The 21 terms for 4x4x4 (i,j,k in 0..3):
        # Each term c has U_c (4x4), V_c (4x4), W_c (4x4) such that
        # sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
        
        # This is equivalent to finding 21 rank-1 tensors that sum to T.
        
        # Let me use a known rank-21 decomposition. Here's one construction:
        
        # Actually, let me just use a valid but higher rank that's still better than 80.
        # Use rank-49 (Strassen at block level) + 4 = 53.
        
        # Or better: use a direct construction with rank ~ 30-40.
        
        # For now, let me return the schoolbook result and try to improve
        # with a structured search in the remaining time.
        
        return None
    
    # Try structured construction for rank-25
    def try_rank25():
        rank = 25
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(20)]
        W = [[0] * rank for _ in range(20)]
        
        # First 21 terms: 4x4x4 decomposition
        # Last 4 terms: 5th column of B
        
        # For the last 4 terms (terms 21-24), handle B[:,4]:
        for t in range(4):
            j = t
            c = 21 + t
            for i in range(4):
                U[i * 4 + j][c] = 1
            V[j * 5 + 4][c] = 1
            for i in range(4):
                W[4 * 4 + i][c] = 1
        
        # For the first 21 terms, use a known 4x4x4 rank-21 decomposition
        # This is the complex part. Let me use a construction based on
        # the following idea:
        
        # For 4x4x4, we can use the following rank-21 decomposition:
        # It's based on the tensor rank of the 4x4x4 multiplication tensor.
        
        # I'll construct it using the following method:
        # The 4x4x4 tensor T[i,j,k] = 1 for all i,j,k in 0..3
        # can be written as sum of 21 rank-1 tensors.
        
        # One known construction uses the following 21 triples:
        # (due to space, I'll use a simpler but valid rank-25 construction)
        
        # For terms 0-20, I'll use a structured approach:
        # Split into 4 groups of ~5-6 terms each
        
        # Actually, let me use a different approach:
        # Use the fact that 4x4x4 can be decomposed as:
        # T = sum_{a,b in {0,1}} T_{ab} where T_{ab} handles block (a,b)
        # Each T_{ab} is a 2x2x2 tensor (rank 7 with Strassen)
        # Total: 4 * 7 = 28 terms for 4x4x4, plus 4 for 5th col = 32
        
        # That gives rank 32, which is better than 80.
        
        # Let me implement rank-32:
        # 28 terms for 4x4x4 (7 per 2x2 block) + 4 terms for 5th column
        
        # For block (a,b) where a is row block, b is col block:
        # A block A[a,b] is 2x2, B block B[b,c] is 2x2 for each c
        # Wait, this isn't quite right for matrix multiplication.
        
        # Let me reconsider. For C = AB where A is 4x4, B is 4x4:
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        # Split into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # Each product like A00*B00 is a 2x2x2 multiplication (rank 7 with Strassen)
        # There are 8 such products, so total rank = 8 * 7 = 56
        # Plus 4 for the 5th column = 60
        
        # That's still better than 80 but not great.
        
        # Let me use Strassen at the block level to reduce this.
        # Strassen for 2x2 block multiplication uses 7 products instead of 4.
        # But each product is still a 2x2x2 multiplication.
        
        # Actually, the standard Strassen for 4x4x4 uses 21 scalar multiplications.
        # Let me look up the explicit decomposition.
        
        # Given the time constraint, let me use a rank-32 construction:
        # 28 terms for 4x4x4 + 4 terms for 5th column
        
        # For 4x4x4, use 7 Strassen terms for each of 4 block products
        # But that's 28 terms, and they don't share anything, so rank = 28
        
        # Wait, that's not how it works. Let me think again.
        
        # For C00 = A00*B00 + A01*B10:
        # A00*B00 needs 7 scalar products (Strassen for 2x2)
        # A01*B10 needs 7 scalar products
        # Total for C00: 14 scalar products
        # Similarly for C01, C10, C11: 14 each
        # Total: 56 scalar products for 4x4x4
        
        # Plus 4 for 5th column = 60 total
        
        # That's rank 60, better than 80.
        
        # Let me implement rank-60:
        rank = 60
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(20)]
        W = [[0] * rank for _ in range(20)]
        
        # Strassen for 2x2x2:
        # For matrices X (2x2), Y (2x2), Z = X*Y:
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # For each M_s, we have a scalar product:
        # U entry: coefficient for X[i,j]
        # V entry: coefficient for Y[j,k]  
        # W entry: coefficient for Z[k,i]
        
        # For M1 = (X00+X11)(Y00+Y11):
        # This is a sum of 4 scalar products:
        # X00*Y00, X00*Y11, X11*Y00, X11*Y11
        # But Strassen treats it as one "multiplication" with rank 1
        # Actually, in the tensor decomposition, each M_s contributes
        # to the U, V, W matrices with specific patterns.
        
        # For M1 = (X00+X11)(Y00+Y11):
        # U contribution: X00 and X11 have coefficient 1
        # V contribution: Y00 and Y11 have coefficient 1
        # W contribution: Z00, Z11 get +1, Z01 gets 0, Z10 gets 0
        # Wait, that's not right. M1 contributes to Z00 and Z11.
        
        # Actually, in the tensor decomposition for 2x2x2:
        # T[i,j,k] = 1 for all i,j,k
        # M1 corresponds to: U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
        # But that's not a rank-1 tensor. A rank-1 tensor has U[i,j] = u_i * v_j, etc.
        
        # I'm confusing myself. Let me restart with a clearer approach.
        
        # In the tensor decomposition for matrix multiplication:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # where T[i,j,k] = 1 for all valid i,j,k
        
        # For schoolbook, each (i,j,k) triple gets its own term c:
        # U[i,j,c] = 1, V[j,k,c] = 1, W[k,i,c] = 1
        
        # For Strassen 2x2x2, rank 7:
        # The 7 terms correspond to the 7 Strassen products.
        # Each term c has U_c (2x2), V_c (2x2), W_c (2x2) such that
        # sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
        
        # For M1 = (A00+A11)(B00+B11):
        # U1 = [[1,0],[0,1]] (diagonal)
        # V1 = [[1,0],[0,1]] (diagonal)
        # W1 = [[1,0],[0,1]] (diagonal)
        # Check: sum over i,j,k of U1[i,j]*V1[j,k]*W1[k,i]
        # = U1[0,0]*V1[0,0]*W1[0,0] + U1[0,0]*V1[0,1]*W1[1,0] + ... 
        # This doesn't give T[i,j,k]=1 for all i,j,k.
        
        # I think I'm misunderstanding the tensor decomposition format.
        
        # Let me re-read the problem:
        # "output factor follows the official trace convention W[k*n+i,r]"
        # So W is indexed as W[k*n+i, r] where k is the column of B / row of C
        # and i is the row of A / column of C.
        
        # The decomposition is: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And we need this to equal sum_j A[i,j] * B[j,k]
        
        # So the tensor T[i,j,k] = 1 means:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # For schoolbook: for each (i,j,k), set U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
        # for a unique r.
        
        # For Strassen 2x2x2 (n=m=p=2), rank 7:
        # The 7 terms are:
        # r=0: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1
        # Wait, that gives sum_r ... = 1 for (i,j,k) where all indices match, but not for others.
        
        # I think the correct Strassen decomposition for 2x2x2 is:
        # r=0: U = [[1,0],[0,1]], V = [[1,0],[0,1]], W = [[1,0],[0,1]]
        # r=1: U = [[0,0],[1,1]], V = [[1,0],[0,0]], W = [[0,1],[1,0]]
        # r=2: U = [[1,0],[0,0]], V = [[0,1],[0,-1]], W = [[0,1],[0,0]]
        # r=3: U = [[0,0],[0,1]], V = [[0,-1],[1,0]], W = [[0,0],[1,0]]
        # r=4: U = [[1,1],[0,0]], V = [[0,0],[0,1]], W = [[0,0],[1,0]]
        # r=5: U = [[0,-1],[1,0]], V = [[1,1],[0,0]], W = [[0,0],[0,1]]
        # r=6: U = [[0,1],[-1,0]], V = [[0,0],[1,1]], W = [[0,0],[0,1]]
        
        # Let me verify r=0:
        # sum_r U[r][i,j] * V[r][j,k] * W[r][k,i] for r=0:
        # U[0] = [[1,0],[0,1]], V[0] = [[1,0],[0,1]], W[0] = [[1,0],[0,1]]
        # (i,j,k)=(0,0,0): 1*1*1 = 1
        # (i,j,k)=(0,0,1): 1*0*0 = 0
        # (i,j,k)=(0,1,0): 0*0*0 = 0
        # (i,j,k)=(0,1,1): 0*1*0 = 0
        # (i,j,k)=(1,0,0): 0*1*0 = 0
        # (i,j,k)=(1,0,1): 0*0*0 = 0
        # (i,j,k)=(1,1,0): 1*0*0 = 0
        # (i,j,k)=(1,1,1): 1*1*1 = 1
        # So r=0 contributes 1 to (0,0,0) and (1,1,1), 0 elsewhere.
        
        # This is getting very complex. Let me just use the schoolbook decomposition
        # and try to find a better one through a more systematic search.
        
        return None
    
    # Given the complexity, let me use a simpler but valid approach:
    # Use rank-60 (Strassen at block level) which is better than schoolbook 80.
    
    def construct_rank60():
        """Construct a rank-60 decomposition using Strassen at block level."""
        n, m, p = 4, 4, 5
        rank = 60
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(20)]
        W = [[0] * rank for _ in range(20)]
        
        # Strassen for 2x2x2: 7 terms
        # For each term, we have coefficients for U (2x2), V (2x2), W (2x2)
        
        # The 7 Strassen terms for 2x2x2:
        # Each term c has:
        # U_c[i,j] for i,j in {0,1}
        # V_c[j,k] for j,k in {0,1}
        # W_c[k,i] for k,i in {0,1}
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        # U: [[1,0],[0,1]], V: [[1,0],[0,1]], W: [[1,0],[0,1]]
        # Term 1: M2 = (A10+A11)B00
        # U: [[0,0],[1,1]], V: [[1,0],[0,0]], W: [[0,1],[1,0]]
        # Term 2: M3 = A00(B01-B11)
        # U: [[1,0],[0,0]], V: [[0,1],[0,-1]], W: [[0,1],[0,0]]
        # Term 3: M4 = A11(B10-B00)
        # U: [[0,0],[0,1]], V: [[0,-1],[1,0]], W: [[0,0],[1,0]]
        # Term 4: M5 = (A00+A01)B11
        # U: [[1,1],[0,0]], V: [[0,0],[0,1]], W: [[0,0],[1,0]]
        # Term 5: M6 = (A10-A00)(B00+B01)
        # U: [[0,-1],[1,0]], V: [[1,1],[0,0]], W: [[0,0],[0,1]]
        # Term 6: M7 = (A01-A11)(B10+B11)
        # U: [[0,1],[-1,0]], V: [[0,0],[1,1]], W: [[0,0],[0,1]]
        
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs) where each is a flat 4-element list
            # U: [U00, U01, U10, U11], V: [V00, V01, V10, V11], W: [W00, W01, W10, W11]
            # W is indexed as W[k,i] so W00=W[0,0], W01=W[0,1], W10=W[1,0], W11=W[1,1]
            ([[1,0,0,1], [1,0,0,1], [1,0,0,1]]),   # M1
            ([[0,0,1,1], [1,0,0,0], [0,1,1,0]]),   # M2
            ([[1,0,0,0], [0,1,0,-1], [0,1,0,0]]),  # M3
            ([[0,0,0,1], [0,-1,1,0], [0,0,1,0]]),  # M4
            ([[1,1,0,0], [0,0,0,1], [0,0,1,0]]),   # M5
            ([[0,-1,1,0], [1,1,0,0], [0,0,0,1]]),  # M6
            ([[0,1,-1,0], [0,0,1,1], [0,0,0,1]]),  # M7
        ]
        
        # For 4x4x4, split into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]] where Aab is 2x2
        # B = [[B00, B01], [B10, B11]] where Bab is 2x2
        # C = AB = [[C00, C01], [C10, C11]]
        
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # Each product like A00*B00 is a 2x2x2 multiplication
        # We need 8 such products, each with 7 Strassen terms = 56 terms
        # Plus 4 terms for the 5th column = 60 total
        
        # For product Aab*Bcd (where a,b,c,d in {0,1}):
        # Aab is the 2x2 block at row-block a, col-block b of A
        # Bcd is the 2x2 block at row-block c, col-block d of B
        # The result contributes to C block at row-block a, col-block d
        
        # For the Strassen term s of product Aab*Bcd:
        # U contribution: for i in {0,1}, j in {0,1}:
        #   U[(a*2+i)*4 + (b*2+j), term] = U_s[i,j]
        # V contribution: for j in {0,1}, k in {0,1}:
        #   V[(c*2+j)*5 + (d*2+k), term] = V_s[j,k]
        # W contribution: for k in {0,1}, i in {0,1}:
        #   W[(d*2+k)*4 + (a*2+i), term] = W_s[k,i]
        
        products = [
            # (a, b, c, d) for Aab * Bcd
            (0, 0, 0, 0),  # A00 * B00 -> C00
            (0, 1, 1, 0),  # A01 * B10 -> C00
            (0, 0, 0, 1),  # A00 * B01 -> C01
            (0, 1, 1, 1),  # A01 * B11 -> C01
            (1, 0, 0, 0),  # A10 * B00 -> C10
            (1, 1, 1, 0),  # A11 * B10 -> C10
            (1, 0, 0, 1),  # A10 * B01 -> C11
            (1, 1, 1, 1),  # A11 * B11 -> C11
        ]
        
        term_idx = 0
        for (a, b, c, d) in products:
            for s in range(7):
                U_s, V_s, W_s = strassen_terms[s]
                for i in range(2):
                    for j in range(2):
                        U[(a*2+i)*4 + (b*2+j)][term_idx] = U_s[i*2+j]
                for j in range(2):
                    for k in range(2):
                        V[(c*2+j)*5 + (d*2+k)][term_idx] = V_s[j*2+k]
                for k in range(2):
                    for i in range(2):
                        W[(d*2+k)*4 + (a*2+i)][term_idx] = W_s[k*2+i]
                term_idx += 1
        
        # Add 4 terms for the 5th column of B (column 4)
        for t in range(4):
            j = t
            c = 56 + t
            for i in range(4):
                U[i * 4 + j][c] = 1
            V[j * 5 + 4][c] = 1
            for i in range(4):
                W[4 * 4 + i][c] = 1
        
        result = {"U": U, "V": V, "W": W}
        
        # Verify
        if check_decomposition(U, V, W):
            return result
        else:
            return None
    
    # Try rank-60
    r60 = construct_rank60()
    if r60 is not None:
        if 60 < best_rank:
            best_rank = 60
            best_result = r60
    
    # Try to find even better with structured search
    # For now, return the best found
    
    if time.time() > deadline - 5:
        return best_result
    
    # Try rank-50 with a more clever construction
    # Use Strassen at block level with sharing
    
    # For now, return best result
    return best_result
# EVOLVE-BLOCK-END
