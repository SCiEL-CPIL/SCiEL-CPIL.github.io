"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (24, rank)
    
    # Indexing convention:
    # U[i*m + j, r] corresponds to A[i,j]
    # V[j*p + k, r] corresponds to B[j,k]
    # W[k*n + i, r] corresponds to C[i,k] (trace convention)
    
    # Strategy:
    # 1. Implement a correct Strassen-based decomposition for 4x4x4 (rank 28) and 4x4x2 (rank 16).
    #    Total rank = 28 + 16 = 44.
    #    Wait, standard Strassen for 4x4x4 gives rank 28?
    #    Let's check: 4x4x4 can be seen as 4 2x2x2 multiplications? No, it's 4 blocks of C, each is sum of 2 products.
    #    If we use Strassen for each product, we get 4 * (7+7) = 56?
    #    Actually, we can group them.
    #    C00 = A00B00 + A01B10
    #    C01 = A00B01 + A01B11
    #    C10 = A10B00 + A11B10
    #    C11 = A10B01 + A11B11
    #    
    #    Let's use a known rank-28 decomposition for 4x4x4?
    #    Actually, the rank of 4x4x4 is known to be 21 (Lader et al.), but that's complex to hardcode.
    #    A simpler structured decomposition:
    #    Use Strassen for 2x2x2. Rank 7.
    #    For 4x4x4, we can view it as 2x2 blocks of 2x2.
    #    There are 4 such 2x2x2 sub-tensors if we don't share?
    #    No, C00 involves A00B00 and A01B10.
    #    If we compute A00B00 (rank 7), A01B10 (rank 7), sum them -> rank 14 for C00.
    #    Total 4 * 14 = 56.
    #    
    #    Can we do better?
    #    Yes, by sharing the "multiplications" M1..M7 across blocks.
    #    This is the standard "Strassen for 4x4" which has rank 28?
    #    Let's derive it or find a reliable source.
    #    Actually, a common result is rank(4,4,4) <= 28 via Strassen recursion?
    #    Strassen recursion: M(4,4,4) = 7 * M(2,2,2) = 7 * 7 = 49? No, that's for square matrices where you split into 4 blocks and do 7 multiplications of half-size matrices.
    #    M(2,2,2) = 7.
    #    M(4,4,4) <= 7 * 7 = 49.
    #    M(8,8,8) <= 7 * 49 = 343.
    #    So 49 is a valid upper bound for 4x4x4 using Strassen recursion.
    #    But we can do better.
    #    Actually, the "Strassen algorithm" for 4x4 matrices uses 28 scalar multiplications?
    #    Let's check.
    #    If we split 4x4 into 2x2 blocks, we have 4 A-blocks, 4 B-blocks.
    #    We need to compute 4 C-blocks.
    #    C00 = A00B00 + A01B10
    #    C01 = A00B01 + A01B11
    #    C10 = A10B00 + A11B10
    #    C11 = A10B01 + A11B11
    #    
    #    If we use Strassen for the 2x2x2 multiplication of the *blocks*, we treat the blocks as "scalars".
    #    The "scalar" multiplication is 2x2x2, which costs 7.
    #    The block-level algorithm is 2x2x2, which costs 7 block-multiplications.
    #    Total cost = 7 * 7 = 49.
    #    
    #    So rank(4,4,4) <= 49.
    #    
    #    For 4x4x2:
    #    A is 4x4, B is 4x2, C is 4x2.
    #    Split into 2x2 blocks?
    #    A: 2x2 blocks of 2x2.
    #    B: 2x1 blocks of 2x2? No, B is 4x2.
    #    Let's split A into 2x2 blocks. B into 2x2 blocks (but B has width 2, so 1 block column).
    #    So A has 4 blocks (2x2 grid), B has 2 blocks (2x1 grid).
    #    C has 2 blocks (2x1 grid).
    #    C_top = A_top * B_top
    #    C_bot = A_bot * B_bot
    #    Where A_top is 2x4, B_top is 4x2 -> C_top is 2x2.
    #    This is not a simple 2x2x2.
    #    
    #    Alternative:
    #    4x4x2 can be seen as two 4x4x1 multiplications.
    #    rank(4,4,1) = 16 (since it's a linear map A -> C, C = A*b, which is 16-dimensional).
    #    So rank(4,4,2) <= 16 + 16 = 32.
    #    
    #    Total rank = 49 + 32 = 81.
    #    81 < 96. This is an improvement.
    #    
    #    Let's implement this.
    #    1. Strassen 4x4x4 (rank 49).
    #    2. Schoolbook 4x4x2 (rank 32).
    #    Total 81.
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook fallback
    U_school = np.zeros((16, 96), dtype=int)
    V_school = np.zeros((24, 96), dtype=int)
    W_school = np.zeros((24, 96), dtype=int)
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                U_school[i*4+j, col] = 1
                V_school[j*6+k, col] = 1
                W_school[k*4+i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = 96
    
    # Implement Strassen 4x4x4
    # We will use the recursive Strassen approach.
    # This requires a function to add a 2x2x2 Strassen decomposition with offsets.
    
    U_4x4x4 = []
    V_4x4x4 = []
    W_4x4x4 = []
    
    def add_strassen_2x2x2(u_offset_i, u_offset_j, v_offset_i, v_offset_j, w_offset_i, w_offset_j, scale=1):
        # This adds 7 terms for a 2x2x2 multiplication.
        # u_offset_i, u_offset_j: global row/col offset for A block
        # v_offset_i, v_offset_j: global row/col offset for B block
        # w_offset_i, w_offset_j: global row/col offset for C block
        # Note: W index is k*n + i. So w_offset_i is the 'k' offset (col of C), w_offset_j is the 'i' offset (row of C).
        # Wait, the local indices for C in 2x2x2 are (row, col).
        # In the global W matrix, index is col*4 + row.
        # So if local C is (r, c), global is (w_offset_c + c)*4 + (w_offset_r + r).
        
        # M1 = (A00+A11)(B00+B11)
        # A00: (u_offset_i, u_offset_j)
        # A11: (u_offset_i+1, u_offset_j+1)
        # B00: (v_offset_i, v_offset_j)
        # B11: (v_offset_i+1, v_offset_j+1)
        # C00: (w_offset_i, w_offset_j)
        # C11: (w_offset_i+1, w_offset_j+1)
        
        # Term 1: A00*B00
        U_4x4x4.append((u_offset_i*4 + u_offset_j, scale))
        V_4x4x4.append((v_offset_i*6 + v_offset_j, scale))
        W_4x4x4.append((w_offset_i*4 + w_offset_j, scale))
        
        # Term 2: A00*B11
        U_4x4x4.append((u_offset_i*4 + u_offset_j, scale))
        V_4x4x4.append((v_offset_i*6 + v_offset_j + 1, scale))
        W_4x4x4.append((w_offset_i*4 + w_offset_j + 1, scale))
        
        # Term 3: A11*B00
        U_4x4x4.append((u_offset_i*4 + u_offset_j + 1, scale))
        V_4x4x4.append((v_offset_i*6 + v_offset_j, scale))
        W_4x4x4.append((w_offset_i*4 + w_offset_j + 1, scale))
        
        # Term 4: A11*B11
        U_4x4x4.append((u_offset_i*4 + u_offset_j + 1, scale))
        V_4x4x4.append((v_offset_i*6 + v_offset_j + 1, scale))
        W_4x4x4.append((w_offset_i*4 + w_offset_j + 1, scale))
        
        # M2 = (A10+A11)B00
        # A10: (u_offset_i+1, u_offset_j)
        # A11: (u_offset_i+1, u_offset_j+1)
        # B00: (v_offset_i, v_offset_j)
        # C10: (w_offset_i+1, w_offset_j)
        
        # Term 5: A10*B00
        U_4x4x4.append((u_offset_i*4 + u_offset_j + 1, scale))
        V_4x4x4.append((v_offset_i*6 + v_offset_j, scale))
        W_4x4x4.append((w_offset_i*4 + w_offset_j + 1, scale))
        
        # Term 6: A11*B00 (already have this as Term 3? No, Term 3 was A11*B00 for M1. M2 needs A11*B00 for C10? No, M2 contributes to C10 and C11?
        # Let's re-verify Strassen formulas.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M5 + M7
        
        # My previous manual expansion was getting messy. Let's use a cleaner recursive approach.
        # Instead of hardcoding the 49 terms, let's just use the schoolbook for 4x4x4 (rank 64) and schoolbook for 4x4x2 (rank 32).
        # Total 96. That's not an improvement.
        
        # Let's try a different approach:
        # Use Strassen for 2x2x2 blocks directly.
        # 4x4x4 consists of 4 2x2x2 blocks? No.
        # It consists of 16 1x1x1 blocks? No.
        
        # Let's stick to the simplest valid improvement:
        # Use Strassen for 2x2x2 (rank 7).
        # Decompose 4x4x6 into 4x4x2 + 4x4x2 + 4x4x2.
        # Each 4x4x2 is rank 32 (schoolbook). Total 96.
        
        # What if we use Strassen for 4x4x4?
        # The rank is 49.
        # Let's implement the 49-term Strassen decomposition for 4x4x4.
        # I will generate the terms programmatically using the recursive definition.
        
    # Let's abandon the manual Strassen implementation for 4x4x4 due to complexity and error risk.
    # Instead, let's use a known lower rank decomposition if I can find one quickly.
    # Or, let's use the fact that 4x4x6 can be decomposed as 2x4x6 + 2x4x6? No.
    
    # Let's try a random search for rank 50-95 with a better initialization.
    # The previous random search was too sparse.
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    val = np.dot(U[i*4+j], np.multiply(V[j*6+k], W[k*4+i]))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a decomposition with rank < 96.
    # Use a structured random search:
    # Initialize U, V, W with random values from {-1, 0, 1}.
    # Then use a simple gradient descent or coordinate descent to minimize the error?
    # Given the time limit, random search is safer.
    
    np.random.seed(42)
    for target_rank in range(50, 96):
        if time.time() > deadline:
            break
        for attempt in range(100):
            if time.time() > deadline:
                break
            U = np.zeros((16, target_rank))
            V = np.zeros((24, target_rank))
            W = np.zeros((24, target_rank))
            
            # Initialize with random sparse patterns
            for r in range(target_rank):
                i = np.random.randint(0, 4)
                j = np.random.randint(0, 4)
                k = np.random.randint(0, 6)
                
                U[i*4+j, r] = 1
                V[j*6+k, r] = 1
                W[k*4+i, r] = 1
                
                # Add more terms to increase density
                for _ in range(5):
                    i2 = np.random.randint(0, 4)
                    j2 = np.random.randint(0, 4)
                    k2 = np.random.randint(0, 6)
                    U[i2*4+j2, r] += np.random.choice([-1, 0, 1])
                    V[j2*6+k2, r] += np.random.choice([-1, 0, 1])
                    W[k2*4+i2, r] += np.random.choice([-1, 0, 1])
            
            if verify(U, V, W):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break
        if best_rank < 96:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
