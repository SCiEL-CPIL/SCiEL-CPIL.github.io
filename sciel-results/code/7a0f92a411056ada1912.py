import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Construct 4x4x4 decomposition using Strassen (Rank 49)
    # 2x2x2 Strassen decomposition
    # U (2x2), V (2x2), W (2x2)
    # Indices: i,j for U; j,k for V; k,i for W
    # 7 terms
    S = []
    # M1: (A11+A22)(B11+B22)
    # A11=(0,0), A22=(1,1), B11=(0,0), B22=(1,1)
    # Terms:
    # (0,0;0,0) (0,0;0,0) (0,0;0,0)
    # (0,0;0,0) (1,1;1,1) (0,0;0,0) ? No.
    # Strassen M1 = (A11+A22)(B11+B22)
    # A' = [[1,0],[0,1]] (A11+A22)
    # B' = [[1,0],[0,1]] (B11+B22)
    # This is a 2x2x2 multiplication of A' and B'.
    # We need the 7 rank-1 decompositions for 2x2x2.
    
    # Let's generate the 7 rank-1 tensors for 2x2x2.
    # T_ijk = 1 for all i,j,k in {0,1}.
    # Decomposition:
    # 1. U=[[1,0],[0,0]], V=[[1,0],[0,0]], W=[[1,0],[0,0]]
    # 2. U=[[0,0],[0,1]], V=[[0,0],[0,1]], W=[[0,0],[0,1]]
    # ... This is schoolbook (4 terms).
    # Strassen uses 7 terms with +/- 1.
    
    # Instead of hardcoding Strassen 2x2x2, we can use the known 7-term decomposition.
    # Or just use schoolbook for 2x2x2 (rank 4) -> 4x4x4 rank 16*4=64? No.
    # If we use schoolbook for 2x2x2, the 4x4x4 rank is 16 (4 blocks * 4 terms).
    # 16 is better than 49? No, 16 < 49.
    # Wait, schoolbook for 4x4x4 is 64.
    # If we use schoolbook for the 2x2 blocks, we have 4 blocks of A, 4 blocks of B.
    # Each block product is 2x2x2.
    # If we use schoolbook for 2x2x2 (rank 4), total rank is 16 * 4 = 64.
    # This is the same as full schoolbook.
    # Strassen reduces 2x2x2 to 7.
    # But 7 > 4. So Strassen is worse for 2x2x2?
    # No, Strassen is 7, schoolbook is 4.
    # Strassen is better for larger matrices because $7^k < 8^k$.
    # For k=1, 7 > 4.
    # For k=2, 49 < 64.
    # So for 4x4x4, Strassen (49) is better than schoolbook (64).
    
    # So we should use Strassen for 4x4x4.
    # We need the 7 rank-1 terms for 2x2x2.
    
    # 2x2x2 Strassen decomposition
    # U, V, W are 2x2 matrices.
    # Term 1: U=[[1,0],[0,0]], V=[[1,0],[0,0]], W=[[1,0],[0,0]] -> A11 B11
    # This is not Strassen.
    
    # Let's use the explicit Strassen 7 terms for 2x2x2.
    # See Wikipedia or similar.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a 2x2 matrix product.
    # We need to decompose each 2x2x2 product into 4 rank-1 terms (schoolbook) or 7 (Strassen)?
    # No, the "Strassen algorithm" for 2x2x2 *is* the 7-term decomposition.
    # The 7 terms *are* the rank-1 tensors.
    # So for 2x2x2, rank is 7.
    
    # For 4x4x4, we apply this to the 4x4 blocks?
    # No, 4x4 is composed of 2x2 blocks.
    # The multiplication of 4x4 matrices is NOT 7 multiplications of 2x2 matrices.
    # It is 7 multiplications of 2x2 matrices IF we use Strassen on the block level.
    # But Strassen on block level requires the blocks to be 2x2.
    # So yes, we do 7 products of 2x2 matrices.
    # Each 2x2 product has rank 7.
    # So total rank 49.
    
    # We need the 7 rank-1 tensors for 2x2x2.
    # Let's define them.
    # T1: A11 B11 -> U=[[1,0],[0,0]], V=[[1,0],[0,0]], W=[[1,0],[0,0]]
    # This is just schoolbook.
    # Strassen 2x2x2 terms:
    # 1. U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]] ? No.
    
    # I will use the schoolbook for 2x2x2 (rank 4) for simplicity, 
    # which gives rank 64 for 4x4x4.
    # And schoolbook for 4x4x2 (rank 32).
    # Total 96. No improvement.
    
    # I MUST use Strassen 2x2x2 terms.
    # The 7 terms for 2x2x2 are:
    # 1. U=[[1,0],[0,0]], V=[[1,0],[0,0]], W=[[1,0],[0,0]] (A11 B11)
    # 2. U=[[0,0],[0,1]], V=[[0,0],[0,1]], W=[[0,0],[0,1]] (A22 B22)
    # 3. U=[[0,1],[0,0]], V=[[0,0],[0,1]], W=[[0,1],[0,0]] (A12 B21)
    # 4. U=[[1,0],[0,0]], V=[[0,0],[0,1]], W=[[1,0],[0,0]] (A11 B21)
    # ... This is schoolbook.
    
    # Strassen 2x2x2 decomposition:
    # 1. U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]] (A11+A22)(B11+B22)
    # 2. U=[[0,1],[0,1]], V=[[1,0],[0,0]], W=[[0,1],[0,0]] (A21+A22)B11
    # 3. U=[[1,0],[0,0]], V=[[0,1],[0,-1]], W=[[1,0],[0,0]] A11(B12-B22)
    # 4. U=[[0,0],[0,1]], V=[[0,-1],[0,1]], W=[[0,0],[0,1]] A22(B21-B11)
    # 5. U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[0,1],[0,0]] (A11+A12)B22
    # 6. U=[[0,-1],[0,1]], V=[[1,1],[0,0]], W=[[0,1],[0,0]] (A21-A11)(B11+B12)
    # 7. U=[[0,1],[0,-1]], V=[[0,1],[0,1]], W=[[0,1],[0,0]] (A12-A22)(B21+B22)
    
    # Note: W indices are k,i.
    # For 2x2, i,k in {0,1}.
    # W11 (k=0,i=0), W12 (k=0,i=1), W21 (k=1,i=0), W22 (k=1,i=1).
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will construct the 49 terms for 4x4x4.
    
    # Then add 32 terms for 4x4x2 schoolbook.
    
    # Total rank 81.
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u, v, w, shift_k=0):
        # u: 4x4 flattened? No, U is 4x4 for A.
        # U shape (16, rank).
        # u is 16x1 vector.
        U.append(u)
        V.append(v)
        W.append(w)
        
    # 2x2x2 Strassen terms (u, v, w)
    # u, v, w are 4x1 vectors (flattened 2x2).
    # i,j in {0,1}. k in {0,1}.
    # U index i*2+j.
    # V index j*2+k.
    # W index k*2+i.
    
    S2 = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,1],[0,1]], [[1,0],[0,0]], [[0,1],[0,0]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[1,0],[0,0]]),
        ([[0,0],[0,1]], [[0,-1],[0,1]], [[0,0],[0,1]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[0,1],[0,0]]),
        ([[0,-1],[0,1]], [[1,1],[0,0]], [[0,1],[0,0]]),
        ([[0,1],[0,-1]], [[0,1],[0,1]], [[0,1],[0,0]])
    ]
    
    # Map 2x2 indices to 4x4 indices
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00, B01, B10, B11
    
    # Strassen 4x4x4:
    # We apply the 7 terms to the block matrices.
    # Each term involves specific blocks.
    
    # M1: (A00+A11)(B00+B11)
    # A' = A00 + A11. B' = B00 + B11.
    # We need to decompose A' B'.
    # A' is 2x2. B' is 2x2.
    # We use the 7 rank-1 terms for 2x2x2.
    # But A' is a sum of blocks.
    # So U for this term is the sum of U for A00 and A11.
    
    # This is getting complex to implement correctly in short time.
    # I will use the schoolbook for 4x4x4 (64) and 4x4x2 (32). Total 96.
    # This is the seed.
    
    # To improve, I will use a simple heuristic: 
    # Try to find a rank 80 decomposition?
    # No, I'll stick to the Strassen 4x4x4 (49) + Schoolbook 4x4x2 (32) = 81.
    # I will implement it carefully.
    
    # 4x4x4 Strassen construction
    # Blocks:
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # For each of the 7 Strassen steps:
    # 1. Determine A' and B' (2x2 matrices of blocks)
    # 2. For each of the 7 2x2x2 terms:
    #    - Determine u, v, w (2x2 matrices)
    #    - Map to 4x4 vectors
    
    # A' is a 2x2 matrix where each entry is a 2x2 block.
    # u is a 2x2 matrix.
    # The 4x4 U vector is formed by placing u in the correct block positions.
    
    # Example: M1 = (A00+A11)(B00+B11)
    # A' = [[1,0],[0,1]] (coeffs for A00, A11)
    # B' = [[1,0],[0,1]]
    # Term 1 of 2x2x2: u=[[1,0],[0,0]], v=[[1,0],[0,0]], w=[[1,0],[0,0]]
    # This corresponds to A'11 B'11 = A00 B00.
    # U vector: A00 block has u, others 0.
    
    # I will implement this.
    
    pass
