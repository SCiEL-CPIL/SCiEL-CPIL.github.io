import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    
    # We will construct a decomposition of rank 60.
    # Split B into B1 (5x4) and B2 (5x2)
    # B1 cols 0..3, B2 cols 4..5
    
    # Split A into A1 (4x4) and A5 (4x1)
    # A1 cols 0..3, A5 col 4
    
    # Part 1: A1 * B1[0:4, 0:4] (4x4x4) -> Strassen Rank 49
    # Part 2: A5 * B1[4, 0:4] (4x1x4) -> Rank 1
    # Part 3: A1 * B2[0:4, 0:2] (4x4x2) -> Rank 8
    # Part 4: A5 * B2[4, 0:2] (4x1x2) -> Rank 2
    
    total_rank = 49 + 1 + 8 + 2
    R = total_rank
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    r_offset = 0
    
    # Helper to build Strassen 2x2x2
    def build_strassen_2x2(U, V, W, r_offset, i_base, k_base, j_base, n_sub, m_sub, p_sub):
        # This function is complex to generalize for arbitrary offsets in the global arrays.
        # Instead, we will just hardcode the 49 terms for 4x4x4 or build it recursively.
        # Given the complexity, let's use a simpler approach:
        # We know the rank is 49. We can just use schoolbook for 4x4x4 if Strassen is too hard to code quickly?
        # No, schoolbook is 64. 64+1+8+2 = 75. 120/75 = 1.6. Still good.
        # But Strassen is 49. 120/60 = 2.0. Better.
        
        # Let's implement Strassen 4x4x4 by combining 2x2x2 blocks.
        # It's safer to just use schoolbook for the 4x4x4 part to avoid bugs, 
        # unless I can verify the Strassen implementation easily.
        # Actually, 4x4x4 Strassen is standard.
        # Let's try to implement a generic Strassen builder.
        
        pass

    # To be safe and fast, let's use Schoolbook for 4x4x4 (Rank 64) 
    # and the low rank tricks for the rest.
    # Total Rank: 64 + 1 + 8 + 2 = 75.
    # Score: 120/75 = 1.6.
    
    # Let's re-evaluate if 4x4x4 Strassen is worth the risk.
    # The prompt inspirations achieved 1.188 (Rank 101).
    # Rank 75 is much better.
    
    # Let's stick to the safe Rank 75 solution first, then optimize if possible.
    # Actually, I can just write the Strassen 4x4x4 decomposition.
    # It consists of 49 terms.
    # I will generate them programmatically.
    
    # Define Strassen 2x2x2 terms (U, V, W coefficients)
    # U indices: i, k. V indices: k, j. W indices: k, i.
    # For 2x2x2:
    # M1: U(0,0)=1, U(1,1)=1; V(0,0)=1, V(1,1)=1; W(0,0)=1, W(1,1)=1? No.
    # The decomposition is sum of rank 1 tensors.
    # For M1 = (A11+A22)(B11+B22):
    # This corresponds to terms where i=0,k=0 and i=1,k=1 etc.
    # It's easier to just use the known fact that Strassen 4x4x4 has rank 49.
    # I will implement a function to generate the 49 terms.
    
    def get_strassen_4x4x4():
        # Returns U, V, W for 4x4x4 Strassen
        # U shape (16, 49), V shape (16, 49), W shape (16, 49)
        u = np.zeros((16, 49))
        v = np.zeros((16, 49))
        w = np.zeros((16, 49))
        
        # This is tedious to hardcode.
        # Let's use Schoolbook for 4x4x4 to ensure correctness.
        # Rank 64.
        return None

    # Okay, I will use Schoolbook for 4x4x4.
    # Part 1: 4x4x4 Schoolbook
    # i in 0..3, k in 0..3, j in 0..3
    for i in range(4):
        for k in range(4):
            for j in range(4):
                idx = i * 4 + k # Local U index? No, global U index is i*m + k
                # Global U index: i * 5 + k
                # Global V index: k * 6 + j
                # Global W index: k * 4 + i
                U[i * 5 + k, r_offset] = 1.0
                V[k * 6 + j, r_offset] = 1.0
                W[k * 4 + i, r_offset] = 1.0
                r_offset += 1
                
    # r_offset is now 64.
    
    # Part 2: A5 * B1[4, 0:4] (4x1x4) Rank 1
    # i in 0..3, k=4, j in 0..3
    # We need sum_r U[i,4,r] V[4,j,r] W[4,i,r] = 1
    # Set r_offset to 64.
    for i in range(4):
        for j in range(4):
            # This is 1 term for all i,j?
            # If we use 1 term, U[i,4,64]=1, V[4,j,64]=1, W[4,i,64]=1.
            # Sum is 1*1*1 = 1. Correct.
            pass # We just need to set the values for r=64.
            
    U[:, r_offset] = 0 # Clear
    for i in range(4):
        U[i * 5 + 4, r_offset] = 1.0
        W[4 * 4 + i, r_offset] = 1.0
        
    for j in range(4):
        V[4 * 6 + j, r_offset] = 1.0
        
    r_offset += 1 # Now 65
    
    # Part 3: A1 * B2[0:4, 0:2] (4x4x2) Rank 8
    # i in 0..3, k in 0..3, j in 4..5
    # For each j_local in 0..1 (global j = 4+j_local):
    #   For each k in 0..3:
    #     Term r.
    #     U[i,k,r]=1, V[k,j,r]=1, W[k,i,r]=1.
    
    for j_local in range(2):
        j_global = 4 + j_local
        for k in range(4):
            for i in range(4):
                U[i * 5 + k, r_offset] = 1.0
                V[k * 6 + j_global, r_offset] = 1.0
                W[k * 4 + i, r_offset] = 1.0
            r_offset += 1 # 8 terms total
            
    # r_offset is now 65 + 8 = 73.
    
    # Part 4: A5 * B2[4, 0:2] (4x1x2) Rank 2
    # i in 0..3, k=4, j in 4..5
    # For each j_local in 0..1:
    #   Term r.
    #   U[i,4,r]=1, V[4,j,r]=1, W[4,i,r]=1.
    
    for j_local in range(2):
        j_global = 4 + j_local
        for i in range(4):
            U[i * 5 + 4, r_offset] = 1.0
            V[4 * 6 + j_global, r_offset] = 1.0
            W[4 * 4 + i, r_offset] = 1.0
        r_offset += 1 # 2 terms total
        
    # r_offset is now 73 + 2 = 75.
    
    # Trim to actual rank used
    U = U[:, :r_offset]
    V = V[:, :r_offset]
    W = W[:, :r_offset]
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
        
    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
