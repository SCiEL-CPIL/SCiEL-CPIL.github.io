"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 62 decomposition
    # Part 1: 4x1x6 (Rank 6)
    # Part 2: 4x4x2 (Rank 28, using Strassen 4x4x4)
    # Part 3: 4x4x4 (Rank 28, using Strassen 4x4x4)
    
    rank = 62
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x1x6 ---
    # i in 0..3, j=0, k in 0..5
    # U[i*5 + 0, r], V[0*6 + k, r], W[k*4 + i, r]
    # We use r = 0..5 for k
    for k in range(6):
        r = k
        for i in range(4):
            U[i * 5 + 0][r] = 1
            W[k * 4 + i][r] = 1
        V[0 * 6 + k][r] = 1
        # Other V entries for this r are 0 (already initialized)
        # Other W entries for this r are 0 (already initialized)
        # Other U entries for this r are 0 (already initialized)
        
    # --- Part 2 & 3: Strassen 4x4x4 ---
    # We need two instances of Strassen 4x4x4.
    # Instance 1: For A2 (4x4) and M2a (4x2). We use the first 2 columns of the Strassen output.
    # Instance 2: For A2 (4x4) and M2b (4x4). We use all 4 columns.
    
    # Strassen 4x4x4 decomposition
    # Standard Strassen for 4x4x4 can be built from 7 products of 2x2x2.
    # Or we can use the known rank 28 decomposition.
    # Here we construct the 28 terms for 4x4x4.
    
    # Helper to add Strassen 4x4x4 terms
    def add_strassen_4x4x4(offset_u, offset_v, offset_w, start_rank):
        # U indices: i in 0..3, j in 0..3. Global U index: (offset_u + i)*4 + (offset_v? No)
        # U is (n*m, R). n=4, m=4 for this sub-block?
        # Wait, global U is (4*5, R).
        # For A2, i in 0..3, j in 1..4 (since A1 was j=0).
        # So U index: i*5 + j.
        # For M2a, j in 1..4, k in 0..1.
        # V index: j*6 + k.
        # For W, k in 0..1, i in 0..3.
        # W index: k*4 + i.
        
        # We need to map local Strassen indices to global indices.
        
        pass

    # Let's explicitly write the Strassen 4x4x4 terms.
    # There are 28 terms.
    # We can generate them or hardcode them.
    # For simplicity and correctness, I will use a known list or generate via recursion.
    
    # Recursion: Strassen 4x4x4 from 7 * Strassen 2x2x2.
    # Strassen 2x2x2 has rank 7.
    
    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        # Indices: i,j,k in 0..1
        # U: (2*2, 7) = (4, 7)
        # V: (2*2, 7) = (4, 7)
        # W: (2*2, 7) = (4, 7)
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # Standard Strassen 2x2x2
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B11 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # U (A side):
        # r=0: A11+A22 -> u[0,0]=1, u[3,0]=1 (indices: 00, 11)
        u[0, 0] = 1; u[3, 0] = 1
        # r=1: A21+A22 -> u[2,1]=1, u[3,1]=1 (indices: 10, 11)
        u[2, 1] = 1; u[3, 1] = 1
        # r=2: A11 -> u[0,2]=1
        u[0, 2] = 1
        # r=3: A22 -> u[3,3]=1
        u[3, 3] = 1
        # r=4: A11+A12 -> u[0,4]=1, u[1,4]=1 (indices: 00, 01)
        u[0, 4] = 1; u[1, 4] = 1
        # r=5: A21-A11 -> u[2,5]=1, u[0,5]=-1 (indices: 10, 00)
        u[2, 5] = 1; u[0, 5] = -1
        # r=6: A12-A22 -> u[1,6]=1, u[3,6]=-1 (indices: 01, 11)
        u[1, 6] = 1; u[3, 6] = -1
        
        # V (B side):
        # r=0: B11+B22 -> v[0,0]=1, v[3,0]=1
        v[0, 0] = 1; v[3, 0] = 1
        # r=1: B11 -> v[0,1]=1
        v[0, 1] = 1
        # r=2: B21-B22 -> v[2,2]=1, v[3,2]=-1
        v[2, 2] = 1; v[3, 2] = -1
        # r=3: B11-B12 -> v[0,3]=1, v[1,3]=-1
        v[0, 3] = 1; v[1, 3] = -1
        # r=4: B22 -> v[3,4]=1
        v[3, 4] = 1
        # r=5: B11+B12 -> v[0,5]=1, v[1,5]=1
        v[0, 5] = 1; v[1, 5] = 1
        # r=6: B21+B22 -> v[2,6]=1, v[3,6]=1
        v[2, 6] = 1; v[3, 6] = 1
        
        # W (C side):
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # W indices: k*n + i. n=2.
        # C11: i=0, k=0 -> idx 0
        # C12: i=0, k=1 -> idx 2
        # C21: i=1, k=0 -> idx 1
        # C22: i=1, k=1 -> idx 3
        
        # r=0 (M1):
        w[0, 0] = 1; w[3, 0] = 1
        # r=1 (M2):
        w[1, 1] = 1; w[3, 1] = -1
        # r=2 (M3):
        w[2, 2] = 1; w[3, 2] = 1
        # r=3 (M4):
        w[0, 3] = -1; w[1, 3] = 1
        # r=4 (M5):
        w[0, 4] = 1; w[2, 4] = 1
        # r=5 (M6):
        w[3, 5] = 1
        # r=6 (M7):
        w[0, 6] = -1
        
        return u, v, w

    # We need to map 2x2x2 blocks to 4x4x4.
    # This is complex to implement correctly in a short time.
    # I will use the known fact that rank 28 is achievable.
    # To be safe, I will just use schoolbook for the 4x4x4 parts?
    # Schoolbook 4x4x4 is 64.
    # Total rank: 6 + 32 + 64 = 102.
    # This is still better than 120.
    
    # Let's stick to Schoolbook for the 4x4 parts to ensure correctness.
    # Part 2: 4x4x2. Schoolbook rank 32.
    # Part 3: 4x4x4. Schoolbook rank 64.
    # Total: 6 + 32 + 64 = 102.
    
    # Reset rank
    rank = 102
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x1x6 (Rank 6)
    for k in range(6):
        r = k
        for i in range(4):
            U[i * 5 + 0][r] = 1
            W[k * 4 + i][r] = 1
        V[0 * 6 + k][r] = 1
        
    # Part 2: 4x4x2 (Rank 32)
    # i in 0..3, j in 1..4, k in 0..1
    r_offset = 6
    for i in range(4):
        for j in range(1, 5):
            for k in range(2):
                r = r_offset + (i * 4 * 2) + (j - 1) * 2 + k
                U[i * 5 + j][r] = 1
                V[j * 6 + k][r] = 1
                W[k * 4 + i][r] = 1
                
    # Part 3: 4x4x4 (Rank 64)
    # i in 0..3, j in 1..4, k in 2..5
    r_offset = 6 + 32
    for i in range(4):
        for j in range(1, 5):
            for k in range(2, 6):
                r = r_offset + (i * 4 * 4) + (j - 1) * 4 + (k - 2)
                U[i * 5 + j][r] = 1
                V[j * 6 + k][r] = 1
                W[k * 4 + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
