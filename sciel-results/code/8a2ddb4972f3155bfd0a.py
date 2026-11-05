"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen rank-7 factors for 2x2 x 2x2 -> 2x2
    # U: (i,j) -> 7, V: (j,k) -> 7, W: (k,i) -> 7
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, -1, 0, 0, 1],
        [1, 0, -1, 0, 1, 1, 0]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, -1, 1, 0, -1],   # k=0, i=0
        [0, 0, 1, 0, 1, 0, 0],      # k=0, i=1
        [0, 1, 0, 1, 0, 0, 0],      # k=1, i=0
        [1, 0, -1, 0, 0, 1, -1]     # k=1, i=1
    ], dtype=float)
    
    # For 4x4 x 4x4: use Kronecker product of Strassen factors
    # U4 = U2 ⊗ U2, V4 = V2 ⊗ V2, W4 = W2 ⊗ W2
    # Each has shape (16, 49)
    U4 = np.kron(U2, U2)  # 16 x 49
    V4 = np.kron(V2, V2)  # 16 x 49
    W4 = np.kron(W2, W2)  # 16 x 49
    
    # For 4x4 x 4x8: split B into two 4x4 matrices B1, B2
    # C = A * [B1 | B2] = [A*B1 | A*B2]
    # Use rank-49 decomposition for each 4x4 x 4x4 multiplication
    # Total rank = 49 + 49 = 98
    
    rank = 98
    U = np.zeros((16, rank))  # n*m = 16
    V = np.zeros((16, rank))  # m*p = 4*8 = 32... wait
    # m=4, p=8, so m*p = 32
    # But we're splitting B into two 4x4, so each V part is 4*4=16 rows
    # We need to map the 32 rows of V (for 4x8) to two blocks of 16
    
    V = np.zeros((32, rank))  # m*p = 32
    W = np.zeros((32, rank))  # p*n = 8*4 = 32
    
    # First half: A * B1 where B1 is columns 0-3 of B (4x4)
    # U part: same U4 for both (since A is the same)
    U[:, 0:49] = U4
    # V part: B1 occupies rows 0-15 of V (j*p+k for j=0,1,2,3 and k=0,1,2,3)
    # V row index: j*p + k = j*8 + k. For B1 (first 4 columns), k=0,1,2,3
    # So rows: 0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27
    # But we want to use V4 which assumes contiguous 4x4. We need to permute.
    # V4 has rows indexed by (j,k) with j,k in {0,1,2,3} in row-major: j*4+k
    # But actual V rows are j*8+k. So we need to map.
    
    # Let's build V properly. V4[i4, r] where i4 = j*4+k (for 4x4 block)
    # Actual V row index for element (j,k) in B is j*p + k = j*8 + k
    # For B1 (k=0,1,2,3): row = j*8 + k
    # For B2 (k=4,5,6,7): row = j*8 + k = j*8 + (k-4) + 4
    
    # V for B1: for each j in 0..3, k in 0..3: V[j*8+k, r] = V4[j*4+k, r]
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
    
    # V for B2: for each j in 0..3, k in 4..7: V[j*8+k, r] = V4[j*4+(k-4), 49+...]
    # Wait, we're using separate rank-49 for each, so B2 uses columns 49:98
    for j in range(4):
        for k in range(4):
            V[j*8 + (k+4), 49:98] = V4[j*4 + k, :]
    
    # W for C1 = A*B1 (columns 0-3 of C, which is 4x4):
    # W row index: k*n + i = k*4 + i for k=0,1,2,3 (first 4 columns of C)
    # W4 has rows indexed by (k,i) with k,i in {0,1,2,3} in row-major: k*4+i
    # So W[k*4+i, r] = W4[k*4+i, r] for k=0..3, i=0..3
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, :]
    
    # W for C2 = A*B2 (columns 4-7 of C, which is 4x4):
    # k=4,5,6,7: W[k*4+i, r] = W4[(k-4)*4+i, 49+...]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4 + i, 49:98] = W4[k*4 + i, :]
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
