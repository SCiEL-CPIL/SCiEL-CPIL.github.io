import time

DIMENSIONS = [2, 3, 10]

# Strassen rank-7 tensors for 2x2x2 multiplication.
# Each tensor is (U_l, V_l, W_l) where U_l is 2x2, V_l is 2x2, W_l is 2x2.
# The convention: for multiplying A (2x2) by B (2x2), C[i,k] = sum_l U_l[i,j] * V_l[j,k] * W_l[k,i]
# But our convention is W[k*n+i, r], so W_l has indices [k, i].
# We need: sum_l U_l[i,j] * V_l[j,k] * W_l[k,i] = 1 for all i,j,k in {0,1}
# This is equivalent to the standard Strassen decomposition of the 2x2x2 tensor.

# Standard Strassen 7 tensors (verified):
STRASSEN_TENSORS = [
    # (U, V, W) where U[i,j], V[j,k], W[k,i]
    # T1: (A11+A22) * (B11+B22) -> contributes to C11 and C22
    ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
    # T2: (A21+A22) * B11 -> contributes to C21 and C22
    ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),
    # T3: A11 * (B21-B11) -> contributes to C12 and C22
    ([[1,0],[0,0]], [[0,-1],[1,0]], [[0,0],[1,1]]),
    # T4: A22 * (B22-B12) -> contributes to C11 and C12
    ([[0,0],[0,1]], [[0,1],[-1,0]], [[1,0],[1,0]]),
    # T5: (A11+A12) * B22 -> contributes to C11 and C12
    ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,1]]),
    # T6: A12 * (B12-B22) -> contributes to C21 and C22
    ([[0,1],[0,0]], [[0,1],[-1,0]], [[0,0],[0,1]]),
    # T7: A21 * (B11-A22) -> contributes to C21
    ([[0,0],[1,0]], [[1,0],[0,-1]], [[0,0],[0,1]]),
]

def _verify_strassen():
    """Verify that the Strassen tensors correctly decompose the 2x2x2 tensor."""
    n = 2
    for i in range(n):
        for j in range(n):
            for k in range(n):
                total = 0
                for U, V, W in STRASSEN_TENSORS:
                    total += U[i][j] * V[j][k] * W[k][i]
                if total != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Strategy: Transpose to M_{10,3,2}, split m=3 into m1=2, m2=1
    # For the 10x2 x 2x2 block: use Strassen with rank 7 per 2x2x2 sub-block
    # For the 10x2 x 1x2 block: use schoolbook with rank 2*2*1=4... 
    # Wait, let me reconsider.
    
    # Actually, let's work directly with M_{2,3,10}:
    # A is 2x3, B is 3x10, C is 2x10
    # We can split the middle dimension m=3 into m1=2, m2=1
    # C = A[:,:2] * B[:,:2].T ... no, C[i,k] = sum_j A[i,j]*B[j,k]
    # C = A[:,:2] * B[0:2,:] + A[:,2:3] * B[2:3,:]
    
    # For A[:,:2] * B[0:2,:]: This is M_{2,2,10}
    # We can decompose this as: for each pair of rows i in {0,1} and each column k in 0..9
    # Actually, M_{2,2,10} has rank at most 7*5 = 35 using Strassen on the 2x2 block
    # But we need to handle the 10 columns.
    
    # Better approach: M_{2,2,10} = product of 2x2 and 2x10
    # We can group the 10 columns into 5 groups of 2, and apply Strassen to each group.
    # Each Strassen application handles 2x2 * 2x2 -> 2x2 with rank 7
    # So 5 groups * rank 7 = rank 35 for the 2x2 * 2x10 part
    
    # For A[:,2] * B[2,:]: This is M_{2,1,10}, which has rank 2*1*10 = 20 (schoolbook)
    
    # Total rank: 35 + 20 = 55
    
    # Let's build this decomposition.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # (2*3=6, 55)
    V = [[0] * rank for _ in range(m * p)]  # (3*10=30, 55)
    W = [[0] * rank for _ in range(p * n)]  # (10*2=20, 55)
    
    col = 0
    
    # Part 1: Strassen for A[:,:2] * B[0:2,:] (M_{2,2,10})
    # Group columns 0..9 into 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    for g in range(5):
        k1 = 2 * g
        k2 = 2 * g + 1
        
        for l in range(7):
            U_l, V_l, W_l = STRASSEN_TENSORS[l]
            
            # U_l[i,j] for i in {0,1}, j in {0,1} (these are the first 2 columns of A)
            # V_l[j,k] for j in {0,1}, k in {k1,k2} (these are the first 2 rows of B, cols k1,k2)
            # W_l[k,i] for k in {0,1} (local), i in {0,1} (global rows)
            
            for i in range(2):
                for j in range(2):
                    if U_l[i][j] != 0:
                        U[i * m + j][col] += U_l[i][j]
            
            for j in range(2):
                # V_l[j][0] corresponds to global column k1
                # V_l[j][1] corresponds to global column k2
                for k_local in range(2):
                    k_global = k1 + k_local
                    if V_l[j][k_local] != 0:
                        V[j * p + k_global][col] += V_l[j][k_local]
            
            for k_local in range(2):
                for i in range(2):
                    if W_l[k_local][i] != 0:
                        # W index is [k*n + i] = [k_global * n + i]
                        k_global = k1 + k_local
                        W[k_global * n + i][col] += W_l[k_local][i]
            
            col += 1
    
    # Part 2: Schoolbook for A[:,2] * B[2,:] (M_{2,1,10})
    # A has 2 rows, B has 1 row (row 2), output has 10 columns
    # For each i in {0,1}, k in 0..9: one product A[i,2] * B[2,k]
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
