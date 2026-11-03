import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Helper to verify a decomposition
    def verify(U, V, W, n, m, p):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        
        # Compute tensor contraction
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We check if T matches the identity tensor for multiplication
        # T[i, j, k] should be 1 for all valid i, j, k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if val != 1:
                        return False
        return True

    # Helper to find a decomposition using random search
    def find_decomposition(n, m, p, target_rank, max_iter=20000):
        # Initialize with random small integers
        # U: n*m x rank
        # V: m*p x rank
        # W: p*n x rank
        U = np.random.randint(-2, 3, size=(n * m, target_rank))
        V = np.random.randint(-2, 3, size=(m * p, target_rank))
        W = np.random.randint(-2, 3, size=(p * n, target_rank))
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return None
            
            if verify(U, V, W, n, m, p):
                return U, V, W
            
            # Randomly perturb a few elements
            for _ in range(3):
                idx = np.random.randint(0, target_rank)
                # Perturb U
                if np.random.rand() < 0.33:
                    row = np.random.randint(0, n * m)
                    U[row, idx] += np.random.choice([-1, 1])
                # Perturb V
                elif np.random.rand() < 0.66:
                    row = np.random.randint(0, m * p)
                    V[row, idx] += np.random.choice([-1, 1])
                # Perturb W
                else:
                    row = np.random.randint(0, p * n)
                    W[row, idx] += np.random.choice([-1, 1])
                    
        return None

    # Strategy: Decompose 2x3x10 into two 2x3x5 blocks
    # 2x3x5 decomposed into 2x3x3 (rank 16) and 2x3x2 (rank 10)
    
    # Find 2x3x3 rank 16
    U33, V33, W33 = None, None, None
    if time.time() < deadline:
        U33, V33, W33 = find_decomposition(2, 3, 3, 16, max_iter=5000)
        
    # Find 2x3x2 rank 10
    U32, V32, W32 = None, None, None
    if time.time() < deadline:
        U32, V32, W32 = find_decomposition(2, 3, 2, 10, max_iter=5000)

    if U33 is not None and U32 is not None:
        # Construct 2x3x5
        # U5 = [U33 | U32]
        # V5 = [V33 | V32] but V33 is 3x3, V32 is 3x2. 
        # Wait, V for 2x3x3 is 3x3. V for 2x3x2 is 3x2.
        # We need to align the columns.
        # U33 is 6x16. U32 is 6x10.
        # V33 is 9x16. V32 is 6x10. (m*p = 3*2=6)
        # W33 is 6x16. W32 is 4x10. (p*n = 3*2=6? No, p*n = 3*2=6. Wait. 2x3x2: p=2, n=2. W is 2x2=4? No.
        # W shape is (p*n, rank). For 2x3x3: p=3, n=2 -> 6. For 2x3x2: p=2, n=2 -> 4.
        
        # We need to pad the smaller matrices to match dimensions?
        # No, we are concatenating the decomposition.
        # The tensor for 2x3x5 is T5[i,j,k].
        # If we use block 1 for k=0,1,2 and block 2 for k=3,4.
        # Then U is same for all k? No.
        # U depends on i,j. V depends on j,k. W depends on k,i.
        
        # For block 1 (2x3x3):
        # U1[i,j,c] for c in 0..15
        # V1[j,k,c] for k in 0..2, c in 0..15
        # W1[k,i,c] for k in 0..2, c in 0..15
        
        # For block 2 (2x3x2):
        # U2[i,j,c] for c in 0..9
        # V2[j,k,c] for k in 0..1, c in 0..9
        # W2[k,i,c] for k in 0..1, c in 0..9
        
        # Combined for 2x3x5:
        # U5[i,j, c] = U1[i,j,c] if c<16 else U2[i,j,c-16]
        # V5[j,k, c] = V1[j,k,c] if k<3 and c<16 else V2[j,k-3,c-16] if k>=3
        # W5[k,i, c] = W1[k,i,c] if k<3 and c<16 else W2[k-3,i,c-16] if k>=3
        
        rank5 = 16 + 10
        U5 = np.zeros((2 * 3, rank5))
        V5 = np.zeros((3 * 5, rank5))
        W5 = np.zeros((5 * 2, rank5))
        
        # Copy block 1
        U5[:, :16] = U33
        # V33 is 9x16 (3x3). V5 rows 0..8 correspond to k=0,1,2.
        V5[0:9, :16] = V33
        # W33 is 6x16 (3x2). W5 rows 0..5 correspond to k=0,1,2.
        W5[0:6, :16] = W33
        
        # Copy block 2
        U5[:, 16:] = U32
        # V32 is 6x10 (3x2). V5 rows 9..14 correspond to k=3,4.
        V5[9:15, 16:] = V32
        # W32 is 4x10 (2x2). W5 rows 6..9 correspond to k=3,4.
        W5[6:10, 16:] = W32
        
        # Verify 2x3x5
        if verify(U5, V5, W5, 2, 3, 5):
            # Duplicate for 2x3x10
            rank10 = rank5 * 2
            U10 = np.zeros((2 * 3, rank10))
            V10 = np.zeros((3 * 10, rank10))
            W10 = np.zeros((10 * 2, rank10))
            
            # Block 1: k=0..4
            U10[:, :rank5] = U5
            V10[0:15, :rank5] = V5
            W10[0:10, :rank5] = W5
            
            # Block 2: k=5..9
            U10[:, rank5:] = U5
            V10[15:30, rank5:] = V5
            W10[10:20, rank5:] = W5
            
            if verify(U10, V10, W10, 2, 3, 10):
                return {
                    "U": U10.tolist(),
                    "V": V10.tolist(),
                    "W": W10.tolist()
                }

    # Fallback to seed
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
