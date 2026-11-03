"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    
    def schoolbook():
        rank = rank_schoolbook
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
    best_rank = rank_schoolbook
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = len(U[0])
        # Verify shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        # Check the tensor equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # for all i, j, k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    # Strategy: Try to find a decomposition with rank < 80 using a structured approach.
    # For 4x4 * 4x5, we can use block decomposition.
    # Split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    # Split B (4x5) into 2x5 blocks: B1, B2
    # C = A*B = [A11*B1 + A12*B2; A21*B1 + A22*B2]
    # Each block multiplication is 2x2 * 2x5.
    # We can use Strassen for the 2x2 part? No, Strassen is for square matrices.
    # But we can still use a bilinear decomposition for 2x2 * 2x5.
    # The rank of 2x2 * 2x5 is 10 (schoolbook).
    # Total rank would be 4 * 10 = 40? No, that's not how it works.
    # Actually, the block decomposition doesn't reduce the rank unless we use a clever combination.
    
    # Let's try a different approach: use a known decomposition for smaller matrices
    # and combine them.
    
    # For now, let's try a randomized search for rank 64, 72, 56, 48 with entries in {-1, 0, 1}.
    # This is very hard, but let's try a few iterations.
    
    np.random.seed(42)
    
    # Try to find a decomposition with rank < 80
    for target_rank in [72, 64, 56, 48]:
        if time.time() > deadline - 10:
            break
        
        # Try random initialization with entries in {-1, 0, 1}
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            # Convert to list of lists for the return format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if check_decomposition(U_list, V_list, W_list):
                result = {"U": U_list, "V": V_list, "W": W_list}
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                break
    
    return best_result
# EVOLVE-BLOCK-END
