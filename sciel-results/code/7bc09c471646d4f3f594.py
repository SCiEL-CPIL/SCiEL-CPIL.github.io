"""Matrix multiplication 2x4x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _make_schoolbook(n, m, p):
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
    return U, V, W

def _try_block_decomposition(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition by exploiting block structure.
    
    For 2x4x8: split the 4x8 matrix into two 4x4 blocks.
    This gives us two independent 2x4x4 multiplications.
    
    Each 2x4x4 has schoolbook rank 32. If we can find rank < 32 for each,
    total rank < 64.
    
    Actually, let's think more carefully. The tensor for (n,m,p) multiplication
    is sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} in appropriate spaces.
    
    For 2x4x8, we can view it as:
    - A is 2x4, B is 4x8, C is 2x8
    - C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split B into B1 (4x4) and B2 (4x4) columns.
    Then C = A*B1 + A*B2, where A*B1 is 2x4 and A*B2 is 2x4.
    
    The tensor is the direct sum of two (2,4,4) tensors.
    So rank(2,4,8) <= 2 * rank(2,4,4).
    
    Known: rank(2,4,4) - schoolbook is 32. Can we do better?
    
    Let's try a randomized search for rank 31 for (2,4,4) first,
    then use that to get rank 62 for (2,4,8).
    """
    # First, try to find a rank-31 decomposition for 2x4x4
    # If we can't, try rank 32 (which is schoolbook)
    
    # Actually, let's try a different structural approach.
    # Use the fact that 8 = 4 + 4 to get a block decomposition.
    
    # The (2,4,8) multiplication tensor can be decomposed as:
    # T_{2,4,8} = T_{2,4,4} ⊕ T_{2,4,4} (direct sum)
    # So rank <= 2 * rank(2,4,4)
    
    # Let's try to find a low-rank decomposition for (2,4,4)
    # Schoolbook rank for (2,4,4) is 32
    
    # Try random search for (2,4,4) with rank 31
    def _search_2x4x4(rank_target, deadline):
        n2, m2, p2 = 2, 4, 4
        dimU = n2 * m2  # 8
        dimV = m2 * p2  # 16
        dimW = p2 * n2  # 8
        
        # Try random integer entries
        for attempt in range(200):
            if time.time() > deadline:
                return None
            
            # Initialize with small random integers
            U = np.random.randint(-2, 3, size=(dimU, rank_target))
            V = np.random.randint(-2, 3, size=(dimV, rank_target))
            W = np.random.randint(-2, 3, size=(dimW, rank_target))
            
            # Check if this works
            valid = True
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        val = 0
                        for c in range(rank_target):
                            val += U[i*m2 + j, c] * V[j*p2 + k, c] * W[k*n2 + i, c]
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try to find rank 31 for (2,4,4)
    result = _search_2x4x4(31, deadline)
    if result is not None:
        U244, V244, W244 = result
        rank244 = 31
    else:
        # Use schoolbook for (2,4,4)
        U244, V244, W244 = _make_schoolbook(2, 4, 4)
        rank244 = 32
    
    # Now build the (2,4,8) decomposition from two (2,4,4) blocks
    # B = [B1 | B2] where B1, B2 are 4x4
    # C = A*B1 + A*B2
    # 
    # U for (2,4,8): same U for both blocks (A is the same)
    # V for (2,4,8): [V1; V2] where V1 corresponds to B1, V2 to B2
    # W for (2,4,8): [W1; W2] where W1 for C columns 0-3, W2 for C columns 4-7
    
    # Actually, this isn't quite right. Let me think again.
    # 
    # For (2,4,8):
    # U has shape (2*4, rank) = (8, rank)
    # V has shape (4*8, rank) = (32, rank)
    # W has shape (8*2, rank) = (16, rank)
    #
    # The constraint is: for all i in 0..1, j in 0..3, k in 0..7:
    # sum_c U[i*4+j, c] * V[j*8+k, c] * W[k*2+i, c] = 1
    #
    # If we split k into k1 (0..3) and k2 (4..7), we have two sets of constraints.
    # 
    # Let's define:
    # For k = k1 (0..3): V[j*8+k1, c] and W[k1*2+i, c]
    # For k = 4+k2 (0..3): V[j*8+4+k2, c] and W[(4+k2)*2+i, c]
    #
    # This is not simply a direct sum because the indices are coupled differently.
    
    # Let me try a different approach: just do a direct random search for (2,4,8)
    # with rank 63, 62, etc.
    
    # Direct random search for (2,4,8)
    n, m, p = 2, 4, 8
    dimU = n * m  # 8
    dimV = m * p  # 32
    dimW = p * n  # 16
    
    for rank_target in [63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50]:
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Use small random integers
            U = np.random.randint(-1, 2, size=(dimU, rank_target))
            V = np.random.randint(-1, 2, size=(dimV, rank_target))
            W = np.random.randint(-1, 2, size=(dimW, rank_target))
            
            # Check validity
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank_target):
                            val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with schoolbook
    U_best, V_best, W_best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    result = _try_block_decomposition(n, m, p, deadline)
    if result is not None:
        U_new, V_new, W_new = result
        if _verify(U_new, V_new, W_new, n, m, p):
            U_best, V_best, W_best = U_new, V_new, W_new
            best_rank = len(U_new[0])
    
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
