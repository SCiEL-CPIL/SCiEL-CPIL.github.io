"""Rewritten program: search for low-rank exact decomposition of 5x5x5 matmul tensor."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    # Target: find rank-minimal decomposition
    # Known: schoolbook = 125, best known upper bound for 5x5 is 48
    
    best_result = None
    best_rank = n * m * p  # start with schoolbook as fallback
    
    # First, create the schoolbook decomposition as fallback
    U_fb = np.zeros((n*m, best_rank))
    V_fb = np.zeros((m*p, best_rank))
    W_fb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition using structured approach
    # Use Strassen-like blocks: split 5x5 into smaller blocks
    
    # Attempt 1: Use 3x3 Strassen for the top-left 3x3 block, and schoolbook for remaining
    # This gives rank = 7 (for 3x3) + 3*2 + 2*3 + 2*2 = 7 + 6 + 6 + 4 = 23
    # But we need to handle the full 5x5 carefully
    
    # Better approach: Try to find valid decompositions using known patterns
    
    # Try ranks from 48 upward with structured initialization
    for target_rank in [48, 50, 55, 60, 70, 80, 90, 100, 110, 120]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple structured approaches
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            # Initialize with sparse random values
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Use a structured pattern: each column covers a small sub-block
            for c in range(target_rank):
                # Determine which sub-block this column handles
                block_idx = c % 9  # 3x3 grid of blocks
                
                # Define block boundaries
                bi = block_idx // 3
                bj = block_idx % 3
                
                # Block size
                block_size = 2 if bi < 2 else 1  # first 2 blocks are 2x2, last is 1x1
                block_size_j = 2 if bj < 2 else 1
                
                # Actually, let's use a simpler approach: random sparse
                n_u = np.random.randint(2, 5)
                n_v = np.random.randint(2, 5)
                n_w = np.random.randint(2, 5)
                
                u_idx = np.random.choice(n*m, size=n_u, replace=False)
                v_idx = np.random.choice(m*p, size=n_v, replace=False)
                w_idx = np.random.choice(p*n, size=n_w, replace=False)
                
                U[u_idx, c] = np.random.choice([-1, 1], size=n_u)
                V[v_idx, c] = np.random.choice([-1, 1], size=n_v)
                W[w_idx, c] = np.random.choice([-1, 1], size=n_w)
            
            # Check if this random init happens to work
            if verify(U, V, W):
                best_rank = target_rank
                best_result = (U, V, W)
                break
        
        if best_result is not None:
            break
    
    # If no better found, use schoolbook
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = U_fb, V_fb, W_fb
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded * 2 - round(rounded * 2)) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(0)
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
