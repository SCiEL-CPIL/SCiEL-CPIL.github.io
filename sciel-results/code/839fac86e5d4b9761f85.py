"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin
    
    # Build the matrix multiplication tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    tensor = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school), dtype=np.int64)
    V_school = np.zeros((m*p, rank_school), dtype=np.int64)
    W_school = np.zeros((p*n, rank_school), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the tensor."""
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # Check tensor equation: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    expected = tensor[a, b, c]
                    computed = 0
                    for r_idx in range(r):
                        computed += U[a, r_idx] * V[b, r_idx] * W[c, r_idx]
                    if computed != expected:
                        return False
        return True
    
    best_rank = rank_school
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    # Try to find lower rank decompositions
    # Strategy: use the fact that for small tensors, we can try structured decompositions
    # with small integer entries
    
    # First, try rank = rank_school - 1, rank_school - 2, etc.
    # For each rank, try to find U, V, W with small integer entries
    
    # Use a smarter approach: exploit symmetry and structure
    # For matrix multiplication, there are known lower bounds and constructions
    
    # The Strassen-like approach: try to find decompositions where
    # some columns of U, V, W have patterns that cancel out
    
    # Try rank from rank_school-1 down to max(n*m, m*p, p*n)
    min_possible_rank = max(n*m, m*p, p*n)
    
    for target_rank in range(rank_school - 1, min_possible_rank - 1, -1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of this rank
        # Use a structured search: assume entries are in {-1, 0, 1}
        # and try to find valid U, V, W
        
        # For small ranks, we can use a more systematic approach
        # Key insight: the tensor has a specific structure we can exploit
        
        # Try a few random structured initializations
        max_attempts = 50 if target_rank >= 20 else 100
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            np.random.seed(attempt)
            U = np.zeros((n*m, target_rank), dtype=np.int64)
            V = np.zeros((m*p, target_rank), dtype=np.int64)
            W = np.zeros((p*n, target_rank), dtype=np.int64)
            
            # Initialize with random sparse patterns
            for r in range(target_rank):
                # Choose random non-zero entries
                u_nonzero = np.random.choice(n*m, size=max(1, n*m//3), replace=False)
                v_nonzero = np.random.choice(m*p, size=max(1, m*p//3), replace=False)
                w_nonzero = np.random.choice(p*n, size=max(1, p*n//3), replace=False)
                
                for idx in u_nonzero:
                    U[idx, r] = np.random.choice([-1, 1])
                for idx in v_nonzero:
                    V[idx, r] = np.random.choice([-1, 1])
                for idx in w_nonzero:
                    W[idx, r] = np.random.choice([-1, 1])
            
            if verify(U, V, W):
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
    
    # Convert to lists and return
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
