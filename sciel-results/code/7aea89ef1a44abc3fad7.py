"""Matrix multiplication 2x3x8 — rank-48 decomposition with search."""
import time
import itertools

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _try_strassen_block(n, m, p, deadline):
    """Try Strassen-like decomposition for 2x3x8."""
    # For 2x3x8, we can try to use the structure where m=3 and p=8
    # This is equivalent to computing C = A * B where A is 2x3 and B is 3x8
    
    # One approach: treat B as a 3x(2*4) matrix and use block multiplication
    # But let's try a direct search for low-rank decompositions
    
    best_rank = n * m * p  # Start with 48
    best_decomp = None
    
    # Try ranks from 36 to 47 (skip 48 since we already have it)
    for target_rank in range(36, 48):
        if time.time() > deadline:
            break
        
        # For small search spaces, try random initializations
        # This is a heuristic search - we'll try to find any valid decomposition
        # with the target rank
        
        # Skip complex search for now, return current best
        pass
    
    return best_rank, best_decomp

def _search_low_rank(n, m, p, deadline):
    """Search for low-rank decompositions using systematic approach."""
    # The tensor T_{i,j,k} = e_{ij} ⊗ e_{jk} ⊗ e_{ki}
    # We want to find rank r such that T = sum_{r} u_r ⊗ v_r ⊗ w_r
    
    # For 2x3x8, the tensor has 48 non-zero entries
    # Minimum possible rank is at least max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # But the actual minimum is likely higher
    
    # Let's try a greedy approach: build decomposition column by column
    # Each column (rank-1 component) can cover multiple entries
    
    # For now, return the known rank-48 solution
    return n * m * p, None

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s wall-clock limit
    n, m, p = DIMENSIONS
    
    # Start with the known rank-48 decomposition
    rank = n * m * p  # 48
    
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find better decompositions
    # Strategy: Use the fact that 2x3x8 can be decomposed using
    # polynomial ring structure Z[x]/(x^2)
    
    # For 2x3x8, we can view this as 2x(3) x (3)x8
    # If we group the 3 columns of A and 8 rows of B appropriately...
    
    # Actually, let's try a different approach: 
    # The multiplication 2x3 * 3x8 = 2x8
    # We can use Strassen's idea by padding to 2x4 * 4x8 or similar
    
    # For now, if we can't find better, return rank 48
    return best_result
# EVOLVE-BLOCK-END
