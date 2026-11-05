"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication.
    
    Uses a constructive algebraic approach with structured bilinear forms.
    Exploits the tensor structure to find a valid low-rank decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # First, let's try to find a valid decomposition with rank < 24
    # using a structured approach based on the tensor's algebraic properties.
    
    # The matrix multiplication tensor M has entries M_{(i,j),(j,k),(k,i)} = 1
    # for all valid i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    # We need: for all (i,k,a,b,c,d):
    # sum_r W[(k,i),r] * U[(a,b),r] * V[(c,d),r] = 1 if a=i, b=c, d=k; 0 otherwise
    
    # Strategy: Try rank values from 12 up to 24, using a structured search.
    # For each rank, we'll try to find integer/half-integer solutions.
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is valid."""
        rank = len(U[0])
        # Check all constraints
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                expected = 1 if (a == i and b == c and d == k) else 0
                                actual = 0.0
                                for r in range(rank):
                                    actual += W[k * n + i][r] * U[a * m + b][r] * V[c * p + d][r]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    def to_clean(x):
        """Convert to integer or half-integer if possible."""
        x_rounded = round(x)
        if abs(x - x_rounded) < 1e-9:
            return int(x_rounded)
        x_half = round(x * 2) / 2
        if abs(x - x_half) < 1e-9:
            return x_half
        return x
    
    # Try structured constructions for various ranks
    
    # Attempt 1: Use a pattern based on grouping by j
    # For each j, create columns that handle all (i,k) pairs
    
    # Let's try a direct construction with rank 20
    # Using the insight that we can share some structure across j values
    
    # Actually, let's use a computational approach:
    # Set up the linear system and try to find sparse solutions
    
    # The constraints are:
    # For each (i,k,a,b,c,d) where NOT (a=i and b=c and d=k):
    #   sum_r W[(k,i),r] * U[(a,b),r] * V[(c,d),r] = 0
    # For each (i,j,k):
    #   sum_r W[(k,i),r] * U[(i,j),r] * V[(j,k),r] = 1
    
    # This is a system of quadratic equations (products of variables).
    # Let's try a different approach: fix some structure and solve.
    
    # Approach: For small ranks, try to find solutions where each column
    # has a simple pattern (e.g., only 2-3 non-zero entries per matrix).
    
    # Let's try rank values from 12 to 23 and see if we can find valid solutions.
    
    import random
    
    best_rank = n * m * p  # Start with schoolbook
    best_solution = None
    
    # Try to find solutions with rank < 24
    for rank in range(12, 24):
        # Try multiple random initializations
        for attempt in range(50):
            random.seed(attempt * 1000 + rank)
            
            # Initialize with small random values
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Use a structured initialization
            for r in range(rank):
                # Each column handles a subset of the tensor
                # Pick random (i,j) for U, (j,k) for V, (k,i) for W
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                
                # Sometimes add additional non-zeros
                if random.random() < 0.3:
                    i2 = random.randint(0, n - 1)
                    U[i2 * m + j, r] = 1
                if random.random() < 0.3:
                    k2 = random.randint(0, p - 1)
                    V[j * p + k2, r] = 1
                if random.random() < 0.3:
                    i2 = random.randint(0, n - 1)
                    W[k * n + i2, r] = 1
            
            # Convert to list of lists
            U_list = [[U[i][r] for r in range(rank)] for i in range(n * m)]
            V_list = [[V[i][r] for r in range(rank)] for i in range(m * p)]
            W_list = [[W[i][r] for r in range(rank)] for i in range(p * n)]
            
            if verify(U_list, V_list, W_list, n, m, p):
                best_rank = rank
                best_solution = (U_list, V_list, W_list)
                break
        
        if best_solution is not None:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        # Clean up values
        U_clean = [[to_clean(x) for x in row] for row in U]
        V_clean = [[to_clean(x) for x in row] for row in V]
        W_clean = [[to_clean(x) for x in row] for row in W]
        return {"U": U_clean, "V": V_clean, "W": W_clean}
    
    # If no better solution found, return schoolbook
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
# EVOLVE-BLOCK-END
