"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    # We seek a rank-r decomposition of the 2x4x5 all-ones tensor.
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Try various ranks and use a structured approach.
    # Known: rank(2x2x2)=7, rank(2x2x3)=8, rank(2x3x3)=11
    # For 2x4x5, let's try to build from blocks with shared structure.
    
    # Strategy: Use a decomposition based on partitioning.
    # Partition j = {0,1} and {2,3}, k = {0,1,2} and {3,4}
    # This gives 4 sub-tensors: 2x2x3, 2x2x2, 2x2x3, 2x2x2
    
    # Let me try a direct optimization approach with small rank.
    # Start with rank 18 and see if we can find an exact solution.
    
    deadline = time.time() + 140
    
    def try_rank(r):
        """Try to find exact decomposition of rank r using structured initialization."""
        if time.time() > deadline:
            return None
        
        # Use a structured initialization based on block decomposition
        # with some shared terms.
        
        # Initialize with random rational-like values and refine
        # For exactness, we need integer or half-integer entries.
        
        # Try a specific structured decomposition:
        # Based on the idea that 2xm*p can be done with rank ~ 2*max(m,p) + min(m,p) - 2
        
        # Let me try rank 18 with a specific pattern.
        
        # Use the following approach:
        # For each i in {0,1}, we have a 4x5 matrix multiplication.
        # We can use the fact that rank(2xm*p) <= rank(m*p) + rank(2x(m-1)*p) + ...
        
        # Actually, let me just try to solve it numerically and round.
        
        # Set up the system: we need U (8xr), V (20xr), W (10xr)
        # such that for all (i,j,k): sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
        
        # This is a system of 40 equations in 8r + 20r + 10r = 38r unknowns.
        # For r=18, that's 684 unknowns and 40 equations - very underdetermined.
        
        # Let me try a specific construction:
        # Use the 2x2x2 Strassen decomposition as a building block.
        
        # Strassen for 2x2x2:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B11-B21)
        # M4 = A22*(B22-B12)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For 2x4x5, let me try a block approach:
        # Split j into two groups of 2, k into groups of 2 and 3.
        
        # Let me try a direct numerical approach:
        # Initialize U, V, W with small random values, then use iterative refinement.
        
        np.random.seed(42)
        
        # For exactness, let's work with a structured approach.
        # Try rank 18.
        
        # Use a pattern based on Hadamard-like structure.
        
        # Actually, let me try a simpler approach: use the fact that
        # rank(2xm*p) <= 2*rank(m*p) - 1 or similar subadditivity.
        
        # Let me just try several ranks from low to high and use
        # a numerical solver to find approximate solutions, then round.
        
        for r in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
            if time.time() > deadline:
                break
            
            # Try to find a solution of rank r
            # Use a structured initialization
            
            # Initialize U, V, W
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Use a random initialization with values in {-1, 0, 1}
            for c in range(r):
                for idx in range(n * m):
                    U[idx, c] = np.random.choice([-1, 0, 1])
                for idx in range(m * p):
                    V[idx, c] = np.random.choice([-1, 0, 1])
                for idx in range(p * n):
                    W[idx, c] = np.random.choice([-1, 0, 1])
            
            # Check if this gives the right tensor
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if val != 1:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            
            if success:
                # Convert to lists
                U_list = [[int(U[i, j]) for j in range(r)] for i in range(n*m)]
                V_list = [[int(V[i, j]) for j in range(r)] for i in range(m*p)]
                W_list = [[int(W[i, j]) for j in range(r)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    result = try_rank(18)
    if result is not None:
        return result
    
    # Fallback to schoolbook
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
