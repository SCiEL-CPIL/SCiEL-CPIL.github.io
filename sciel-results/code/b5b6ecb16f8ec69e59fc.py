"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p

    # Build schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank), dtype=np.float64)
    V = np.zeros((m * p, schoolbook_rank), dtype=np.float64)
    W = np.zeros((p * n, schoolbook_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank

    def _validate(U_arr, V_arr, W_arr, R):
        """Verify decomposition using random integer matrices."""
        for _ in range(30):
            A = np.random.randint(-3, 4, (n, m))
            B = np.random.randint(-3, 4, (m, p))
            C_expected = A @ B
            C_calc = np.zeros((n, p), dtype=np.float64)
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        for r in range(R):
                            s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    C_calc[i, k] = s
            if not np.allclose(C_calc, C_expected, atol=1e-9):
                return False
        return True

    if time.time() > deadline - 10:
        return {"U": [list(map(float, row)) for row in best_U],
                "V": [list(map(float, row)) for row in best_V],
                "W": [list(map(float, row)) for row in best_W]}

    # Strategy: Try to find a lower-rank decomposition using Strassen-style block recursion.
    # For 2x5x6, we can try to decompose into smaller blocks where known low-rank decompositions apply.
    
    # Key insight: The rank of matrix multiplication tensor is sub-multiplicative in a block sense.
    # If we can express the multiplication as a sum of smaller multiplications with known low-rank decompositions,
    # we can potentially achieve a lower total rank.
    
    # Known ranks:
    # 2x2x2: rank 7 (Strassen)
    # 2x2x1: rank 3
    # 2x1x2: rank 3
    # 1x2x2: rank 3
    # 2x3x3: unknown, but <= 18
    # 2x2x3: unknown, but <= 12
    
    # Let's try to decompose 2x5x6 by splitting m=5 into 2+3 and p=6 into 3+3.
    # A (2x5) = [A1 (2x2) A2 (2x3)]
    # B (5x6) = [B1 (2x3); B2 (3x3)]
    # C = A1 @ B1 + A2 @ B2
    
    # This gives us two sub-problems: 2x2x3 and 2x3x3.
    # If we can find good decompositions for these, we might beat 60.
    
    # For now, let's try a direct approach: random search for a lower-rank decomposition.
    # We'll use gradient-free optimization to minimize the Frobenius norm of the error
    # while constraining the rank to be less than schoolbook.
    
    # However, this is very challenging. Let's try a different approach:
    # Use the fact that the multiplication tensor has structure that can be exploited.
    
    # Alternative: Try to find a decomposition of rank R < 60 by solving a system of equations.
    # T[i*m+j, j*p+k, k*n+i] = sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a nonlinear system. We can try to solve it using iterative methods.
    
    # For simplicity, let's try a few specific ranks below 60 and see if we can find a valid decomposition.
    
    # Try ranks: 59, 58, 57, ... down to some minimum
    # For each rank, try to find U, V, W that satisfy the tensor equation.
    
    # We'll use a random initialization and then try to optimize using a simple iterative method.
    
    def try_rank(R):
        """Try to find a decomposition of rank R."""
        if time.time() > deadline - 5:
            return None
        
        # Random initialization
        U_r = np.random.randn(n * m, R)
        V_r = np.random.randn(m * p, R)
        W_r = np.random.randn(p * n, R)
        
        # Try to optimize using a simple gradient-free approach
        # We'll use a variant of the Nelder-Mead method, but simplified.
        
        # For now, let's just check if a random decomposition works (very unlikely)
        if _validate(U_r, V_r, W_r, R):
            return (U_r, V_r, W_r)
        
        # If random doesn't work, we need a better approach.
        # Let's try to use the schoolbook decomposition as a starting point
        # and try to "merge" terms.
        
        return None
    
    # Try a few ranks
    for R in range(schoolbook_rank - 1, max(40, schoolbook_rank - 20), -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(R)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = R
            break
    
    # Convert to list format
    return {"U": [list(map(float, row)) for row in best_U],
            "V": [list(map(float, row)) for row in best_V],
            "W": [list(map(float, row)) for row in best_W]}
# EVOLVE-BLOCK-END
