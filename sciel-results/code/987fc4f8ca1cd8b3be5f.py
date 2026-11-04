"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid i,j,k
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Try to find a lower rank decomposition
    # Lower bound is max(n*m, m*p, p*n) = 9, but likely higher.
    # Known rank for M_{2,3,3} is 14.
    
    def check_decomposition(U, V, W):
        """Verify if U, V, W exactly decompose T."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for l in range(r):
                        val += U[i * m + j, l] * V[j * p + k, l] * W[k * n + i, l]
                    if not np.isclose(val, T[i * m + j, j * p + k, k * n + i], atol=1e-9):
                        return False
        return True

    def find_decomposition(r, max_trials=10000):
        """
        Search for a rank r decomposition using randomized V, W and solving for U.
        """
        np.random.seed(42 + r) # Fixed seed for reproducibility in trials
        
        for trial in range(max_trials):
            # Random V and W with small integer entries
            # Using -1, 0, 1
            V = np.random.randint(-1, 2, (m * p, r))
            W = np.random.randint(-1, 2, (p * n, r))
            
            # Solve for U
            U = np.zeros((n * m, r))
            valid = True
            
            for i in range(n):
                for j in range(m):
                    # Construct A_ij of shape (p, r)
                    # A_ij[k, l] = V[j*p+k, l] * W[k*n+i, l]
                    A_ij = np.zeros((p, r))
                    for k in range(p):
                        for l in range(r):
                            A_ij[k, l] = V[j * p + k, l] * W[k * n + i, l]
                    
                    # Solve A_ij @ U_ij = ones(p)
                    # U_ij is a vector of length r
                    try:
                        # Use lstsq to find least squares solution
                        U_ij, residuals, rank, sv = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)
                        # Check if the solution is exact (residual close to 0)
                        if np.allclose(A_ij @ U_ij, np.ones(p), atol=1e-9):
                            U[i * m + j, :] = U_ij
                        else:
                            valid = False
                            break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    if not valid:
                        break
            
            if valid:
                # Verify the full decomposition
                if check_decomposition(U, V, W):
                    # Convert to lists for output
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
        return None

    # Try ranks from 10 up to 18
    # Start from 10 as a reasonable guess, go up to 18 (schoolbook)
    for r in range(10, 19):
        result = find_decomposition(r, max_trials=5000)
        if result:
            return result

    # Fallback to schoolbook rank 18
    rank = n * m * p
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
# EVOLVE-BLOCK-END
