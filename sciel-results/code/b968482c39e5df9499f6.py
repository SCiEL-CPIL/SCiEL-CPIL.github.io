"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # Tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Try rank from 18 upward (known lower bounds suggest ~20-25 for 3x3x3x4)
    # Use randomized search over GF(3)
    
    def solve_over_gf3(rank):
        """Try to find rank-r decomposition over GF(3)."""
        for attempt in range(5000):
            random.seed(attempt * 7919 + 12345)
            np.random.seed(attempt * 7919 + 12345)
            
            # Initialize U and V randomly over GF(3)
            U = np.random.randint(0, 3, size=(n*m, rank))
            V = np.random.randint(0, 3, size=(m*p, rank))
            
            W = np.zeros((p*n, rank), dtype=int)
            valid = True
            
            # For each (i, k), solve for W[k*n+i, :]
            for i in range(n):
                for k in range(p):
                    # Build system: for each j in range(m):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    # This is a linear system in W[k*n+i, :]
                    A = np.zeros((m, rank), dtype=int)
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = (U[i*m + j, c] * V[j*p + k, c]) % 3
                    
                    b = np.ones(m, dtype=int)
                    
                    # Solve A * w = b over GF(3)
                    # Use Gaussian elimination over GF(3)
                    aug = np.column_stack([A, b]) % 3
                    rows = m
                    cols = rank
                    rank_gauss = 0
                    pivot_cols = []
                    
                    for col in range(cols):
                        # Find pivot
                        pivot_row = -1
                        for row in range(rank_gauss, rows):
                            if aug[row, col] % 3 != 0:
                                pivot_row = row
                                break
                        if pivot_row == -1:
                            continue
                        # Swap
                        aug[[rank_gauss, pivot_row]] = aug[[pivot_row, rank_gauss]]
                        # Normalize pivot row
                        inv = 1 if aug[rank_gauss, col] % 3 == 1 else 2  # inverse of 2 mod 3 is 2
                        aug[rank_gauss] = (aug[rank_gauss] * inv) % 3
                        # Eliminate
                        for row in range(rows):
                            if row != rank_gauss and aug[row, col] % 3 != 0:
                                factor = aug[row, col] % 3
                                aug[row] = (aug[row] - factor * aug[rank_gauss]) % 3
                        pivot_cols.append(col)
                        rank_gauss += 1
                    
                    # Check consistency
                    for row in range(rank_gauss, rows):
                        if aug[row, cols] % 3 != 0:
                            valid = False
                            break
                    if not valid:
                        break
                    
                    # Extract solution (set free variables to 0)
                    w = np.zeros(rank, dtype=int)
                    for idx, col in enumerate(pivot_cols):
                        w[col] = aug[idx, cols] % 3
                    
                    W[k*n + i, :] = w
            
            if not valid:
                continue
            
            # Verify the decomposition
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val = (val + U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]) % 3
                        if val % 3 != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Convert to list of lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None
    
    # Try increasing ranks
    for rank in range(18, 37):
        result = solve_over_gf3(rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook decomposition
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
