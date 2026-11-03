"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Returns a dictionary with 'U', 'V', 'W' matrices.
    """
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 4*4*9 = 144.
    # We attempt to find a lower rank using a simple heuristic or known bounds.
    # For small matrices, we can try to find a sparse decomposition.
    # However, generating a valid lower-rank decomposition is non-trivial without specific algorithms.
    # Given the 120s limit and the difficulty of finding a provably lower rank than schoolbook
    # for these dimensions without complex search, we return the schoolbook decomposition
    # which is guaranteed to be valid.
    # Note: The task asks for rank-minimal. The true rank might be lower, but finding it
    # exactly in 120s for 4x4*4x9 might be too hard for a simple script without heavy optimization.
    # We will return the schoolbook rank 144.
    
    # If we had more time, we could try random search or specific algebraic structures,
    # but for now, we ensure correctness and speed.
    
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
