"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition for n x m * m x p
    # C_{ik} = sum_j A_{ij} B_{jk}
    # Tensor T_{ijk} = 1
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    R_schoolbook = n * m * p
    
    U = np.zeros((n * m, R_schoolbook), dtype=np.float64)
    V = np.zeros((m * p, R_schoolbook), dtype=np.float64)
    W = np.zeros((p * n, R_schoolbook), dtype=np.float64)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row (i, j) -> i*m + j
                # Index in V: row (j, k) -> j*p + k
                # Index in W: row (k, i) -> k*n + i (Trace convention W[k*n+i, r])
                
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Try to improve using Winograd-style transformations
    # For 4x5 * 5x5, we can try to reduce rank by exploiting structure
    
    # Winograd's method for matrix multiplication uses the idea of:
    # 1. Transform A and B using linear combinations
    # 2. Perform fewer multiplications
    # 3. Transform back
    
    # For general m x m multiplication, Winograd showed rank can be less than m^3
    # For 5x5, the known best is around 49-50 (vs 125 schoolbook)
    # But for 4x5 * 5x5, we need to be more careful
    
    # Let's try a simple Winograd-like approach:
    # Group the middle dimension into blocks and use linear combinations
    
    # Actually, let's try a more systematic approach:
    # Use the fact that we can combine terms when possible
    
    # For now, return the schoolbook decomposition which is guaranteed correct
    # The rank is 100 (4*5*5)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
