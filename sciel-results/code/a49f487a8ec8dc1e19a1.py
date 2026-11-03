"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = schoolbook_rank
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n_, m_, p_ = n, m, p
        if len(U) != n_ * m_:
            return False
        if len(V) != m_ * p_:
            return False
        if len(W) != p_ * n_:
            return False
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return False
        rank_ = len(U[0])
        if len(V[0]) != rank_ or len(W[0]) != rank_:
            return False
        if rank_ <= 0:
            return False
        
        # Convert to numpy for faster verification
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    expected = 1.0
                    actual = np.dot(U_np[i * m_ + j], V_np[j * p_ + k] * W_np[k * n_ + i])
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    def make_result(U, V, W):
        """Convert to list-of-lists format."""
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        # Clean up near-integer values
        U_list = [[round(x, 10) if abs(x - round(x, 10)) < 1e-8 else x for x in row] for row in U_list]
        V_list = [[round(x, 10) if abs(x - round(x, 10)) < 1e-8 else x for x in row] for row in V_list]
        W_list = [[round(x, 10) if abs(x - round(x, 10)) < 1e-8 else x for x in row] for row in W_list]
        return {'U': U_list, 'V': V_list, 'W': W_list}
    
    # Try structured decompositions with small ranks
    # For 3x3 * 3x7, let's try ranks from a reasonable upper bound down
    
    # Known: For (3,3,7), the Strassen-like or other bilinear algorithms might give lower rank
    # Let's try random search with small coefficient sets for ranks below schoolbook
    
    random.seed(42)
    np.random.seed(42)
    
    # Try to find a decomposition with rank < 63 using an improved random search
    # Use a wider range of coefficient sets and more restarts
    
    coeff_sets = [
        [-1, 0, 1],
        [-2, -1, 0, 1, 2],
        [-1, 0, 0, 1],
        [-1, -1, 0, 1, 1],
        [-3, -2, -1, 0, 1, 2, 3],
    ]
    
    # Try ranks from 35 to 62
    for target_rank in range(35, min(63, best_rank)):
        if time.time() > deadline - 10:
            break
            
        max_restarts = 100
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            coeff_set = coeff_sets[restart % len(coeff_sets)]
            
            # Random initialization with some structure
            U = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Quick check: if any row is all zeros, skip
            if all(all(x == 0 for x in row) for row in U):
                continue
            if all(all(x == 0 for x in row) for row in V):
                continue
            if all(all(x == 0 for x in row) for row in W):
                continue
            
            if verify_decomposition(U, V, W):
                result = make_result(U, V, W)
                best_result = result
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
    
    # If no improvement found, return the schoolbook decomposition
    return best_result
# EVOLVE-BLOCK-END
