import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 2x4 by 4x4 matrix multiplication.
    Falls back to schoolbook (rank 32) if search doesn't find better.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    # U: shape (n*m, rank) = (8, rank)
    # V: shape (m*p, rank) = (16, rank)  
    # W: shape (p*n, rank) = (8, rank)
    # W follows trace convention W[k*n+i, r]
    
    # Schoolbook decomposition (rank 32)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=object)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=object)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def to_fraction_list(mat):
        """Convert numpy object matrix to list of lists of Fraction."""
        return [[Fraction(x) for x in row] for row in mat]
    
    # Return schoolbook as fallback
    best_result = {
        "U": to_fraction_list(U_sb),
        "V": to_fraction_list(V_sb),
        "W": to_fraction_list(W_sb)
    }
    
    # Try to find lower rank decompositions using random search with rational entries
    # For small ranks, try structured decompositions
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Where i in [0,n), j in [0,m), k in [0,p)
        # U is indexed as U[i*m+j, r]
        # V is indexed as V[j*p+k, r]
        # W is indexed as W[k*n+i, r]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1  # T[i,j,k] = 1 if this is the standard matrix mult tensor
                    actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(float(actual - expected)) > 1e-10:
                        return False
        return True
    
    # Try rank 20, 24, 28 with random rational search
    # Use a simple approach: try to find sparse decompositions
    
    # For now, try a structured approach for rank 24
    # Split into two 2x4*4x2 blocks and use known results
    
    # Actually, let me try a more systematic search for small ranks
    # with random initialization and local optimization
    
    np.random.seed(42)
    
    for target_rank in [20, 22, 24, 26, 28]:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random rational matrices with small integer entries
            scale = 2
            U_r = np.random.randint(-scale, scale + 1, size=(n * m, target_rank))
            V_r = np.random.randint(-scale, scale + 1, size=(m * p, target_rank))
            W_r = np.random.randint(-scale, scale + 1, size=(p * n, target_rank))
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1
                        actual = sum(U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r] for r in range(target_rank))
                        if actual != expected:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Found a valid decomposition!
            best_result = {
                "U": [[int(x) for x in row] for row in U_r.tolist()],
                "V": [[int(x) for x in row] for row in V_r.tolist()],
                "W": [[int(x) for x in row] for row in W_r.tolist()]
            }
            break
        
        if best_result != to_fraction_list(U_sb):  # If we found something better
            break
    
    return best_result

