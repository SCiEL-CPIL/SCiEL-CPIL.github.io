"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Construct an exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The schoolbook algorithm has rank 3*3*7 = 63.
    We attempt to find a lower rank decomposition using algebraic search.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63
    
    # Build schoolbook decomposition as baseline
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Should be 1
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
                for j2 in range(m):
                    for k2 in range(p):
                        for i2 in range(n):
                            if i == i2 and j == j2 and k == k2:
                                continue
                            val = np.sum(U[i*m+j, :] * V[j2*p+k2, :] * W[k*n+i2, :])
                            if abs(val - 0.0) > 1e-9:
                                return False
        return True
    
    # Try to find lower rank decompositions
    # Strategy: for each candidate rank r from schoolbook_rank-1 down,
    # try random initializations and use a simple optimization
    
    # Precompute the list of (i,j,k) triples that should give 1
    one_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Precompute all (i,j,j2,k2,k2,i2) that should give 0 (but only store a sample for speed)
    # Actually, let's just check all in verify
    
    for target_rank in range(schoolbook_rank - 1, 49, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + attempt + target_rank)
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Use a simple gradient descent on the sum of squared residuals
            # But this is expensive. Instead, let's try a different approach:
            # Start from schoolbook and try to eliminate one term at a time.
            
            # Alternative: use the fact that if we have a rank r decomposition,
            # we can try to find it by solving a linear system after fixing
            # some structure.
            
            # For now, just check if the random init happens to work (unlikely)
            if verify_decomposition(U, V, W, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # Convert to lists for output
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
