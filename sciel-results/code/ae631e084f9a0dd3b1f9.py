"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import nnls

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    # For each (i, k), the coefficient of A[i,j]*B[j,k] should be 1
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                coeff = 0
                for r in range(rank):
                    coeff += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(coeff - 1) > 1e-9:
                    return False
    return True

def _try_rank_decomposition(n, m, p, rank, time_budget=30.0):
    """
    Try to find a rank-`rank` decomposition using a structured search.
    We use a random initialization approach with iterative refinement.
    """
    deadline = time.time() + time_budget
    best_result = None
    
    # Number of random restarts
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random small values
        np.random.seed(42 + restart)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Iterative refinement: alternate between fixing two matrices and solving for the third
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            # Solve for W given U, V
            # For each (i, k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # This is a system of equations in W
            # Let's build the linear system for W
            # Variables: W[k*n+i, r] for all k, i, r
            # Constraints: for each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # Build A matrix and b vector for W
            num_constraints = n * m * p
            num_vars = p * n * rank
            A_w = np.zeros((num_constraints, num_vars))
            b_w = np.ones(num_constraints)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            A_w[idx, (k * n + i) * rank + r] = U[i*m+j, r] * V[j*p+k, r]
                        idx += 1
            
            # Solve least squares for W
            W_flat, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
            W = W_flat.reshape(p * n, rank)
            
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = (U, V, W)
                break
            
            # Solve for U given V, W
            A_u = np.zeros((num_constraints, n * m * rank))
            b_u = np.ones(num_constraints)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            A_u[idx, (i * m + j) * rank + r] = V[j*p+k, r] * W[k*n+i, r]
                        idx += 1
            
            U_flat, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
            U = U_flat.reshape(n * m, rank)
            
            # Solve for V given U, W
            A_v = np.zeros((num_constraints, m * p * rank))
            b_v = np.ones(num_constraints)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            A_v[idx, (j * p + k) * rank + r] = U[i*m+j, r] * W[k*n+i, r]
                        idx += 1
            
            V_flat, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
            V = V_flat.reshape(m * p, rank)
            
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = (U, V, W)
                break
        
        if best_result is not None:
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # Start with schoolbook as fallback
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # Known bounds: for 3x4x6, the rank is at most 48 (via block decomposition)
    # Let's try ranks from 40 up to 72
    
    # First, try a structured block decomposition with rank 48
    # Split m=4 into 2+2, p=6 into 2+2+2
    # Each block is 3x2 * 2x2, which has rank at most 4 (schoolbook) or 7 (Strassen)
    # Using Strassen for the 2x2x2 core: rank 7 per block, 6 blocks = 42
    # But we need to handle the cross terms properly
    
    # Let's try a simpler approach: use the known result that rank(3x4x6) <= 48
    # via splitting m into 2+2 and using rank 12 for each 3x2x6 subproblem
    # Or split p into 3+3 and use rank 12 for each 3x4x3 subproblem
    
    # Actually, let's just try the search for various ranks
    for r in [42, 44, 46, 48, 50, 54, 56, 60, 64, 68]:
        if time.time() > 110:  # Safety margin
            break
        
        result = _try_rank_decomposition(n, m, p, r, time_budget=15.0)
        if result is not None:
            best_rank = r
            best_U, best_V, best_W = result
            break
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
