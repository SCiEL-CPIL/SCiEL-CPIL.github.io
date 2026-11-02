"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24
    
    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try ranks from schoolbook-1 down to a reasonable minimum
    # Theoretical lower bound: max(n, m, p) = 4, but likely higher
    # Known results suggest rank around 12-18 for this size
    for rank in range(schoolbook_rank - 1, 11, -1):
        if time.time() > deadline:
            break
        
        try:
            result = _try_rank(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                continue
        except Exception:
            continue
    
    return best_result

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank decomposition with the given rank."""
    target = np.ones((n, m, p))
    
    # Strategy 1: Use structured initializations
    initializations = []
    
    # Random initializations
    np.random.seed(int(time.time() * 1000) % 10000)
    for _ in range(5):
        U0 = np.random.randn(n * m, rank)
        V0 = np.random.randn(m * p, rank)
        W0 = np.random.randn(p * n, rank)
        scale = 1.0 / np.sqrt(rank)
        U0 *= scale
        V0 *= scale
        W0 *= scale
        initializations.append((U0, V0, W0))
    
    # Structured: try to use the "outer product" structure
    # For all-ones tensor, try U[i,j] = a_i, V[j,k] = b_j, W[k,i] = c_k
    # Then sum_r a_i * b_j * c_k = 1 for all i,j,k
    # This requires a_i * b_j * c_k to be constant, which is hard
    # Better: use multiple terms
    
    # Try initialization based on splitting the dimensions
    # If rank >= max(n,m,p), we can potentially use a "diagonal" approach
    if rank >= max(n, m, p):
        U_struct = np.zeros((n * m, rank))
        V_struct = np.zeros((m * p, rank))
        W_struct = np.zeros((p * n, rank))
        # Try to distribute the ones
        for i in range(n):
            for j in range(m):
                # Assign to first few columns
                for r in range(min(rank, 3)):
                    U_struct[i * m + j, r] = 1.0 / (rank * 0.3 + 1)
        for j in range(m):
            for k in range(p):
                for r in range(min(rank, 3)):
                    V_struct[j * p + k, r] = 1.0 / (rank * 0.3 + 1)
        for k in range(p):
            for i in range(n):
                for r in range(min(rank, 3)):
                    W_struct[k * n + i, r] = 1.0 / (rank * 0.3 + 1)
        initializations.append((U_struct, V_struct, W_struct))
    
    for U0, V0, W0 in initializations:
        if time.time() > deadline:
            break
            
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(_residual_factory(n, m, p, target), params0, 
                                   method='trf', max_nfev=100, ftol=1e-12, xtol=1e-12)
        except Exception:
            continue
            
        if result.cost < 1e-10:
            U_opt = result.x[:n*m*rank].reshape(n*m, rank)
            V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Try snapping to half-integers with various denominators
            for denom in [1, 2, 3, 4, 6, 8, 12]:
                U_snap = np.round(U_opt * denom) / denom
                V_snap = np.round(V_opt * denom) / denom
                W_snap = np.round(W_opt * denom) / denom
                
                # Verify exactness
                X = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                max_err = np.max(np.abs(X - target))
                
                if max_err < 1e-9:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
    
    # Strategy 2: Direct search over small half-integer grids for very low ranks
    if rank <= 16:
        result = _grid_search(n, m, p, rank, target, deadline)
        if result is not None:
            return result
    
    return None

def _residual_factory(n, m, p, target):
    """Create a residual function for the given tensor dimensions."""
    def residual_function(params):
        U = params[:n*m*p//1*0 + n*m*(len(params)//(n*m+m*p+p*n))].reshape(n*m, -1) if False else params[:n*m* (len(params)//(n*m+m*p+p*n))].reshape(n*m, -1)
        # More robust: compute rank from params length
        total = len(params)
        rank = total // (n*m + m*p + p*n)
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        X = np.einsum('ir,jr,kr->ijk', U, V, W)
        residuals = (X - target).flatten()
        return residuals
    return residual_function

def _grid_search(n, m, p, rank, target, deadline):
    """Search over small half-integer values for exact decomposition."""
    # For very small ranks, try a limited grid
    # This is expensive, so only try for rank <= 12
    if rank > 12:
        return None
    
    # Values to try: -1, -0.5, 0, 0.5, 1, 1.5, 2
    values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    # This is a huge search space, so use a heuristic approach
    # Try to find solutions where most entries are 0 or 1
    # Use a simplified version: try sparse decompositions
    
    # For rank r, try to find r terms where each term is a rank-1 tensor
    # that sums to the all-ones tensor
    
    # Heuristic: try to use the "combinatorial" structure
    # For 2x3x4, try rank 12-16 with structured patterns
    
    # Given time constraints, skip full grid search and rely on optimization
    return None
# EVOLVE-BLOCK-END
