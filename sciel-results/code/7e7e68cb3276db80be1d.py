"""Matrix multiplication 2,3,8 - rank-minimal decomposition search."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]
DEADLINE = 150.0
SAFETY_MARGIN = 20.0

def _build_tensor(n, m, p):
    """Build the matmul tensor T[i,j,k] = 1 for all valid (i,j,k)."""
    return np.ones((n, m, p))

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W correctly decompose the matmul tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False, val
    return True, 1.0

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Vectorized verification."""
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # Compute sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # For each (i,j,k): sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if np.max(np.abs(result - 1.0)) > tol:
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_optimization(n, m, p, rank, max_iter=500, seed=42):
    """Try to find a rank-r decomposition via optimization."""
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    def objective(x):
        Ux = x[:n*m*rank].reshape(n*m, rank)
        Vx = x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        Wx = x[n*m*rank+m*p*rank:].reshape(p*n, rank)
        U3 = Ux.reshape(n, m, rank)
        V3 = Vx.reshape(m, p, rank)
        W3 = Wx.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        # We want result == 1 everywhere
        return np.sum((result - 1.0)**2)
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', maxiter=max_iter,
                       options={'maxls': 20, 'ftol': 1e-12})
        if res.fun < 1e-8:
            Ux = res.x[:n*m*rank].reshape(n*m, rank)
            Vx = res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
            Wx = res.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
            return Ux, Vx, Wx
    except Exception:
        pass
    return None

def _try_structured_rank(n, m, p, rank):
    """Try structured decompositions with specific patterns."""
    # For 2x3x8, try various structured approaches
    # One idea: use the fact that n=2 is small
    
    # Try: decompose into rank contributions
    # Each rank element contributes a product of three vectors
    # U[:,r] is in R^(nm), V[:,r] in R^(mp), W[:,r] in R^(pn)
    
    # For small n, we can parameterize more cleverly
    # U[i*m+j, r] = u_r[i,j], V[j*p+k, r] = v_r[j,k], W[k*n+i, r] = w_r[k,i]
    
    # The constraint is: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k
    
    # For n=2: i in {0,1}
    # sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
    
    # This is a system of n*m*p = 48 equations in rank*3*n*m ... variables
    # Actually rank variables for each of U, V, W: rank*(nm + mp + pn) = rank*(6+24+16) = rank*46
    
    # For rank < 48, we have more variables than equations potentially
    # rank*46 > 48 when rank > 48/46 ≈ 1.04, so rank >= 2 gives more vars than eqs
    # But the equations are nonlinear (cubic in the variables per term)
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p  # 48
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Try to find lower rank decompositions
    # Known results: for 2x3 matrix mult, rank is 7 (vs schoolbook 6... wait)
    # Actually for 2x2 it's 7, for 2x3 it might be lower than 6
    # For 2x3x8: schoolbook is 48
    
    # Strategy: try ranks from some lower bound up
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Also rank >= n*m*p / something... 
    # Strassen-like: for 2x2x2, rank 7 < 8
    # For our case, let's try to find rank < 48
    
    # Try various ranks with optimization
    # Given time constraints, try a few promising ranks
    
    # First, try a structured approach for rank 32 (halving via Strassen-like)
    # Actually, let's try a direct search for small improvements
    
    # Try rank values: start from a reasonable lower bound
    # For 2x3x8, let's try ranks: 40, 42, 44, 46, 47, 48
    # And also try much lower: 24, 28, 32, 36
    
    candidate_ranks = [47, 46, 45, 44, 43, 42, 41, 40, 38, 36, 34, 32, 30, 28, 26, 24]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Try a few random seeds
        for seed in [42, 123, 7, 99, 256]:
            if time.time() > deadline:
                break
            
            result = _try_optimization(n, m, p, rank, max_iter=200, seed=seed)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify_fast(U_r, V_r, W_r, n, m, p, tol=1e-6):
                    best_rank = rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    break  # Found valid decomposition at this rank
    
    # Also try a structured approach: exploit the n=2 structure
    if time.time() < deadline and best_rank == 48:
        # Try a specific structured decomposition
        # For n=2, we have i in {0,1}
        # The condition is:
        # For i=0: sum_r u_r[0,j] * v_r[j,k] * w_r[k,0] = 1
        # For i=1: sum_r u_r[1,j] * v_r[j,k] * w_r[k,1] = 1
        
        # This means: for each j,k:
        # sum_r u_r[0,j] * v_r[j,k] * w_r[k,0] = 1
        # sum_r u_r[1,j] * v_r[j,k] * w_r[k,1] = 1
        
        # Let's define for each r:
        # a_r[j] = u_r[0,j], b_r[j] = u_r[1,j]
        # c_r[j,k] = v_r[j,k]
        # d_r[k] = w_r[k,0], e_r[k] = w_r[k,1]
        
        # Then: sum_r a_r[j]*c_r[j,k]*d_r[k] = 1 for all j,k
        #       sum_r b_r[j]*c_r[j,k]*e_r[k] = 1 for all j,k
        
        # This is two separate "matrix multiplication" problems:
        # A (3x8) * via c * D (8x3) = J (3x8 all ones)
        # B (3x8) * via c * E (8x3) = J (3x8 all ones)
        
        # Where the "multiplication" uses the shared c_r[j,k]
        
        # This is complex. Let's just rely on the optimization above.
        pass
    
    # Ensure we return valid result
    # Convert to lists as expected
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
