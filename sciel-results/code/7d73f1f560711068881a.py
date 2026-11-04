"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the target tensor: T[i,j,k] = 1 if the product term (A[i,j] * B[j,k]) exists
    # The decomposition must satisfy: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Actually, the standard matrix multiplication tensor is:
    # T_{i,j,k} = A_{i,j} * B_{j,k}
    # A rank-r decomposition gives: T_{i,j,k} = sum_{c=1}^{r} U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # And the convention for W is W[k*n+i, c]
    
    # Let's verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    # because T[i,j,k] = 1 for all valid i,j,k in matrix multiplication (each term is A[i,j]*B[j,k])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_initial_guess(n, m, p, rank, seed=0):
    """Create a random initial guess with integer/half-integer values."""
    rng = np.random.RandomState(seed)
    # Initialize with small random values, then round to half-integers
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    # Scale to reasonable magnitude
    U /= np.sqrt(rank)
    V /= np.sqrt(rank)
    W /= np.sqrt(rank)
    # Round to half-integers
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    # Ensure non-zero
    for c in range(rank):
        if np.all(U[:, c] == 0):
            U[0, c] = 1.0
        if np.all(V[:, c] == 0):
            V[0, c] = 1.0
        if np.all(W[:, c] == 0):
            W[0, c] = 1.0
    return U, V, W

def _objective(params, n, m, p, rank):
    """Objective function: sum of squared errors in the tensor decomposition."""
    U = params[:n*m*rank].reshape(n*m, rank)
    V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                error += (s - 1.0) ** 2
    return error

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # Schoolbook rank is n*m*p = 3*3*6 = 54
    # Try to find a lower rank
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Also rank >= n*p + m*p - 1 (for some cases), but let's just try from low to high
    
    best_result = None
    
    # First, provide the schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    fallback = schoolbook()
    
    # Try ranks from a reasonable lower bound up to schoolbook rank
    # For 3x3 * 3x6, the minimal rank is not known to be much less than 54
    # Let's try a range of ranks and see if we can find exact decompositions
    
    # Actually, for matrix multiplication, the rank is typically close to n*m*p
    # For 3x3 * 3x6, let's try ranks from 40 to 54
    
    for rank in range(40, n*m*p + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            
            U, V, W = _build_initial_guess(n, m, p, rank, seed)
            
            # Check if initial guess already works
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
            
            # Use optimization to find a solution
            # Flatten parameters
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(x):
                Ux = x[:nm*rank].reshape(nm, rank)
                Vx = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                Wx = x[nm*rank+mp*rank:].reshape(pn, rank)
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(Ux[i*m+j], Vx[j*p+k])
                            # Wait, that's not right. The decomposition is:
                            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            # This is not a simple dot product of U and V
                            s = 0.0
                            for c in range(rank):
                                s += Ux[i*m+j, c] * Vx[j*p+k, c] * Wx[k*n+i, c]
                            err += (s - 1.0) ** 2
                return err
            
            try:
                res = minimize(obj, params0, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-6:
                    Ux = res.x[:nm*rank].reshape(nm, rank)
                    Vx = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    Wx = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Round to half-integers
                    Ux = np.round(Ux * 2) / 2
                    Vx = np.round(Vx * 2) / 2
                    Wx = np.round(Wx * 2) / 2
                    
                    if _verify(Ux.tolist(), Vx.tolist(), Wx.tolist(), n, m, p):
                        best_result = {"U": Ux.tolist(), "V": Vx.tolist(), "W": Wx.tolist()}
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    else:
        return fallback
# EVOLVE-BLOCK-END
