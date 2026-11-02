"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition satisfies all n*m*p tensor equations."""
    rank = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > tol:
                    return False
    return True

def _verify(U, V, W):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = len(W[0])
    rng = np.random.default_rng(12345)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                lA = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                lB = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                C_computed[i, k] += W[k*n+i, r] * lA * lB
    return np.allclose(C_computed, C_expected, atol=1e-8)

def _rank_11_solution():
    """Return the known rank-11 decomposition (Strassen + schoolbook for col 2)."""
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen part (terms 0-6) for 2x2 * 2x2 (cols 0,1 of B)
    # Term 0: (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[4,0]=1
    W[0,0]=1; W[2,0]=1
    
    # Term 1: (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    W[1,1]=1; W[2,1]=-1
    
    # Term 2: A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[4,2]=-1
    W[1,2]=1; W[2,2]=1
    
    # Term 3: A11(B10-B00)
    U[3,3]=1; V[3,3]=1; V[0,3]=-1
    W[0,3]=1; W[1,3]=1
    
    # Term 4: (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[4,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # Term 5: (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
    W[2,5]=1
    
    # Term 6: (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[3,6]=1; V[4,6]=1
    W[0,6]=1
    
    # Schoolbook for col 2 (terms 7-10)
    U[0,7]=1; V[2,7]=1; W[4,7]=1
    U[1,8]=1; V[5,8]=1; W[4,8]=1
    U[2,9]=1; V[2,9]=1; W[5,9]=1
    U[3,10]=1; V[5,10]=1; W[5,10]=1
    
    return U, V, W

def _try_rank_10_search(deadline):
    """
    Attempt to find a rank-10 decomposition.
    Strategy: Start from rank-11, drop one term, and optimize the remaining 10 terms
    to satisfy all 12 equations. Use multiple random initializations and L-BFGS-B.
    """
    n, m, p = DIMENSIONS
    rank = 10
    
    # Get the base rank-11 solution
    U11, V11, W11 = _rank_11_solution()
    
    def objective(params):
        U_flat = params[0:4*rank]
        V_flat = params[4*rank:4*rank+6*rank]
        W_flat = params[4*rank+6*rank:4*rank+6*rank+6*rank]
        
        U = U_flat.reshape((4, rank))
        V = V_flat.reshape((6, rank))
        W = W_flat.reshape((6, rank))
        
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    obj += (val - 1)**2
        return obj
    
    rng = np.random.default_rng(42)
    best_obj = float('inf')
    best_params = None
    
    # Try dropping each of the 11 terms from the rank-11 solution
    # and see if we can close the gap by adjusting the remaining 10.
    for drop_idx in range(11):
        if time.time() > deadline - 10:
            break
            
        # Create initial guess by removing the dropped term
        keep_indices = [i for i in range(11) if i != drop_idx]
        U0 = U11[:, keep_indices]
        V0 = V11[:, keep_indices]
        W0 = W11[:, keep_indices]
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Try a few random perturbations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            if attempt == 0:
                p_try = params0.copy()
            else:
                p_try = params0 + rng.standard_normal(len(params0)) * 0.1
            
            try:
                from scipy.optimize import minimize
                result = minimize(objective, p_try, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-12})
                
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
                    
                    if best_obj < 1e-8:
                        break
                        
            except Exception:
                continue
                
        if best_obj < 1e-8:
            break
            
    if best_params is not None and best_obj < 1e-6:
        U_flat = best_params[0:4*rank]
        V_flat = best_params[4*rank:4*rank+6*rank]
        W_flat = best_params[4*rank+6*rank:4*rank+6*rank+6*rank]
        U = U_flat.reshape((4, rank))
        V = V_flat.reshape((6, rank))
        W = W_flat.reshape((6, rank))
        
        if _check_decomposition(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank 10
    result_10 = _try_rank_10_search(deadline)
    if result_10 is not None:
        U, V, W = result_10
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to rank 11 (Strassen + schoolbook)
    U, V, W = _rank_11_solution()
    
    # Verify the decomposition
    if not _verify(U.tolist(), V.tolist(), W.tolist()):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = 12
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
