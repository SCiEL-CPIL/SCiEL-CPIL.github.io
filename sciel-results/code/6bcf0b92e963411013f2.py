"""Matrix multiplication 3x5x6 rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# Known best rank for 3x5x6 is 75 (schoolbook). We try to beat this.
# Lower bounds: max(n*m, n*p, m*p, n+m+p-2) = max(15, 18, 30, 12) = 30.
# So we search ranks from 30 upwards.

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The tensor T[i,j,k] for (A*B)[i,k] = sum_j A[i,j]*B[j,k] must satisfy:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = U.shape[1]
    # Build the full tensor via the factorization
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Compare with the identity tensor T[i,j,k] = 1 if j==j (always 1 for the MM tensor)
    # Actually the MM tensor T[i,j,k] = delta_{j,j} = 1 for all i,j,k? No.
    # The matrix multiplication tensor is: T[i,j,k] such that (AB)_{ik} = sum_j T[i,j,k] A_{ij} B_{jk}
    # So T[i,j,k] = 1 for all i,j,k (it's the identity in the middle index pairing).
    # Wait no. T[i,j,k] = 1 if the indices match properly. For MM: T[i,j,k] = 1 for all valid i,j,k.
    # Actually the MM tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    
    # Let's verify by checking: for random A, B, does sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j]*B[j,k] == (AB)[i,k]?
    # Simpler: check that the tensor defined by the factors equals the MM tensor.
    # MM tensor: T[i,j,k] = 1 for all i,j,k (since (AB)_{ik} = sum_j A_{ij} B_{jk}, the coefficient of A_{ij}B_{jk} is 1).
    
    # Compute tensor from factors
    # T_fact[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a tensor contraction. Use einsum.
    T_fact = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # MM tensor is all ones
    T_mm = np.ones((n, m, p))
    
    if not np.allclose(T_fact, T_mm, atol=tol):
        return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using the trace convention directly."""
    rank = U.shape[1]
    # T_fact[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_fact = np.einsum('ir,jr,kr->ijk', U, V, W)
    T_mm = np.ones((n, m, p))
    diff = np.abs(T_fact - T_mm)
    return diff.max() < tol

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 90
    
    # Initialize with schoolbook
    best_rank = schoolbook_rank
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Search strategy: use random initialization + gradient-free optimization
    # or structured approaches.
    
    # For 3x5x6, known results: the rank is likely 75 or less.
    # Let's try a few approaches:
    
    # Approach 1: Try to construct a decomposition using structured patterns
    # The MM tensor T[i,j,k]=1 can be factored if we can find rank-r factors.
    
    # Approach 2: Use the fact that for small matrices, we can try specific rank values
    # and use numerical optimization.
    
    # Let's try ranks from max(n*m, n*p, m*p) = 30 up to 89
    # But 30 is very optimistic. Realistic lower bound is around 50-70.
    
    # Use a simple random restart + local search approach
    # For each rank r, try to find U, V, W such that the tensor matches.
    
    # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of n*m*p = 90 equations in 3*r unknowns (with scaling freedom).
    # For r >= 30, we have 3r >= 90 unknowns.
    
    # Strategy: Fix one of the matrices (say W) to simplify, then solve for U and V.
    # Or use iterative optimization.
    
    # Let's try a gradient-free approach: random initialization + coordinate descent
    # or use scipy's minimize.
    
    from scipy.optimize import minimize
    
    def tensor_error(U_flat, V_flat, W_flat, r):
        """Compute error of tensor decomposition."""
        U = U_flat.reshape(n * m, r)
        V = V_flat.reshape(m * p, r)
        W = W_flat.reshape(p * n, r)
        T_fact = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_mm = np.ones((n, m, p))
        return np.sum((T_fact - T_mm) ** 2)
    
    def try_rank(r, max_iters=100, seed=0):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Multiple random restarts
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            U0 = rng.randn(n * m, r) * 0.1
            V0 = rng.randn(m * p, r) * 0.1
            W0 = rng.randn(p * n, r) * 0.1
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def obj(x):
                Uf = x[:n*m*r].reshape(n*m, r)
                Vf = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                Wf = x[n*m*r+m*p*r:].reshape(p*n, r)
                Tf = np.einsum('ir,jr,kr->ijk', Uf, Vf, Wf)
                return np.sum((Tf - 1.0) ** 2)
            
            result = minimize(obj, x0, method='Nelder-Mead', 
                             options={'maxiter': max_iters, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.fun < 1e-6:
                Uf = result.x[:n*m*r].reshape(n*m, r)
                Vf = result.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                Wf = result.x[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Verify
                Tf = np.einsum('ir,jr,kr->ijk', Uf, Vf, Wf)
                if np.abs(Tf - 1.0).max() < 1e-6:
                    # Convert to list of lists with integer/half-integer entries
                    # The entries might not be integer/half-integer, so we need to check
                    # or round. The task says "integer/half-integer entries".
                    # Let's check if they're close to half-integers.
                    U_check = Uf * 2
                    if np.allclose(U_check, np.round(U_check), atol=1e-4) and \
                       np.allclose(Vf * 2, np.round(Vf * 2), atol=1e-4) and \
                       np.allclose(Wf * 2, np.round(Wf * 2), atol=1e-4):
                        U_int = np.round(Uf * 2).astype(int) / 2.0
                        V_int = np.round(Vf * 2).astype(int) / 2.0
                        W_int = np.round(Wf * 2).astype(int) / 2.0
                        # Final verification
                        Tf2 = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
                        if np.abs(Tf2 - 1.0).max() < 1e-9:
                            return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                    else:
                        # Not half-integer, but still valid if exact
                        # The task requires integer/half-integer entries
                        # So this doesn't qualify unless we can find a half-integer solution
                        pass
        
        return None
    
    # Try ranks from lower to upper
    # Lower bound: max(n*m, n*p, m*p) = max(15, 18, 30) = 30
    # But realistically, for 3x5x6, the rank is probably around 50-80.
    # Let's try a range of ranks.
    
    ranks_to_try = list(range(50, min(90, schoolbook_rank)))
    # Also try some specific promising ranks
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        result = try_rank(r, max_iters=50, seed=r)
        if result is not None:
            U_list, V_list, W_list = result
            best_rank = r
            best_U = U_list
            best_V = V_list
            best_W = W_list
            break  # Found a lower rank, stop
    
    # If no improvement found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
