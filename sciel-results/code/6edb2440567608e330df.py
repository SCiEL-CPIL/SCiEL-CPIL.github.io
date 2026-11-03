import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, leave 10s margin for 150s limit

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the tensor."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # Reshape: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # Compute sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Result should be 1 for all (i,j,k)
    result = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                result[i, j, k] = s
    return np.allclose(result, 1.0, atol=1e-9)

def _try_round(U, V, W, n, m, p):
    """Try to round entries to half-integers and verify."""
    # We try rounding to nearest half-integer
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def _als_decompose(n, m, p, rank, max_iters=100, tol=1e-9):
    """
    Perform ALS to find a rank decomposition.
    """
    rng = np.random.default_rng()
    U = rng.standard_normal((n * m, rank)) * 0.1
    V = rng.standard_normal((m * p, rank)) * 0.1
    W = rng.standard_normal((p * n, rank)) * 0.1
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            return None
            
        # Update U
        for i in range(n):
            for j in range(m):
                # Solve for U[i*m+j, :]
                # Equation: sum_c U[i,j,c] * (sum_k V[j,k,c] * W[k,i,c]) = 1 for all k
                # Let M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # We want U_vec such that M @ U_vec = 1 (vector of ones of size p)
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                target = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(M, target, rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # Solve for V[j*p+k, :]
                # Equation: sum_c V[j,k,c] * (sum_i U[i*m+j, c] * W[k*n+i, c]) = 1 for all i
                # Let M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                # We want V_vec such that M @ V_vec = 1 (vector of ones of size n)
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                target = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(M, target, rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # Solve for W[k*n+i, :]
                # Equation: sum_c W[k,i,c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = 1 for all j
                # Let M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # We want W_vec such that M @ W_vec = 1 (vector of ones of size m)
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                target = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(M, target, rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i,j,k] = np.sum(U3[i,j,:] * V3[j,k,:] * W3[k,i,:])
        res = np.linalg.norm(recon - np.ones((n,m,p)))
        if res < tol:
            break
            
    if time.time() - _START_TIME > _TIME_LIMIT - 5:
        return None

    if res < 1e-7:
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        rounded = _try_round(U_list, V_list, W_list, n, m, p)
        if rounded:
            return rounded
    return None

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # Search for better ranks
    # Lower bound is max(n*m, m*p, p*n) = 28
    # Try from 28 up to 55
    for rank in range(28, rank_school):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        found = False
        # Try multiple random initializations
        for trial in range(20):
            if time.time() - _START_TIME > _TIME_LIMIT - 5:
                break
            
            res = _als_decompose(n, m, p, rank)
            if res is not None:
                U_r, V_r, W_r = res
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                found = True
                # If we found a valid one at this rank, we can stop searching for this rank
                # But we want the MINIMUM rank, so if we find one at rank R, 
                # we should check if a lower rank exists. 
                # Since we iterate from low to high, the first valid rank we find is the minimum 
                # (among those we successfully find).
                # However, ALS might fail to find a solution even if one exists.
                # So we continue to next rank only if we didn't find one, 
                # or if we want to be sure. 
                # Given time constraints, we return the first valid one found in ascending order.
                return best_result
        
    return best_result
# EVOLVE-BLOCK-END
