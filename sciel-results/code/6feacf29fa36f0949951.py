"""Rewritten program for matrix multiplication 3x4 by 4x5 rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS

# Deadline guard: finish well within 120s (use 110s to be safe)
_DEADLINE = 110.0
_START_TIME = None

def _elapsed():
    return time.time() - _START_TIME

def _check_deadline():
    return _elapsed() > _DEADLINE

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # The schoolbook rank is n*m*p = 60.
    # For 3x4 * 4x5, the tensor is T_{i,j,k} = A_{i,j} * B_{j,k}.
    # We want to find rank r such that there exist U (12 x r), V (20 x r), W (15 x r)
    # with T = sum_c U[:,c] V[:,c] W[:,c] (in the appropriate tensor flattening).
    #
    # The tensor for matrix multiplication C_{i,k} = sum_j A_{i,j} B_{j,k}
    # has entries T_{i,j,k} = delta_{ik} (in the sense that the "multiplication tensor"
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]).
    #
    # Actually, the matrix multiplication tensor M_{i,j,k} is defined so that
    # C_{i,k} = sum_{j} A_{i,j} * B_{j,k} = sum_{j} M_{i,j,k} * A_{i,j} * B_{j,k}
    # where M_{i,j,k} = 1 for all valid i,j,k. Wait, that's not right.
    #
    # The standard tensor decomposition: the multiplication tensor is a 3-tensor
    # T in A^* x B^* x C^* where T(a,b,c) = sum_{i,j,k} T_{i,j,k} a_{i,j} b_{j,k} c_{i,k}.
    # For matrix multiplication, T_{i,j,k} = 1 if (i,j,k) is a valid triple, else 0.
    # But actually T_{i,j,k} = delta_{i,i'} delta_{k,k'} ... no.
    #
    # Let me be precise. The matrix multiplication map is:
    # (A, B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # As a trilinear form: f(A,B,C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}^* 
    # where C^* is the dual. The tensor is T = sum_{i,j,k} e_{i,j}^* x e_{j,k}^* x e_{i,k}^*.
    #
    # So T_{(i,j), (j,k), (i,k)} = 1 for all i,j,k, and 0 otherwise.
    # The tensor lives in (R^{nm})^* x (R^{mp})^* x (R^{pn})^*.
    #
    # A rank-r decomposition means: T = sum_{c=1}^r u_c x v_c x w_c
    # where u_c in (R^{nm})^*, v_c in (R^{mp})^*, w_c in (R^{pn})^*.
    #
    # This means: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since T_{(i,j),(j,k),(i,k)} = 1)
    #
    # And for all other combinations, the sum must be 0. But since the tensor only
    # has non-zero entries at positions where the indices are consistent (j matches
    # between A and B, i and k match between A and C, etc.), we just need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # The schoolbook solution uses rank = n*m*p with U[i*m+j, i*m*p+j*p+k] = 1, etc.
    #
    # Can we do better? For small matrices, the minimum rank is known in some cases.
    # For 3x4 * 4x5, the lower bound is max(n*m, m*p, p*n) = max(12, 20, 15) = 20.
    # The upper bound from schoolbook is 60.
    #
    # Let's try to find a decomposition of rank 20 or slightly above using numerical optimization.
    # But the problem requires exact (integer/half-integer) entries.
    #
    # Strategy: Start with a numerical rank-r decomposition (using tensor decomposition methods),
    # then round to nearest half-integers and verify. Try increasing r from the lower bound.
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = 20
    # Try ranks from 20 upward.
    
    # First, let's build the constraint matrix. For each (i,j,k), we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is a system of n*m*p = 60 equations in r^3 variables (if U,V,W are r x r each... 
    # no, U is (n*m) x r, V is (m*p) x r, W is (p*n) x r).
    # Total unknowns: n*m*r + m*p*r + p*n*r = r*(n*m + m*p + p*n) = r*(12+20+15) = 47*r.
    #
    # The equations are cubic (trilinear) in the unknowns, so this is a hard problem.
    #
    # Alternative approach: Fix two of the matrices and solve for the third.
    # If we fix U and V, then for each c, we have a vector u_c and v_c.
    # Then W[:,c] must satisfy: for all i,j,k: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    #
    # This is still coupled across c.
    #
    # Let's try a different approach: use the fact that for matrix multiplication,
    # there are known fast algorithms. For 3x4 * 4x5, we can pad to 4x4 * 4x4 or 5x5 * 5x5
    # and use Strassen-like methods, but those don't directly give the exact tensor decomposition
    # for the unpadded case.
    #
    # Actually, let's just try to find ANY decomposition with rank < 60 using a randomized
    # approach with rounding, or use a known result.
    #
    # For 2x2 * 2x2, Strassen gives rank 7. For 3x3 * 3x3, the best known is 23.
    # For our case 3x4 * 4x5, let's try to find a decomposition.
    #
    # Let's try a simple approach: use the structure that C = A*B can be computed by
    # splitting the middle dimension. But that doesn't reduce rank.
    #
    # Let's try a numerical optimization approach with scipy.
    
    try:
        from scipy.optimize import minimize
        import random
        random.seed(42)
        np.random.seed(42)
        
        # Try ranks from 20 to 59
        best_rank = n * m * p  # 60
        best_decomp = None
        
        for rank in range(20, 60):
            if _check_deadline():
                break
                
            # Try a few random initializations
            for trial in range(5):
                if _check_deadline():
                    break
                    
                # Initialize U, V, W randomly
                U_init = np.random.randn(n*m, rank) * 0.5
                V_init = np.random.randn(m*p, rank) * 0.5
                W_init = np.random.randn(p*n, rank) * 0.5
                
                # Objective: sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                def objective(params):
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
                                error += (s - 1.0)**2
                    return error
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 200, 'ftol': 1e-10})
                
                if result.fun < 1e-6:
                    U = result.x[:n*m*rank].reshape(n*m, rank)
                    V = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Round to nearest half-integer
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(rank):
                                    s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                                if abs(s - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_rank = rank
                        best_decomp = (U_r, V_r, W_r)
                        break  # Found a valid decomposition for this rank
            
            if best_decomp is not None:
                break  # Found the best rank for this trial
        
        if best_decomp is not None:
            U, V, W = best_decomp
            # Convert to lists of Fractions for exactness
            U_list = [[Fraction(int(round(x*2)), 2) for x in U[row]] for row in range(n*m)]
            V_list = [[Fraction(int(round(x*2)), 2) for x in V[row]] for row in range(m*p)]
            W_list = [[Fraction(int(round(x*2)), 2) for x in W[row]] for row in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
        
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
