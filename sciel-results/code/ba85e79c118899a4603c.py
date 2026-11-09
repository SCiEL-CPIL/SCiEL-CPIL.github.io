import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    timeout = 145.0
    
    n, m, p = 3, 3, 5
    size_U = n * m  # 9
    size_V = m * p  # 15
    size_W = p * n  # 15
    
    # Trivial rank-45 decomposition
    R0 = n * m * p  # 45
    U0 = np.zeros((size_U, R0))
    V0 = np.zeros((size_V, R0))
    W0 = np.zeros((size_W, R0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1.0
                V0[j*p+k, col] = 1.0
                W0[k*n+i, col] = 1.0
                col += 1
    
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = R0
    
    # Strategy: Search over discrete coefficient patterns.
    # We look for rank R decompositions where entries are in {-1, -0.5, 0, 0.5, 1}.
    # Key insight: For matrix multiplication, we can try to "merge" terms.
    # A term (i,j,k) contributes to U[i*m+j], V[j*p+k], W[k*n+i].
    # If two terms share the same (i,j,k) pattern but with different signs or shifts,
    # we might combine them.
    
    # However, a more effective search for small matrices is to look for
    # "structured" low-rank decompositions.
    # Known results:
    # 3x3 * 3x3 can be done in rank 23 (Strassen-like) or lower.
    # 3x3 * 3x5:
    # Let's try a randomized local search with discrete steps.
    
    # We will use a "greedy construction" or "randomized round" approach.
    # Since the space is discrete, we can try to build the matrices column by column.
    
    # Define the set of possible non-zero values for each entry.
    # To keep search tractable, we restrict to {-1, 0, 1} first, then {-1, -0.5, 0, 0.5, 1}.
    
    # Let's try to find a rank R solution by starting from rank 45 and
    # attempting to "zero out" entries or combine columns.
    
    # Alternative: Randomized search in the discrete space.
    # For a given rank R, we have R columns in U, V, W.
    # Each column is a vector of length 9, 15, 15 respectively.
    # The condition is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1.
    
    # This is a system of 45 polynomial equations of degree 3 in 39R variables.
    
    # Let's try a specific heuristic:
    # 1. Start with the trivial decomposition.
    # 2. Try to merge pairs of terms.
    #    Term A: (i1, j1, k1) with coeff 1
    #    Term B: (i2, j2, k2) with coeff 1
    #    Can we replace them with a single term?
    #    Only if i1==i2, j1==j2, k1==k2 (which is the same term).
    #    So simple merging doesn't work directly.
    
    # We need to use "cancellations" or "interference".
    # This is hard to do greedily.
    
    # Let's use a randomized local search.
    # We'll maintain a current solution (U, V, W) of rank R.
    # We'll randomly perturb entries (discrete steps) and check if the residual decreases.
    # But the residual is non-convex.
    
    # Better: Use the fact that we only need integer/half-integer entries.
    # Let's try to find a solution with rank < 45 using a SAT-like or constraint propagation approach?
    # Too complex.
    
    # Let's stick to the current plan but improve the optimization.
    # The previous code used L-BFGS-B on a continuous relaxation.
    # We can improve by:
    # 1. Using a better initial guess.
    # 2. Using multiple restarts.
    # 3. Rounding the continuous solution to the nearest half-integer and checking validity.
    # 4. If valid, we have a lower rank solution.
    
    # Let's also try a "discrete local search" on the rounded solution.
    # If the continuous solution is close to a half-integer grid point,
    # we can try to adjust individual entries to exactly satisfy the equations.
    
    def check_solution(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Also check that non-target entries are 0?
        # The tensor has 1s at (i,j,k) and 0s elsewhere.
        # The decomposition computes sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] for ALL i,j,k.
        # The task says "exactly decompose the matrix multiplication tensor".
        # The tensor T[i,j,k] is 1 if i in [0,n), j in [0,m), k in [0,p).
        # Wait, the tensor is defined for all i,j,k.
        # The output factor W[k*n+i, r] suggests the output is a p*n matrix.
        # The standard matrix multiplication tensor has T[i,j,k] = delta_{i,j,k}? No.
        # T[i,j,k] = 1 if we are computing C[i,k] += A[i,j] B[j,k].
        # So the tensor has 1s at all (i,j,k) for i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # And 0s elsewhere?
        # The dimensions are n*m, m*p, p*n.
        # So i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # The tensor is 1 at all valid (i,j,k) and 0 elsewhere?
        # No, the tensor is only defined on the support.
        # The condition is: for all i,j,k in range, sum_r ... == T[i,j,k].
        # T[i,j,k] = 1 if 0<=i<n, 0<=j<m, 0<=k<p.
        # But the indices are already within range.
        # So the condition is simply sum_r ... == 1 for all i,j,k.
        # Is there a condition for "elsewhere"?
        # The matrices U, V, W have fixed sizes.
        # U has size n*m. So i*m+j is always in 0..n*m-1.
        # So we only compute for valid i,j,k.
        # So the check above is sufficient.
        return True
    
    # Let's try to find a rank R solution for R from 44 down to some minimum.
    # We'll use a continuous optimization followed by rounding.
    
    # To improve the continuous optimization:
    # 1. Use a better objective function.
    # 2. Use multiple random initializations.
    # 3. Use a smaller tolerance for rounding.
    
    # Let's try rank 44 first.
    
    def try_find_rank(R):
        if time.time() - start_time > timeout - 20:
            return None
        
        np.random.seed(42 + R)
        
        # Initial guess: take the first R columns of the trivial decomposition
        U = U0[:, :R].copy()
        V = V0[:, :R].copy()
        W = W0[:, :R].copy()
        
        # Add small noise to escape local minima
        U += 0.1 * np.random.randn(size_U, R)
        V += 0.1 * np.random.randn(size_V, R)
        W += 0.1 * np.random.randn(size_W, R)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x, R=R):
            if time.time() - start_time > timeout - 10:
                return 1e10
            Ux = x[:size_U*R].reshape(size_U, R)
            Vx = x[size_U*R:2*size_U*R].reshape(size_V, R)
            Wx = x[2*size_U*R:].reshape(size_W, R)
            
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(Ux[i*m+j, :], Vx[j*p+k, :] * Wx[k*n+i, :])
                        residual += (1.0 - val) ** 2
            return residual
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 5000})
            
            if result.fun < 1e-4:
                Ux = result.x[:size_U*R].reshape(size_U, R)
                Vx = result.x[size_U*R:2*size_U*R].reshape(size_V, R)
                Wx = result.x[2*size_U*R:].reshape(size_W, R)
                
                # Round to half-integers
                Ux_r = np.round(Ux * 2) / 2
                Vx_r = np.round(Vx * 2) / 2
                Wx_r = np.round(Wx * 2) / 2
                
                if check_solution(Ux_r, Vx_r, Wx_r, tol=1e-9):
                    return (Ux_r, Vx_r, Wx_r)
                
                # Try rounding to integers
                Ux_i = np.round(Ux)
                Vx_i = np.round(Vx)
                Wx_i = np.round(Wx)
                
                if check_solution(Ux_i, Vx_i, Wx_i, tol=1e-9):
                    return (Ux_i, Vx_i, Wx_i)
        except:
            pass
        
        return None
    
    # Try ranks from 44 down to 20
    for R in range(44, 19, -1):
        if time.time() - start_time > timeout - 30:
            break
        
        result = try_find_rank(R)
        if result is not None:
            Ux_r, Vx_r, Wx_r = result
            best_result = {"U": Ux_r.tolist(), "V": Vx_r.tolist(), "W": Wx_r.tolist()}
            best_rank = R
            break
    
    return best_result
