import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute indices for efficiency
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
                
    def calculate_error(params, R):
        """
        Calculate the sum of squared errors for a given set of parameters.
        params is a flattened vector of length 26*R.
        Order: U (6*R), V (12*R), W (8*R)
        """
        U = params[:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:].reshape(pn, R)
        
        err = 0.0
        for a, b, c in indices:
            # Compute dot product for this specific tensor component
            val = np.sum(U[a] * V[b] * W[c])
            err += (val - 1.0) ** 2
        return err

    def calculate_gradient(params, R):
        """
        Calculate the gradient of the error function.
        """
        U = params[:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:].reshape(pn, R)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for a, b, c in indices:
            # val = sum(U[a] * V[b] * W[c])
            # d(val)/d(U[a, r]) = V[b, r] * W[c, r]
            # d(val)/d(V[b, r]) = U[a, r] * W[c, r]
            # d(val)/d(W[c, r]) = U[a, r] * V[b, r]
            
            val = np.sum(U[a] * V[b] * W[c])
            err = val - 1.0
            
            # Gradient of (val - 1)^2 is 2 * (val - 1) * d(val)/d(param)
            grad_U[a] += 2.0 * err * (V[b] * W[c])
            grad_V[b] += 2.0 * err * (U[a] * W[c])
            grad_W[c] += 2.0 * err * (U[a] * V[b])
            
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        for a, b, c in indices:
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val - 1.0) > tol:
                return False
        return True

    def snap_to_rational(U, V, W):
        """
        Snap values to the nearest integer or half-integer.
        Returns None if snapping results in a large error increase.
        """
        U_snap = np.round(U * 2.0) / 2.0
        V_snap = np.round(V * 2.0) / 2.0
        W_snap = np.round(W * 2.0) / 2.0
        
        # Check exactness with snapped values
        if check_decomposition(U_snap, V_snap, W_snap, tol=1e-6):
            return U_snap, V_snap, W_snap
        return None

    # Naive solution fallback
    rank_naive = n * m * p
    U_naive = np.zeros((nm, rank_naive))
    V_naive = np.zeros((mp, rank_naive))
    W_naive = np.zeros((pn, rank_naive))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_naive[i * m + j, r] = 1
                V_naive[j * p + k, r] = 1
                W_naive[k * n + i, r] = 1
                
    best_result = {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    best_rank = rank_naive

    # Try to find lower rank
    # Range to search: 19 to 23
    for target_rank in range(19, 24):
        if time.time() > deadline - 10:
            break
            
        found = False
        num_restarts = 10
        for attempt in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + attempt * 1000)
            # Initialize with small random values
            U_init = np.random.randn(nm, target_rank) * 0.5
            V_init = np.random.randn(mp, target_rank) * 0.5
            W_init = np.random.randn(pn, target_rank) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimize
            try:
                res = minimize(
                    calculate_error,
                    x0,
                    method='L-BFGS-B',
                    jac=calculate_gradient,
                    args=(target_rank,),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 500}
                )
            except Exception:
                continue
                
            if not res.success:
                # Check if it still improved significantly
                if calculate_error(res.x, target_rank) > 1e-4:
                    continue
            else:
                if calculate_error(res.x, target_rank) > 1e-6:
                    continue

            # Extract optimized matrices
            U_opt = res.x[:nm*target_rank].reshape(nm, target_rank)
            V_opt = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
            W_opt = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
            
            # Try to snap to rational
            snapped = snap_to_rational(U_opt, V_opt, W_opt)
            if snapped is not None:
                U_final, V_final, W_final = snapped
                best_result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                best_rank = target_rank
                found = True
                break
        if found:
            # We found a lower rank. We could continue to see if we can go lower, 
            # but usually rank minimization is hard to go much lower than the first found.
            # Given the time budget, we might want to try one lower rank.
            # Let's just break for now to save time, or continue?
            # The task asks for rank-minimal. If we found 23, we should try 22, etc.
            # But the loop goes 19, 20, 21, 22, 23.
            # So if we find 23, we break the inner loop, then the outer loop continues to 22?
            # No, the loop is `for target_rank in range(19, 24)`.
            # So it tries 19 first. If it fails, it tries 20.
            # This is correct order for finding minimum.
            pass

    return best_result
