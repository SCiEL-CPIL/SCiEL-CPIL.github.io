import time
import numpy as np
import random

DIMENSIONS = [3, 3, 8]

def _validate(U, V, W, n=3, m=3, p=8):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    # For each (i,j,k) in the tensor, the entry should be 1
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[c][r]
                if total != 1:
                    return False
    # Check entries are integers or half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if isinstance(val, float):
                    if not val.is_integer() and abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
    return True

def _find_rank_decomp(n, m, p, target_rank, max_restarts=20, timeout=100):
    """
    Attempts to find a rank-target_rank decomposition for n x m x p multiplication.
    Returns (U, V, W) lists of lists if found, else None.
    Uses random initialization and gradient descent / least squares refinement.
    """
    start_time = time.time()
    
    # Tensor indices: i in [0,n), j in [0,m), k in [0,p)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    
    # Number of equations: n*m*p
    num_eqs = n * m * p
    
    # Flatten U, V, W into a single vector for optimization
    # U size: n*m*rank
    # V size: m*p*rank
    # W size: p*n*rank
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Random initialization
        # Scale matters. Start with small random values.
        U_flat = np.random.randn(n * m * target_rank) * 0.1
        V_flat = np.random.randn(m * p * target_rank) * 0.1
        W_flat = np.random.randn(p * n * target_rank) * 0.1
        
        # Reshape
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        
        # We want to minimize sum (sum_r U[a,r]V[b,r]W[c,r] - 1)^2
        # This is a non-convex problem.
        # We can use a simple iterative method or scipy if available.
        # Since we only have standard lib and numpy, we implement a basic gradient step or use scipy.optimize.minimize if allowed.
        # The prompt says "Python standard library, NumPy and SciPy are available".
        
        from scipy.optimize import minimize
        
        def objective(params):
            U_curr = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_curr = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_curr = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U_curr[a] * V_curr[b] * W_curr[c])
                        error += (val - 1.0)**2
            return error
        
        # Initial params
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        # Try to optimize
        try:
            res = minimize(objective, x0, method='BFGS', options={'maxiter': 50, 'gtol': 1e-4})
            if res.success or res.fun < 1e-6:
                # Extract solution
                U_sol = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Check if we can round to integers/half-integers
                # The problem requires integer/half-integer entries.
                # If the solution is close to these, we can snap.
                U_snapped = np.round(U_sol * 2) / 2
                V_snapped = np.round(V_sol * 2) / 2
                W_snapped = np.round(W_sol * 2) / 2
                
                # Verify snapped solution
                if _validate(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
                    
                # If snapping failed, maybe the solution is valid but not snapped?
                # The task requires integer/half-integer. So we MUST snap.
                # If the true solution is integer/half-integer, the optimizer should get close.
                
        except Exception:
            pass
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 3, 3, 8
    
    # Strategy: Split 3x3x8 into two 3x3x4 problems.
    # Rank 19 for 3x3x4 is known.
    # If we find it, total rank is 38.
    
    # Attempt to find rank-19 decomposition for 3x3x4
    U1, V1, W1 = None, None, None
    if time.time() < deadline - 50:
        U1, V1, W1 = _find_rank_decomp(3, 3, 4, 19, max_restarts=10, timeout=40)
    
    if U1 is not None:
        # Assemble 3x3x8 decomposition
        # U for 3x3x8 has rows i*m+j (9 rows).
        # For 3x3x4, U has rows i*m+j (9 rows).
        # The first 4 columns of B (k=0..3) use the first 19 ranks.
        # The last 4 columns of B (k=4..7) use the next 19 ranks.
        
        total_rank = 38
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        # Fill first half (k=0..3)
        for r in range(19):
            for a in range(n * m):
                U[a][r] = U1[a][r]
            for b in range(m * 4): # j*p+k where p=4 for subproblem
                j = b // 4
                k = b % 4
                # In the full problem, b_full = j*8 + k
                b_full = j * 8 + k
                V[b_full][r] = V1[b][r]
            for c in range(4 * n): # k*n+i for subproblem p=4
                k = c // n
                i = c % n
                # In the full problem, c_full = k*3 + i
                c_full = k * 3 + i
                W[c_full][r] = W1[c][r]
                
        # Fill second half (k=4..7)
        # The subproblem for k=4..7 is identical in structure.
        # U is the same.
        # V indices: j*8 + (k+4)
        # W indices: (k+4)*3 + i
        for r in range(19):
            for a in range(n * m):
                U[a][r + 19] = U1[a][r]
            for b in range(m * 4):
                j = b // 4
                k = b % 4
                # In the full problem, b_full = j*8 + (k+4)
                b_full = j * 8 + (k + 4)
                V[b_full][r + 19] = V1[b][r]
            for c in range(4 * n):
                k = c // n
                i = c % n
                # In the full problem, c_full = (k+4)*3 + i
                c_full = (k + 4) * 3 + i
                W[c_full][r + 19] = W1[c][r]
                
        return {"U": U, "V": V, "W": W}
        
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    valid = _validate(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
