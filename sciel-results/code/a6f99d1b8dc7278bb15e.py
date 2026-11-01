"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(len(U[0])):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def _solve_rank11_2x3x3():
    """Find a rank-11 decomposition for 2x3x3 matrix multiplication."""
    n2, m2, p2 = 2, 3, 3
    r = 11
    nm2 = n2 * m2  # 6
    mp2 = m2 * p2  # 9
    pn2 = p2 * n2  # 6
    
    # Try multiple random initializations
    best_solution = None
    best_err = float('inf')
    
    np.random.seed(123)
    
    for attempt in range(50):
        U = np.random.randn(nm2, r)
        V = np.random.randn(mp2, r)
        W = np.random.randn(pn2, r)
        
        def residual(params):
            U_flat = params[:nm2*r].reshape(nm2, r)
            V_flat = params[nm2*r:nm2*r+mp2*r].reshape(mp2, r)
            W_flat = params[nm2*r+mp2*r:].reshape(pn2, r)
            
            res = []
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        val = 0.0
                        for c in range(r):
                            val += U_flat[i*m2+j, c] * V_flat[j*p2+k, c] * W_flat[k*n2+i, c]
                        res.append(val - 1.0)
            return np.array(res)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            from scipy.optimize import least_squares
            result = least_squares(residual, x0, method='lm', max_nfev=5000)
            err = np.max(np.abs(result.fun))
            if err < best_err:
                best_err = err
                best_solution = result.x.copy()
                
                if err < 1e-10:
                    break
        except Exception:
            pass
    
    if best_solution is None:
        return None
    
    U_sol = best_solution[:nm2*r].reshape(nm2, r)
    V_sol = best_solution[nm2*r:nm2*r+mp2*r].reshape(mp2, r)
    W_sol = best_solution[nm2*r+mp2*r:].reshape(pn2, r)
    
    # Convert to list
    U_list = [[float(x) for x in row] for row in U_sol]
    V_list = [[float(x) for x in row] for row in V_sol]
    W_list = [[float(x) for x in row] for row in W_sol]
    
    return U_list, V_list, W_list

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12
    
    # Strategy: Use two rank-11 decompositions for 2x3x3 to build rank-22 for 2x3x6
    # Split p=6 into two groups of 3: k=0,1,2 and k=3,4,5
    
    # Get rank-11 decomposition for 2x3x3
    rank11_decomp = _solve_rank11_2x3x3()
    
    if rank11_decomp is not None:
        U1, V1, W1 = rank11_decomp
        r1 = 11
        
        # Build rank-22 decomposition for 2x3x6
        # U: 6x22 - same for both groups (U1 in cols 0-10, U1 in cols 11-21)
        # Actually, U should be the same structure. Let me think...
        # U[i*m+j, c] for c=0..10: U1[i*m2+j, c] where m2=3
        # But m=3 for both, so U[i*3+j, c] = U1[i*3+j, c] for c=0..10
        # For c=11..21: U[i*3+j, c-11] = U1[i*3+j, c-11]
        # So U is just [U1 | U1] horizontally concatenated
        
        U_combined = [U1[i] + U1[i] for i in range(nm)]  # 6x22
        
        # V: 18x22
        # V[j*p+k, c] where p=6
        # For k=0,1,2 (first group): V[j*6+k, c] = V1[j*3+k, c] for c=0..10
        # For k=3,4,5 (second group): V[j*6+k, c] = V1[j*3+(k-3), c-11] for c=11..21
        V_combined = [[0.0]*22 for _ in range(mp)]
        for j in range(m):
            for k in range(p):
                if k < 3:
                    for c in range(r1):
                        V_combined[j*p+k][c] = V1[j*3+k][c]
                else:
                    for c in range(r1):
                        V_combined[j*p+k][c+r1] = V1[j*3+(k-3)][c]
        
        # W: 12x22
        # W[k*n+i, c] where n=2
        # For k=0,1,2 (first group): W[k*2+i, c] = W1[k*2+i, c] for c=0..10
        # For k=3,4,5 (second group): W[k*2+i, c] = W1[(k-3)*2+i, c-11] for c=11..21
        W_combined = [[0.0]*22 for _ in range(pn)]
        for k in range(p):
            for i in range(n):
                if k < 3:
                    for c in range(r1):
                        W_combined[k*n+i][c] = W1[k*2+i][c]
                else:
                    for c in range(r1):
                        W_combined[k*n+i][c+r1] = W1[(k-3)*2+i][c]
        
        # Verify
        max_err = _verify_decomposition(n, m, p, U_combined, V_combined, W_combined)
        if max_err < 1e-9:
            return {"U": U_combined, "V": V_combined, "W": W_combined}
    
    # If rank-22 didn't work, try rank-24, 26, etc.
    # Fall back to schoolbook
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
# EVOLVE-BLOCK-END
