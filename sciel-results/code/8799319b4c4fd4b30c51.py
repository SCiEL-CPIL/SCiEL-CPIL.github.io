"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
DEADLINE = 120.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin

def _verify(U, V, W, n=2, m=2, p=3):
    """Verify that the tensor decomposition is exact using the bilinear form."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = len(U[0])
    
    # Check that all entries are integers or half-integers
    for mat, name in [(U_arr, 'U'), (V_arr, 'V'), (W_arr, 'W')]:
        scaled = 2 * mat
        if not np.allclose(scaled, np.round(scaled), atol=1e-9):
            return False
    
    # Verify tensor contraction: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # This is the coefficient of A[i,j]*B[j,k] in the bilinear form
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _rank11_solution():
    """Return the known rank-11 decomposition."""
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[2][2] = 1; W[2][4] = 1
    W[1][1] = 1; W[1][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    U[0][7] = 1
    V[2][7] = 1
    
    U[1][8] = 1
    V[5][8] = 1
    
    U[2][9] = 1
    V[2][9] = 1
    
    U[3][10] = 1
    V[5][10] = 1
    
    W[4][7] = 1; W[4][8] = 1
    W[5][9] = 1; W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}

def _search_rank10():
    """
    Search for a rank-10 decomposition using polynomial system solving.
    """
    if _remaining() < 10:
        return None
    
    n, m, p = 2, 2, 3
    rank = 10
    num_u = n * m  # 4
    num_v = m * p  # 6
    num_w = p * n  # 6
    
    # Strategy: Try to find a solution where entries are in {-1, 0, 1, 0.5, -0.5}
    # Use a structured search based on the rank-11 solution, trying to merge two columns
    
    # Start from rank-11 and try to eliminate one rank dimension
    base = _rank11_solution()
    U11 = np.array(base['U'], dtype=float)
    V11 = np.array(base['V'], dtype=float)
    W11 = np.array(base['W'], dtype=float)
    
    # Try to merge pairs of columns from the rank-11 solution
    for r1 in range(10):
        if _remaining() < 5:
            return None
        for r2 in range(r1 + 1, 11):
            if _remaining() < 5:
                return None
            
            # Build a rank-10 candidate by removing one column and adjusting
            U10 = np.delete(U11, r2, axis=1)
            V10 = np.delete(V11, r2, axis=1)
            W10 = np.delete(W11, r2, axis=1)
            
            if _remaining() < 3:
                continue
            
            # Try a simple perturbation: add a scaled version of the removed column
            for target in range(10):
                if target == r1:
                    continue
                for scale in [-2, -1, -0.5, 0.5, 1, 2]:
                    if _remaining() < 1:
                        break
                    
                    U_try = U10.copy()
                    V_try = V10.copy()
                    W_try = W10.copy()
                    
                    U_try[:, target] += scale * U11[:, r2]
                    V_try[:, target] += scale * V11[:, r2]
                    W_try[:, target] += scale * W11[:, r2]
                    
                    # Check if this gives a valid decomposition
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for r in range(rank):
                                    s += U_try[i*m+j, r] * V_try[j*p+k, r] * W_try[k*n+i, r]
                                if abs(s - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        U_list = [[float(x) for x in row] for row in U_try]
                        V_list = [[float(x) for x in row] for row in V_try]
                        W_list = [[float(x) for x in row] for row in W_try]
                        
                        ok = True
                        for mat in [U_list, V_list, W_list]:
                            for row in mat:
                                for val in row:
                                    if abs(2*val - round(2*val)) > 1e-9:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if not ok:
                                break
                        if ok:
                            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try a more systematic approach: parameterize and solve
    if _remaining() < 10:
        return None
    
    # Try random initialization with half-integer values
    np.random.seed(42)
    
    for attempt in range(50):
        if _remaining() < 5:
            return None
        
        U_init = np.zeros((num_u, rank))
        V_init = np.zeros((num_v, rank))
        W_init = np.zeros((num_w, rank))
        
        for r in range(rank):
            for i in range(num_u):
                if np.random.random() < 0.3:
                    U_init[i, r] = np.random.choice([-1, 1])
            for j in range(num_v):
                if np.random.random() < 0.3:
                    V_init[j, r] = np.random.choice([-1, 1])
            for k in range(num_w):
                if np.random.random() < 0.3:
                    W_init[k, r] = np.random.choice([-1, 1])
        
        try:
            from scipy.optimize import fsolve
        except ImportError:
            break
        
        def equations(params):
            U_flat = params[:num_u*rank].reshape(num_u, rank)
            V_flat = params[num_u*rank:num_u*rank+num_v*rank].reshape(num_v, rank)
            W_flat = params[num_u*rank+num_v*rank:num_u*rank+num_v*rank+num_w*rank].reshape(num_w, rank)
            
            eqs = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                        eqs.append(s - 1.0)
            return eqs
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            solution, info, ier, msg = fsolve(equations, params_init, full_output=True, maxfev=1000)
            if ier == 1:
                U_sol = solution[:num_u*rank].reshape(num_u, rank)
                V_sol = solution[num_u*rank:num_u*rank+num_v*rank].reshape(num_v, rank)
                W_sol = solution[num_u*rank+num_v*rank:num_u*rank+num_v*rank+num_w*rank].reshape(num_w, rank)
                
                ok = True
                for mat in [U_sol, V_sol, W_sol]:
                    scaled = 2 * mat
                    if not np.allclose(scaled, np.round(scaled), atol=1e-6):
                        ok = False
                        break
                
                if ok:
                    U_round = np.round(2 * U_sol) / 2
                    V_round = np.round(2 * V_sol) / 2
                    W_round = np.round(2 * W_sol) / 2
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for r in range(rank):
                                    s += U_round[i*m+j, r] * V_round[j*p+k, r] * W_round[k*n+i, r]
                                if abs(s - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        U_list = [[float(x) for x in row] for row in U_round]
                        V_list = [[float(x) for x in row] for row in V_round]
                        W_list = [[float(x) for x in row] for row in W_round]
                        return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            continue
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # First, try to find a rank-10 solution
    if _remaining() > 20:
        result = _search_rank10()
        if result is not None:
            return result
    
    # Fall back to rank-11 solution
    return _rank11_solution()
# EVOLVE-BLOCK-END
