"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 3]
DEADLINE = 120.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin

def _verify(U, V, W, n=2, m=2, p=3):
    """Verify that the tensor decomposition is exact."""
    r = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check that all entries are integers or half-integers
    for mat, name in [(U_arr, 'U'), (V_arr, 'V'), (W_arr, 'W')]:
        scaled = 2 * mat
        if not np.allclose(scaled, np.round(scaled), atol=1e-9):
            return False
    
    # Verify tensor contraction: for each (i,j,k), sum_r U[i,r]*V[j,r]*W[k,r] == T[i,j,k]
    # T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term, else 0
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if this is a valid term
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0  # All terms are valid in matrix multiplication
                actual = 0.0
                for r in range(r):
                    actual += U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
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
    
    The tensor T for 2x2 * 2x3 matrix multiplication has 2*2*3 = 12 nonzero entries.
    We seek U (4x10), V (6x10), W (6x10) such that:
    For each (i,j,k) in {0,1}x{0,1}x{0,1,2}: sum_r U[i,r]*V[j,r]*W[k,r] = 1
    
    This gives 12 polynomial equations in 10*4 + 10*6 + 10*6 = 160 variables.
    We use a randomized approach: fix some variables and solve for others.
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
    # For each pair (r1, r2), see if we can combine them into one column
    for r1 in range(10):
        if _remaining() < 5:
            return None
        for r2 in range(r1 + 1, 11):
            if _remaining() < 5:
                return None
            
            # Try to find alpha, beta such that:
            # alpha * U11[:,r1] + beta * U11[:,r2] works as a single column
            # This is a simplification; in general we need nonlinear combinations
            
            # Build a rank-10 candidate by removing one column and adjusting
            # Remove column r2 and try to absorb its contribution
            U10 = np.delete(U11, r2, axis=1)
            V10 = np.delete(V11, r2, axis=1)
            W10 = np.delete(W11, r2, axis=1)
            
            # Now we need to adjust the remaining columns to compensate
            # This is a hard problem; use numerical optimization
            if _remaining() < 3:
                continue
            
            # Try a simple perturbation: add a scaled version of the removed column
            # to one of the remaining columns
            for target in range(10):
                if target == r1:
                    continue
                for scale in [-2, -1, -0.5, 0.5, 1, 2]:
                    if _remaining() < 1:
                        break
                    
                    U_try = U10.copy()
                    V_try = V10.copy()
                    W_try = W10.copy()
                    
                    # Add scaled removed column to target
                    U_try[:, target] += scale * U11[:, r2]
                    V_try[:, target] += scale * V11[:, r2]
                    W_try[:, target] += scale * W11[:, r2]
                    
                    # Check if this gives a valid decomposition
                    # (This won't generally work, but it's a quick test)
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for r in range(rank):
                                    s += U_try[i, r] * V_try[j, r] * W_try[k, r]
                                if abs(s - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        # Convert to list and check half-integer constraint
                        U_list = [[float(x) for x in row] for row in U_try]
                        V_list = [[float(x) for x in row] for row in V_try]
                        W_list = [[float(x) for x in row] for row in W_try]
                        
                        # Check half-integer
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
    # Use the fact that we can fix some degrees of freedom
    if _remaining() < 10:
        return None
    
    # Try random initialization with half-integer values
    # and use Newton's method on the polynomial system
    np.random.seed(42)
    
    for attempt in range(50):
        if _remaining() < 5:
            return None
        
        # Initialize with small half-integer values
        U_init = np.zeros((num_u, rank))
        V_init = np.zeros((num_v, rank))
        W_init = np.zeros((num_w, rank))
        
        # Set some entries to 1 or -1 to break symmetry
        for r in range(rank):
            # Randomly assign some nonzero entries
            for i in range(num_u):
                if np.random.random() < 0.3:
                    U_init[i, r] = np.random.choice([-1, 1])
            for j in range(num_v):
                if np.random.random() < 0.3:
                    V_init[j, r] = np.random.choice([-1, 1])
            for k in range(num_w):
                if np.random.random() < 0.3:
                    W_init[k, r] = np.random.choice([-1, 1])
        
        # Use scipy's fsolve if available
        try:
            from scipy.optimize import fsolve
        except ImportError:
            break
        
        # Define the system of equations
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
                            s += U_flat[i, r] * V_flat[j, r] * W_flat[k, r]
                        eqs.append(s - 1.0)
            return eqs
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            solution, info, ier, msg = fsolve(equations, params_init, full_output=True, maxfev=1000)
            if ier == 1:
                U_sol = solution[:num_u*rank].reshape(num_u, rank)
                V_sol = solution[num_u*rank:num_u*rank+num_v*rank].reshape(num_v, rank)
                W_sol = solution[num_u*rank+num_v*rank:num_u*rank+num_v*rank+num_w*rank].reshape(num_w, rank)
                
                # Check half-integer constraint
                ok = True
                for mat in [U_sol, V_sol, W_sol]:
                    scaled = 2 * mat
                    if not np.allclose(scaled, np.round(scaled), atol=1e-6):
                        ok = False
                        break
                
                if ok:
                    # Round to nearest half-integer
                    U_round = np.round(2 * U_sol) / 2
                    V_round = np.round(2 * V_sol) / 2
                    W_round = np.round(2 * W_sol) / 2
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for r in range(rank):
                                    s += U_round[i, r] * V_round[j, r] * W_round[k, r]
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
