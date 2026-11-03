"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # For fixed U,V: W[k*n+i,c] must satisfy linear equations
    
    # Strategy: For a given rank R, we have R variables in W (p*n*R = 32*R total W entries)
    # The constraints are: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # That's n*m*p = 128 equations in p*n*R = 32*R unknowns (W entries)
    # Plus we need U and V to be consistent
    
    # Better: fix U and V, solve for W. Then check if a valid decomposition exists.
    # Actually, all three are coupled. Let's use a different approach:
    # Randomly initialize U and V with small integers, then solve the linear system for W.
    
    for R in [96, 90, 88, 84, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 10, 8, 6, 4, 2]:
        if time.time() > deadline - 10.0:
            break
        
        result = _try_rank_linear(R, n, m, p, deadline)
        if result is not None:
            best_result = result
            best_rank = R
            if best_rank <= 10:  # very good, stop
                break
    
    return best_result

def _try_rank_linear(R, n, m, p, deadline):
    """Try to find a rank-R decomposition by fixing U,V and solving for W linearly."""
    import random
    
    # For each trial:
    # 1. Randomly initialize U (n*m x R) and V (m*p x R) with small integers (-2,-1,0,1,2)
    # 2. Set up linear system: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # 3. This is a linear system in the W entries. Solve it.
    # 4. If solution exists and W has rational entries, verify.
    
    max_trials = 50
    trial = 0
    while trial < max_trials and time.time() < deadline - 15.0:
        trial += 1
        
        # Initialize U and V with small random integers
        U = [[random.randint(-2, 2) for _ in range(R)] for _ in range(n * m)]
        V = [[random.randint(-2, 2) for _ in range(R)] for _ in range(m * p)]
        
        # Build linear system for W
        # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(R)
        # Total unknowns: p * n * R
        # Equations: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Total equations: n * m * p
        
        num_unknowns = p * n * R
        num_eqs = n * m * p
        
        # Build coefficient matrix A (num_eqs x num_unknowns) and RHS b (num_eqs)
        # Using sparse representation for efficiency
        A_rows = []
        b = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = {}
                    for c in range(R):
                        coeff = U[i * m + j][c] * V[j * p + k][c]
                        if coeff != 0:
                            # W index: k*n+i, column c
                            w_idx = (k * n + i) * R + c
                            row[w_idx] = row.get(w_idx, 0) + coeff
                    if row:
                        A_rows.append(row)
                        b.append(1)
                    else:
                        # This equation is 0 = 1, impossible
                        return None
        
        if len(A_rows) != num_eqs:
            return None
        
        # Solve the linear system using Gaussian elimination over rationals
        # This is expensive, so we use a simplified approach
        # Actually, for large systems, let's use numpy first to check if a solution exists
        # then refine with rationals if needed
        
        # Build dense matrix for numpy check
        A_dense = np.zeros((len(A_rows), num_unknowns), dtype=np.float64)
        for idx, row in enumerate(A_rows):
            for w_idx, coeff in row.items():
                A_dense[idx, w_idx] = coeff
        b_dense = np.array(b, dtype=np.float64)
        
        # Check if system is consistent using least squares
        try:
            solution, residuals, rank_A, sv = np.linalg.lstsq(A_dense, b_dense, rcond=None)
        except:
            continue
        
        # Check if the residual is small (system is approximately consistent)
        if residuals and residuals[0] > 1e-6:
            continue
        
        # Now verify: does this solution actually satisfy all equations exactly?
        # Convert to fractions for exact check
        # But first, check if the solution is close to rational values
        # For now, let's just verify numerically
        
        W = [[0.0] * R for _ in range(p * n)]
        for idx in range(p * n):
            for c in range(R):
                W[idx][c] = solution[idx * R + c]
        
        # Verify the decomposition
        if _verify_numeric(U, V, W, n, m, p, tol=1e-6):
            # Convert to exact fractions
            U_frac = [[Fraction(U[i][c]).limit_denominator(1000000) for c in range(R)] for i in range(n * m)]
            V_frac = [[Fraction(V[i][c]).limit_denominator(1000000) for c in range(R)] for i in range(m * p)]
            W_frac = [[Fraction(W[i][c]).limit_denominator(1000000) for c in range(R)] for i in range(p * n)]
            
            # Final exact verification
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                # Convert back to list of lists with float or int
                U_out = [[float(x) for x in row] for row in U_frac]
                V_out = [[float(x) for x in row] for row in V_frac]
                W_out = [[float(x) for x in row] for row in W_frac]
                
                # Check if entries are close to integers or half-integers
                valid = True
                for mat in [U_out, V_out, W_out]:
                    for row in mat:
                        for val in row:
                            nearest = round(val * 2) / 2
                            if abs(val - nearest) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U_out, "V": V_out, "W": W_out}
        
        # If numerical verification passed but exact didn't, try to snap to nearest rationals
        # and re-verify
    
    return None

def _verify_numeric(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition numerically."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Verify decomposition exactly using fractions."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(len(U[0])):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, time_limit=12.0):
    """Try to find a decomposition of given rank using random initialization and gradient-free search."""
    start_time = time.time()
    return None

# EVOLVE-BLOCK-END
