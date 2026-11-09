"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication.
    
    The tensor T_{i,j,k} = 1 for all valid i,j,k (matrix multiplication tensor).
    We seek U (shape n*m x r), V (shape m*p x r), W (shape p*n x r) such that
    for all i, j, k: T_{i,j,k} = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Strategy: Use the rectangular asymmetry (n=2, m=4, p=6) to search for a decomposition
    of rank < 48 via a randomized linear algebra approach with integer/half-integer constraints.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    
    def verify_decomposition(U, V, W, r):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    def clean_value(v):
        """Clean up a value to be integer or half-integer."""
        v_rounded = round(v * 2) / 2
        if abs(v - v_rounded) < 1e-9:
            if abs(v_rounded - round(v_rounded)) < 1e-9:
                return int(round(v_rounded))
            else:
                return v_rounded
        return v_rounded
    
    def try_rank_with_random_search(r, num_trials=100, seed=42):
        """
        Try to find a decomposition of rank r using random initialization.
        Uses the structure that for matrix multiplication, we can fix some variables
        and solve for others.
        """
        rng = np.random.RandomState(seed)
        
        for trial in range(num_trials):
            # Initialize with random small integers/half-integers
            # Use a mix of 0, 1, -1, 0.5, -0.5, 1.5, -1.5
            choices = [0, 1, -1, 0.5, -0.5, 1.5, -1.5, 2, -2]
            
            U = np.array([[rng.choice(choices) for _ in range(r)] for _ in range(nm)])
            V = np.array([[rng.choice(choices) for _ in range(r)] for _ in range(mp)])
            W = np.array([[rng.choice(choices) for _ in range(r)] for _ in range(pn)])
            
            # For each (i,j,k), we need:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            #
            # This is a system of nm*mp*pn/r equations... actually it's n*m*p = 48 equations
            # with 3*r unknowns (but they appear in a multiplicative way).
            #
            # Alternative: Fix U and V, then for each (i,j,k), we need:
            # W[k*n+i, c] appears in multiple equations. This is tricky.
            #
            # Let's try a different approach: exploit the structure.
            # For fixed i, the equations are:
            # sum_j,k sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each (i,j,k)
            
            # Actually, let's try a simpler approach: 
            # Use the fact that we can set up a linear system if we fix two of the three matrices.
            
            # Fix U and V randomly, then solve for W.
            # For each i (0 to n-1), we have p equations (one for each k) involving W[k*n+i, :]
            # But W[k*n+i, c] appears in equations for all j.
            
            # Let me think differently. For a given i, the equations are:
            # For each j in [0,m), k in [0,p):
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            #
            # Let's define for fixed i:
            # A_{j,k,c} = U[i*m+j, c] * V[j*p+k, c]
            # Then we need: sum_c A_{j,k,c} * W[k*n+i, c] = 1
            #
            # This is a system where W[k*n+i, c] are the unknowns, but they're shared across j.
            # For fixed i and k, we have m equations (one for each j):
            # sum_c A_{j,k,c} * W[k*n+i, c] = 1
            #
            # So for each (i,k), we have a system of m equations in r unknowns (W[k*n+i, 0..r-1]).
            # If m >= r, we might be able to solve this.
            
            # For our case: m=4, so we need r <= 4 for each (i,k) system to be solvable.
            # But we have n*p = 12 such systems, each with r unknowns.
            # Total unknowns: n*p*r = 12*r
            # Total equations: n*m*p = 48
            
            # So we need 12*r >= 48, i.e., r >= 4. That's a very low bound.
            # But the systems are coupled through U and V.
            
            # Let's try: fix U and V, then for each (i,k), solve the linear system for W[k*n+i, :].
            
            valid = True
            W_sol = np.zeros((pn, r))
            
            for i in range(n):
                for k in range(p):
                    # Build the system: for each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    # This is: M_j * W[k*n+i, :] = 1 where M_j[c] = U[i*m+j, c] * V[j*p+k, c]
                    M = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    # M has shape (m, r), we need M @ W[k*n+i, :] = ones(m)
                    
                    if r <= m:
                        # Underdetermined or square system, use least squares
                        try:
                            W_row, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W_sol[k*n+i, :] = W_row
                        except:
                            valid = False
                            break
                    else:
                        # Overdetermined, use least squares
                        try:
                            W_row, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W_sol[k*n+i, :] = W_row
                        except:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if W_sol entries are close to half-integers
            W_int = np.round(2 * W_sol)
            if not np.allclose(W_sol, W_int/2.0, atol=1e-6):
                continue
            
            # Convert to lists
            U_list = [[clean_value(U[i,c]) for c in range(r)] for i in range(nm)]
            V_list = [[clean_value(V[i,c]) for c in range(r)] for i in range(mp)]
            W_list = [[clean_value(W_sol[i,c]) for c in range(r)] for i in range(pn)]
            
            # Verify
            if verify_decomposition(U_list, V_list, W_list, r):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from lower bound upward
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = 24
    # But let's try from a smaller value
    
    for r in range(8, 48):
        result = try_rank_with_random_search(r, num_trials=50, seed=42 + r)
        if result is not None:
            return result
    
    # If we couldn't find a better decomposition, return schoolbook
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
