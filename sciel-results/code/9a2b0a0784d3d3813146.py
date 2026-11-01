"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition."""
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Start with schoolbook as guaranteed fallback
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try structured rank reduction using linear algebra
    # Strategy: For rank r, fix U and V with structured patterns,
    # then solve linear system for W.
    
    # The constraint: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # If we fix U and V, this is linear in W.
    
    # Let's try various ranks and structured initializations
    
    def try_rank_with_structure(r, seed_offset=0):
        """Try to find a rank-r decomposition using structured U,V and solving for W."""
        if time.time() > deadline:
            return None
        
        # Try multiple random structured patterns
        for attempt in range(20):
            if time.time() > deadline:
                return None
            
            rng = np.random.RandomState(42 + seed_offset + attempt)
            
            # Initialize U and V with small random values or structured patterns
            # Use sparse initialization to encourage low-rank solutions
            U = rng.randn(n * m, r) * 0.1
            V = rng.randn(m * p, r) * 0.1
            
            # Build linear system for W
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is: for each (i,j,k), the row of the system is:
            #   [U[i*m+j,c]*V[j*p+k,c] for c in range(r)] dotted with W[k*n+i,:] = 1
            
            # But W has shape (p*n, r), so W[k*n+i, c] is the unknown.
            # For fixed (i,j,k), the equation involves W[k*n+i, :] (a row of W)
            # weighted by U[i*m+j, c] * V[j*p+k, c] for each c.
            
            # Actually, let's think of it differently. Let W_flat be a vector of size p*n*r.
            # W_flat[(k*n+i)*r + c] = W[k*n+i, c]
            
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W_flat[(k*n+i)*r + c] = 1
            
            # This is a linear system: A @ W_flat = b
            # where A has n*m*p rows and p*n*r columns, b is all ones.
            
            # Number of equations: n*m*p = 40
            # Number of unknowns: p*n*r = 10*r
            
            # For r >= 4, we have more unknowns than equations (10r >= 40 when r >= 4)
            # So the system is underdetermined for r >= 4, meaning solutions exist.
            
            # But we need the solution to be "nice" (integer/half-integer) and the
            # decomposition to be valid.
            
            # Build the system
            num_eq = n * m * p
            num_unk = p * n * r
            
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            # W_flat index for W[k*n+i, c]
                            w_flat_idx = (k * n + i) * r + c
                            A[eq_idx, w_flat_idx] = U[i * m + j, c] * V[j * p + k, c]
                        eq_idx += 1
            
            # Solve least squares (system is underdetermined for r >= 4)
            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if the solution is close to satisfying the equations
            if residuals is not None and len(residuals) > 0:
                if residuals[0] > 1e-6:
                    continue
            
            # Reshape W_flat back to W
            W_sol = W_flat.reshape((p * n, r))
            
            # Verify the decomposition
            if _verify_decomposition(U, V, W_sol, n, m, p, tol=1e-8):
                # Try to round to nice values
                U_rounded = np.round(U * 2) / 2  # half-integers
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                # Check if rounded version still works
                if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-8):
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U_rounded]
                    V_list = [[float(x) for x in row] for row in V_rounded]
                    W_list = [[float(x) for x in row] for row in W_rounded]
                    return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    # Use unrounded but verify it's exact enough
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W_sol]
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # Try from 20 up to schoolbook
    for r in range(20, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_with_structure(r)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = r
                break
    
    return best
# EVOLVE-BLOCK-END
