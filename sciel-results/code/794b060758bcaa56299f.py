import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 3, 3
    # Try ranks from 10 upward (10 is known to be minimal for 2x3x3)
    for rank in range(10, 19):
        result = _try_rank(n, m, p, rank)
        if result is not None:
            return result
    # Fallback to schoolbook
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

def _try_rank(n, m, p, rank):
    """Try to find a rank decomposition with half-integer entries."""
    # The tensor T has T[(i,j),(j',l),(l*n+i)] = delta_{j,j'}
    # We need sum_c U[i*m+j,c] * V[j'*m+l,c] * W[l*n+i,c] = delta_{j,j'}
    
    # Set up as a system: for each (i,j,j',l), the bilinear form must equal delta_{j,j'}
    # Unknowns: U (n*m x rank), V (m*p x rank), W (p*n x rank)
    
    # Use random initialization and iterative refinement
    # Try multiple random starts
    
    np.random.seed(42)
    
    for attempt in range(200):
        # Initialize with random values in {-1, 0, 1} or similar
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Random initialization
        for c in range(rank):
            U[:, c] = np.random.choice([-1, 0, 1], size=n * m)
            V[:, c] = np.random.choice([-1, 0, 1], size=m * p)
            W[:, c] = np.random.choice([-1, 0, 1], size=p * n)
        
        # Check if this works
        if _check_decomposition(n, m, p, U, V, W):
            # Snap to half-integers
            U_h = _snap_to_half_integer(U)
            V_h = _snap_to_half_integer(V)
            W_h = _snap_to_half_integer(W)
            if _check_decomposition(n, m, p, U_h, V_h, W_h):
                return _to_lists(U_h, V_h, W_h)
    
    # Try structured approach: use known patterns
    # For 2x3x3, try a specific construction
    if rank == 10:
        result = _try_structured_2x3x3()
        if result is not None:
            return result
    
    return None

def _check_decomposition(n, m, p, U, V, W):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for l in range(p):
                    # Sum over c: U[i*m+j, c] * V[jp*m+l, c] * W[l*n+i, c]
                    s = np.sum(U[i * m + j, :] * V[jp * m + l, :] * W[l * n + i, :])
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-6:
                        return False
    return True

def _snap_to_half_integer(M):
    """Snap each entry to the nearest half-integer."""
    return np.round(2 * M) / 2.0

def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

def _try_structured_2x3x3():
    """Try a known structured construction for 2x3x3 rank 10."""
    # This is tricky - let me try to set up the linear system properly
    # and solve it.
    
    # Actually, let me try a gradient-based approach with half-integer constraint
    # by solving in the doubled space (integers)
    
    import numpy as np
    
    n, m, p = 2, 3, 3
    rank = 10
    
    # Double the variables to work with integers
    # Let U_int = 2*U, V_int = 2*V, W_int = 2*W
    # Then the equation becomes: sum_c U_int[i*m+j,c] * V_int[jp*m+l,c] * W_int[l*n+i,c] = 4*delta_{j,jp}
    
    # This is still nonlinear. Let me try a different approach.
    
    # Use the fact that we can fix some variables and solve linearly for others.
    # For each column c, U[:,c], V[:,c], W[:,c] are vectors.
    # The constraint is: sum_c outer(U[:,c], V[:,c]) @ (something) ... 
    
    # Actually, let me just do a proper numerical solve.
    # Set up the system as a polynomial system and use Newton's method.
    
    # Variables: flatten U, V, W -> 6*10 + 9*10 + 6*10 = 210 variables
    # Equations: 54 bilinear equations
    
    # Use scipy's fsolve or similar
    
    from scipy.optimize import fsolve
    
    def equations(vars):
        """Return the 54 residual values."""
        U_flat = vars[:n*m*rank]
        V_flat = vars[n*m*rank:n*m*rank + m*p*rank]
        W_flat = vars[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for l in range(p):
                        s = np.sum(U[i*m+j, :] * V[jp*m+l, :] * W[l*n+i, :])
                        expected = 1.0 if j == jp else 0.0
                        residuals.append(s - expected)
        
        return np.array(residuals)
    
    # Try multiple random initializations
    np.random.seed(123)
    
    for attempt in range(50):
        x0 = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 0.5
        
        try:
            sol, info, ier, msg = fsolve(equations, x0, full_output=True, maxfev=10000)
            if ier == 1:
                # Check if solution is close to half-integers
                U_sol = sol[:n*m*rank].reshape(n*m, rank)
                V_sol = sol[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_sol = sol[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Snap to half-integers
                U_h = np.round(2 * U_sol) / 2.0
                V_h = np.round(2 * V_sol) / 2.0
                W_h = np.round(2 * W_sol) / 2.0
                
                if _check_decomposition(n, m, p, U_h, V_h, W_h):
                    return _to_lists(U_h, V_h, W_h)
        except Exception:
            continue
    
    return None
