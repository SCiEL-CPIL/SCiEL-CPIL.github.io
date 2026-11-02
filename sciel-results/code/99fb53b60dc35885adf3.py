<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    The schoolbook rank is 2*3*10 = 60.
    We attempt to find a lower rank decomposition.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p,
    the tensor T_{i,j,k} corresponds to C_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    A rank-r decomposition has:
    C_{i,k} = sum_{a=1}^r U_{i,a} * V_{j,a} * W_{k,a}  -- wait, this isn't quite right for matrix mult.
    
    Actually, the standard tensor decomposition for matrix multiplication is:
    T_{i,j,k} = sum_{a=1}^r U_{i,a} * V_{j,a} * W_{k,a}
    where the tensor T is the multiplication tensor. But for matrix multiplication,
    the structure is slightly different.
    
    Let me reconsider. The multiplication tensor for A(n x m) * B(m x p) -> C(n x p) is:
    T_{i,j,k} = 1 if the product A_{i,j} * B_{j,k} contributes to C_{i,k}, which it always does.
    So T_{i,j,k} = 1 for all valid i,j,k? No, that's not right either.
    
    The multiplication tensor has entries T_{i,j,k} such that:
    (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    The rank of this tensor is the minimum r such that we can write:
    (A*B)_{i,k} = sum_{a=1}^r U_{i,a} * (sum_j V_{j,a} * W_{j,a}) ... no.
    
    Let me use the standard formulation. A rank-r decomposition of the matrix multiplication tensor
    means we can write the multiplication as:
    C_{i,k} = sum_{a=1}^r U_{i,a} * M_a(B)
    where M_a are linear forms on B, or equivalently:
    
    C_{i,k} = sum_{a=1}^r sum_j U_{i,a} * V_{j,a} * W_{j,k,a}
    
    Hmm, let me look at the seed program's convention:
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    The convention in the seed is:
    U[i*m + j, c] = 1, V[j*p + k, c] = 1, W[k*n + i, c] = 1
    
    And the reconstruction is:
    C_{i,k} = sum_{j} sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, that gives C_{i,k} = sum_c sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    For the schoolbook, each (i,j,k) gets its own column c, so:
    C_{i,k} = sum_c sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
           = sum_j A_{i,j} * B_{j,k}
    
    because for column c corresponding to (i',j',k'), U[i*m+j,c] = delta_{i,i'}delta_{j,j'},
    V[j*p+k,c] = delta_{j,j'}delta_{k,k'}, W[k*n+i,c] = delta_{k,k'}delta_{i,i'}.
    
    So the sum over c picks out only the term where i=i', j=j', k=k', giving A_{i,j} * B_{j,k}.
    
    So the general rank-r decomposition satisfies:
    C_{i,k} = sum_{j=0}^{m-1} sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    This must equal sum_j A_{i,j} * B_{j,k} for all A, B.
    
    This is a multilinear identity. The unknowns are U, V, W entries.
    We need: for all i, k:
    sum_j A_{i,j} * B_{j,k} = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    This is a polynomial identity in the variables A_{i,j} and B_{j,k}.
    By comparing coefficients of each monomial A_{i,j} * B_{j,k}, we get:
    For each (i, j, k):
    1 = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    And for all other monomials (which don't appear in the LHS), the coefficient must be 0.
    But since the LHS only has monomials of the form A_{i,j} * B_{j,k} (one A and one B factor),
    and the RHS is also bilinear in A and B (linear in each A_{i,j} when B is fixed, etc.),
    we just need the above equation to hold for all (i,j,k).
    
    Wait, actually the RHS is:
    sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    This is already summed over j. Let me rewrite:
    Let U_{i,j,c} = U[i*m+j, c], V_{j,k,c} = V[j*p+k, c], W_{k,i,c} = W[k*n+i, c].
    
    Then the condition is:
    sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c} = 1  for all i, j, k.
    
    So we need to find matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
    for all i in [0,n), j in [0,m), k in [0,p):
    sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is a system of n*m*p = 60 equations in 3*r*(n*m + m*p + p*n)/... wait, the number of unknowns is
    r*(n*m + m*p + p*n) = r*(6 + 30 + 20) = r*56.
    
    For r = 60, we have 3360 unknowns and 60 equations, so many solutions exist (schoolbook is one).
    For lower r, we need to check if a solution exists.
    
    This is a system of quadratic equations (each equation is linear in each of U, V, W separately,
    but bilinear/trilinear overall). Actually, each equation is linear in the entries of U for fixed V,W,
    etc. This is a 3-linear system.
    
    Known results: For 2x3 * 3x10, the rank is not known to be less than 60 in general, but let's try
    to find a lower rank. Actually, the minimal rank for matrix multiplication is a hard open problem
    for most sizes. For 2x3 * 3x10, the best known upper bound might be 60 or slightly less.
    
    Let me try to see if rank 59 or lower is possible. Given the complexity, let me try a heuristic search
    or use the fact that we can set some entries to 0 or 1 to simplify.
    
    Actually, let me just return the schoolbook decomposition for now, as finding a lower rank is
    computationally intensive and may not be feasible within the time limit.
    """
    n, m, p = DIMENSIONS
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
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import random
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    We try to find a decomposition with rank less than 60 using optimization.
    The constraint is: for all i,j,k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    We use a heuristic approach: start with random initializations and try to minimize
    the sum of squared errors while projecting onto integer/half-integer values.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try ranks from 59 down to some minimum
    for rank in range(schoolbook_rank - 1, 30, -1):
        result = _try_find_decomposition(n, m, p, rank)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_find_decomposition(n, m, p, rank, max_time=60):
    """Try to find a rank-r decomposition using optimization."""
    start_time = time.time()
    
    # Initialize with random values
    random.seed(42)
    np.random.seed(42)
    
    # Try multiple initializations
    for attempt in range(5):
        if time.time() - start_time > max_time:
            break
            
        # Random initialization with small values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Define the objective: minimize sum of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        def objective(params):
            if time.time() - start_time > max_time:
                return 1e10
            U_flat = params[:n*m*rank].reshape(n*m, rank)
            V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                        error += (s - 1.0) ** 2
            return error
        
        # Initial parameters
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use Nelder-Mead for derivative-free optimization
        result = minimize(objective, params0, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < 1e-6:
            # Found a solution, now round to nearest half-integer
            U_sol = result.x[:n*m*rank].reshape(n*m, rank)
            V_sol = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_sol = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Round to nearest half-integer
            U_rounded = np.round(U_sol * 2) / 2
            V_rounded = np.round(V_sol * 2) / 2
            W_rounded = np.round(W_sol * 2) / 2
            
            # Verify the rounded solution
            if _verify_decomposition(n, m, p, rank, U_rounded, V_rounded, W_rounded):
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U_rounded]
                V_list = [[float(x) for x in row] for row in V_rounded]
                W_list = [[float(x) for x in row] for row in W_rounded]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _verify_decomposition(n, m, p, rank, U, V, W):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
