# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import itertools

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    u_val = U[i*m + j][c]
                    v_val = V[j*p + k][c]
                    w_val = W[k*n + i][c]
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Generate the standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _try_reduce_rank(U_init, V_init, W_init, target_rank, n, m, p, max_iter=5000):
    """
    Try to reduce the rank of a decomposition by setting some columns to zero
    and adjusting the remaining columns to satisfy the tensor equations.
    This is a heuristic local search.
    """
    current_rank = len(U_init[0])
    if current_rank < target_rank:
        return None
    
    # We want to find a sub-decomposition of rank target_rank.
    # We select 'target_rank' indices from the current rank and try to solve for them.
    # To make it easier, we try random subsets of indices of size target_rank.
    
    U = [row[:] for row in U_init]
    V = [row[:] for row in V_init]
    W = [row[:] for row in W_init]
    
    # Select random indices to keep
    indices = np.random.choice(current_rank, target_rank, replace=False)
    
    # Zero out non-selected columns
    for c in range(current_rank):
        if c not in indices:
            for i in range(len(U)): U[i][c] = 0
            for i in range(len(V)): V[i][c] = 0
            for i in range(len(W)): W[i][c] = 0
            
    # Now we have a sparse matrix with zeros in non-selected columns.
    # We need to adjust the values in the selected columns to satisfy the equations.
    # This is a non-linear problem. We use a simple iterative refinement (Gauss-Seidel like)
    # or just random perturbations on the selected entries.
    
    # For simplicity and speed, we will try to find an exact integer solution by
    # randomizing the selected entries and checking.
    # This is still random search but constrained to a subspace.
    
    start_time = time.time()
    for _ in range(max_iter):
        if time.time() - start_time > 5:
            break
            
        # Perturb the selected entries
        for c in indices:
            if np.random.rand() < 0.5:
                # Small integer perturbation
                for i in range(len(U)):
                    if U[i][c] != 0:
                        U[i][c] += np.random.randint(-1, 2)
                for i in range(len(V)):
                    if V[i][c] != 0:
                        V[i][c] += np.random.randint(-1, 2)
                for i in range(len(W)):
                    if W[i][c] != 0:
                        W[i][c] += np.random.randint(-1, 2)
            
        if _check_decomposition(U, V, W, n, m, p):
            # Extract only the selected columns to form the final decomposition
            U_final = [[U[i][c] for c in indices] for i in range(len(U))]
            V_final = [[V[i][c] for c in indices] for i in range(len(V))]
            W_final = [[W[i][c] for c in indices] for i in range(len(W))]
            return U_final, V_final, W_final
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Get schoolbook decomposition
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    
    # Try to find a lower rank decomposition
    # Known lower bound is 19.
    for rank in range(19, 24):
        # Try multiple times to find a solution of this rank
        for attempt in range(20):
            result = _try_reduce_rank(U_sb, V_sb, W_sb, rank, n, m, p, max_iter=1000)
            if result is not None:
                U, V, W = result
                # Convert to float for output interface
                U_out = [[float(u) for u in row] for row in U]
                V_out = [[float(v) for v in row] for row in V]
                W_out = [[float(w) for w in row] for row in W]
                return {"U": U_out, "V": V_out, "W": W_out}
                
    # Fallback to schoolbook
    U_out = [[float(u) for u in row] for row in U_sb]
    V_out = [[float(v) for v in row] for row in V_sb]
    W_out = [[float(w) for w in row] for row in W_sb]
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
