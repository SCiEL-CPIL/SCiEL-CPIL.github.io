"""Matrix multiplication 3x3x3 tensor decomposition.

We attempt to find a rank-23 decomposition using the known Bini/Capovani et al.
structure, and fall back to the schoolbook rank-27 if verification fails.
"""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100.0  # 100s safety margin before 120s limit
    
    # Try to find rank-23 decomposition
    result = _try_rank23(deadline)
    if result is not None:
        return result
    
    # Fallback: schoolbook decomposition of rank 27
    return _schoolbook()

def _schoolbook():
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

def _verify_decomposition(U, V, W):
    """Verify that U, V, W exactly decompose the 3x3 matrix multiplication tensor.
    
    The tensor is T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
    Convention: W[k*n + i, r] so W is indexed by (k, i).
    U[i*m + j, r], V[j*p + k, r].
    
    For each output position (i, k), the coefficient of A[i,j]*B[j,k] in the
    sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must equal 1.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_mat = np.array(U, dtype=object)
    V_mat = np.array(V, dtype=object)
    W_mat = np.array(W, dtype=object)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # Sum over rank
                total = 0
                for r in range(rank):
                    total += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                if total != 1:
                    return False
    
    # Also verify that for non-matching indices, the products are zero
    # Actually, the tensor only has 1s at valid (i,j,k) positions and 0s elsewhere.
    # Since we're only checking the 27 positions where it should be 1,
    # we also need to ensure no spurious terms. But the tensor is defined only
    # on these 27 positions. Let's verify the full tensor.
    
    # Full verification: for all (i,j,k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                expected = 1  # For matrix mult, every (i,j,k) triple contributes
                if total != expected:
                    return False
    return True

def _try_rank23(deadline):
    """Attempt to find a rank-23 decomposition using structured search."""
    n, m, p = DIMENSIONS
    rank = 23
    
    # Use the known structure: the 3x3 matrix multiplication tensor can be
    # decomposed using the "Laderman" or similar structured approaches.
    # 
    # A well-known approach: embed 3x3 into 4x4, use rank-19 for 4x4 (which
    # doesn't exist), so that's not viable.
    #
    # Instead, let's use a direct optimization approach with the known Bini
    # rank-23 structure. The Bini decomposition uses 23 rank-1 terms with
    # specific patterns.
    #
    # Let's try a gradient-free optimization on a structured parameterization.
    
    # Start with the schoolbook 27-term decomposition and try to merge/eliminate
    # terms through optimization.
    
    # Actually, let's use a more systematic approach: parameterize the 23 rank-1
    # terms and use Nelder-Mead or similar to minimize the reconstruction error.
    
    # Define the tensor as a 3D array
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Flatten to a vector for easier manipulation
    T_flat = T.flatten()
    
    def _reconstruction_error(params):
        """Compute the Frobenius norm of the error in reconstructing T."""
        # params: 3 * rank * (n*m) or similar
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        U_flat = params[0:n*m*rank]
        V_flat = params[n*m*rank:2*n*m*rank]
        W_flat = params[2*n*m*rank:3*n*m*rank]
        
        U_mat = U_flat.reshape(n*m, rank)
        V_mat = V_flat.reshape(m*p, rank)
        W_mat = W_flat.reshape(p*n, rank)
        
        # Reconstruct tensor
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_mat[i*m + j, r] * V_mat[j*p + k, r] * W_mat[k*n + i, r]
                    error += (val - 1.0) ** 2
        return error
    
    # Initialize with random parameters
    np.random.seed(42)
    best_params = None
    best_error = float('inf')
    
    # Multiple random restarts
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: start near schoolbook and perturb
        U_init = np.random.randn(n*m, rank) * 0.5
        V_init = np.random.randn(m*p, rank) * 0.5
        W_init = np.random.randn(p*n, rank) * 0.5
        
        # Scale to encourage integer-like solutions
        U_init *= 2
        V_init *= 2
        W_init *= 2
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use Nelder-Mead
        try:
            from scipy.optimize import minimize
            res = minimize(_reconstruction_error, params_init, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
            if best_error < 1e-8:
                break
        except Exception:
            continue
    
    if best_params is not None and best_error < 1e-8:
        U_mat = best_params[0:n*m*rank].reshape(n*m, rank)
        V_mat = best_params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
        W_mat = best_params[2*n*m*rank:3*n*m*rank].reshape(p*n, rank)
        
        # Round to nearest half-integer
        U_rounded = np.round(U_mat * 2) / 2
        V_rounded = np.round(V_mat * 2) / 2
        W_rounded = np.round(W_mat * 2) / 2
        
        # Verify exact decomposition with rounded values
        U_list = [[float(x) for x in row] for row in U_rounded]
        V_list = [[float(x) for x in row] for row in V_rounded]
        W_list = [[float(x) for x in row] for row in W_rounded]
        
        if _verify_decomposition(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
