"""Combined program: schoolbook baseline with structured low-rank search for 2x4x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    
    The tensor T[i, j, k] for multiplying (n x m) by (m x p) to get (n x p)
    should satisfy: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where T[i, j, k] = 1 if the indices are valid (which they always are for 0<=i<n, 0<=j<m, 0<=k<p).
    
    Actually, the tensor for matrix multiplication M[i,j] * N[j,k] = P[i,k]
    has the structure: T[i, j, k] = delta (it's 1 for all valid i,j,k in the tensor representation).
    
    Wait, let me reconsider. The matrix multiplication tensor is defined such that:
    P[i, k] = sum_j M[i, j] * N[j, k]
    
    In tensor decomposition form:
    P[i, k] = sum_c U[i, c] * (sum_j ... ) 
    
    Actually, the standard formulation is:
    The tensor T has indices (i, j, k) where i is row of first matrix, j is column of first/row of second, k is column of second.
    T[i, j, k] = 1 for all valid indices (it's the "identity" tensor in a sense, but really it encodes the multiplication).
    
    More precisely: if we have matrices A (n x m) and B (m x p), then C = A * B where C[i,k] = sum_j A[i,j] * B[j,k].
    
    The tensor decomposition approach: C[i,k] = sum_c U[i, c] * V[c] * W[k, c]... no.
    
    Let me use the standard formulation from the task:
    - U has shape (n*m, rank)
    - V has shape (m*p, rank)  
    - W has shape (p*n, rank)
    
    The decomposition should satisfy:
    For all i in [0,n), j in [0,m), k in [0,p):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is because the matrix multiplication tensor has all entries equal to 1 (it's the "multiplication" operation).
    """
    n, m, p = DIMENSIONS
    # Check all tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(r, deadline):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    n, m, p = DIMENSIONS
    rows_U = n * m  # 8
    rows_V = m * p  # 24
    rows_W = p * n  # 12
    
    # Initialize with random values
    np.random.seed(42 + r)
    U = np.random.randn(rows_U, r)
    V = np.random.randn(rows_V, r)
    W = np.random.randn(rows_W, r)
    
    # We need to solve: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is a system of n*m*p = 48 equations in r*3 variables (but with nonlinear structure).
    
    # Use a simple iterative approach: fix two matrices, solve for the third
    # This is like alternating minimization
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(200):
        if time.time() > deadline:
            break
            
        # Fix V and W, solve for U
        # For each row of U (indexed by i*m+j), we need:
        # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # This is a linear system in U[i*m+j,:] for each (i,j)
        
        for i in range(n):
            for j in range(m):
                # Build linear system: A * U_row = b
                # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # b[k] = 1
                A = np.zeros((p, r))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                
                # Solve least squares
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Fix U and W, solve for V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Fix U and V, solve for W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Compute error
        error = 0.0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    val = np.sum(U[ii*m+jj, :] * V[jj*p+kk, :] * W[kk*n+ii, :])
                    error += (val - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_error < 1e-6:
            # Check if it's actually valid
            if _verify(best_U, best_V, best_W):
                return best_U, best_V, best_W
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook()
    best_rank = n * m * p  # 48
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: for matrix multiplication, rank >= max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # But actually the true minimum rank is known to be at least something.
    # Let's try ranks from 24 upward.
    
    for r in range(24, best_rank):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank(r, deadline)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {"U": U, "V": V, "W": W}
            break  # Found a valid decomposition, stop
    
    # Convert to lists for the required interface
    best_result["U"] = [[float(x) for x in row] for row in best_result["U"]]
    best_result["V"] = [[float(x) for x in row] for row in best_result["V"]]
    best_result["W"] = [[float(x) for x in row] for row in best_result["W"]]
    
    return best_result
# EVOLVE-BLOCK-END
