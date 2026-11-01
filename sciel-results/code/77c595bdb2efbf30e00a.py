<<<<<<< SEARCH
"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
_DEADLINE = 130.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def _is_half_integer(val, tol=1e-5):
    nearest = round(val * 2) / 2
    return abs(val - nearest) < tol

def _all_half_integer(mat, tol=1e-5):
    return all(_is_half_integer(val, tol) for val in mat)
=======
"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
_DEADLINE = 130.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def _is_half_integer(val, tol=1e-5):
    nearest = round(val * 2) / 2
    return abs(val - nearest) < tol

def _all_half_integer(mat, tol=1e-5):
    return all(_is_half_integer(val, tol) for val in mat)

def _strassen_2x3x3_init():
    """
    Initialize using a Strassen-like decomposition for 2x3 * 3x3.
    We can think of this as:
    A (2x3) * B (3x3) = C (2x3)
    
    Split B into columns: B = [B1, B2, B3] where Bi is 3x1
    C = A * B = [A*B1, A*B2, A*B3]
    
    For each column, we compute A (2x3) * Bi (3x1) -> Ci (2x1)
    This is like 2x3 * 3x1 multiplication.
    
    Actually, let's think differently. We can use the fact that:
    (2x3) * (3x3) can be done by viewing it as 2 independent (2x3)*(3x1) problems,
    but that gives rank 2*9=18 (schoolbook).
    
    Better: Use a Strassen-like approach for the 2x3 * 3x1 subproblem.
    For 2x3 * 3x1:
    A = [[a1, a2, a3], [b1, b2, b3]]
    v = [x, y, z]^T
    result = [a1*x + a2*y + a3*z, b1*x + b2*y + b3*z]
    
    We can use 5 multiplications:
    m1 = (a1+a2)*x
    m2 = (a2+a3)*y  
    m3 = (a3-a2)*z
    m4 = a1*(x+y)
    m5 = (b1-b2)*(x-y) -- wait, this doesn't work directly
    
    Let me try a different approach. For 2x3 * 3x1:
    r1 = a1*x + a2*y + a3*z
    r2 = b1*x + b2*y + b3*z
    
    We need 6 terms total. Can we do it in 5?
    
    Actually, the rank of the 2x3x1 tensor is 5 (known result).
    So for 2x3 * 3x3, we have 3 independent 2x3*3x1 problems, giving rank 15.
    
    But wait, we can share computations. Let me think about this more carefully.
    
    A (2x3) * B (3x3) where B = [col1, col2, col3]
    C1 = A * col1 (2x1)
    C2 = A * col2 (2x1)  
    C3 = A * col3 (2x1)
    
    Each Ci requires rank 5, so total rank 15 if we can share.
    
    Actually, the rank of the 2x3x3 matrix multiplication tensor is known to be 17.
    Let me use a known decomposition.
    
    For now, let's create a structured random initialization that's closer to the solution.
    """
    n, m, p = 2, 3, 3
    rank = 17
    
    # Create a structured initialization
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Pattern 1: Identity-like terms (6 of them for the diagonal)
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            U[idx, j] = 1.0  # A[i,j]
            V[j * p + j, j] = 1.0  # B[j,j] -- this is wrong, let me fix
    
    # Let me use a different approach: start with schoolbook and then perturb
    # to allow for rank reduction
    
    # Schoolbook has 18 terms. For rank 17, we need to merge two terms.
    # Create schoolbook first
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Now we have 18 terms in 17 slots. Merge the last two terms.
    # This creates a linear combination that might allow cancellation
    # Actually, we can't just merge - we need to find a valid rank-17 decomposition.
    
    # Instead, let's use random initialization with structure
    rng = np.random.default_rng(42)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        # Each rank-1 term should have sparse structure
        # Pick random non-zero positions
        nz_u = rng.integers(2, 5)
        nz_v = rng.integers(2, 5)
        nz_w = rng.integers(2, 5)
        
        idx_u = rng.choice(n*m, size=nz_u, replace=False)
        idx_v = rng.choice(m*p, size=nz_v, replace=False)
        idx_w = rng.choice(p*n, size=nz_w, replace=False)
        
        U[idx_u, r] = rng.choice([-1.0, -0.5, 0.5, 1.0])
        V[idx_v, r] = rng.choice([-1.0, -0.5, 0.5, 1.0])
        W[idx_w, r] = rng.choice([-1.0, -0.5, 0.5, 1.0])
    
    return U, V, W
>>>>>>> REPLACE

