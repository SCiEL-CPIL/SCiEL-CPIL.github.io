import numpy as np
import time
from fractions import Fraction
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Check that sum_r U[:,r] * V[:,r] * W[:,r] equals the matrix multiplication tensor.
    Output factor follows W[k*n+i, r] convention."""
    r = len(U[0])
    # Build tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Target: T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiply index
    # For C[i,k] = sum_j A[i,j]*B[j,k], the tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # Actually the multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is all-ones of shape (n, m, p)
    T = np.zeros((n, m, p))
    for rr in range(r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
    # Target is all ones
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)


def _try_rank_exact(r, n=2, m=3, p=3, deadline=None):
    """Try to find exact rational decomposition of rank r.
    Uses the fact that for small ranks we can set up polynomial equations and solve.
    
    For rank r, we have r rank-1 terms. Each term is indexed by (i_r, j_r, k_r)
    where i_r in [0,n), j_r in [0,m), k_r in [0,p).
    The decomposition is:
    sum_{r=1}^{R} u_r * v_r * w_r where u_r is a vector of length n*m with a single 1 at position i_r*m+j_r,
    v_r has a single 1 at j_r*p+k_r, w_r has a single 1 at k_r*n+i_r.
    
    But this is too restrictive. We need general rank-1 tensors.
    
    A rank-1 tensor in the (n,m,p) space is a product of three vectors:
    a_r (length n) * b_r (length m) * c_r (length p)
    
    So U[i*m+j, r] = a_r[i] * b_r[j]
    V[j*p+k, r] = b_r[j] * c_r[k]  
    W[k*n+i, r] = c_r[k] * a_r[i]
    
    Wait, that's not right either. Let me think more carefully.
    
    The multiplication tensor T is in R^{n x m x p} where T[i,j,k] = 1 for all valid indices.
    A rank-1 tensor is T_r[i,j,k] = a_r[i] * b_r[j] * c_r[k].
    
    So we need: sum_{r=1}^{R} a_r[i] * b_r[j] * c_r[k] = 1 for all i,j,k.
    
    The factor matrices are:
    U[i*m+j, r] = a_r[i] * b_r[j]
    V[j*p+k, r] = b_r[j] * c_r[k]
    W[k*n+i, r] = c_r[k] * a_r[i]
    
    This gives us 3*r*(n+m+p) unknowns (the a_r, b_r, c_r vectors)
    and n*m*p = 18 equations.
    
    For r >= 9 (the lower bound), we have more unknowns than equations, so solutions should exist.
    """
    # Unknowns: for each rank term r, we have a_r (n values), b_r (m values), c_r (p values)
    # Total unknowns: r * (n + m + p) = r * 8
    # Equations: n * m * p = 18
    
    # The equations are: for each (i,j,k): sum_r a_r[i]*b_r[j]*c_r[k] = 1
    
    # This is a system of 18 polynomial equations in 8r unknowns.
    # For r=9: 72 unknowns, 18 equations -> very underdetermined
    # For r=10: 80 unknowns, 18 equations
    # etc.
    
    # Strategy: fix some variables to simplify, then solve.
    # We can use the scaling freedom: multiplying a_r by t, b_r by 1/t, c_r by 1 doesn't change the product.
    # So we can fix, say, a_r[0] = 1 for each r (if n >= 1).
    
    # Let's try a constructive approach for small r.
    # For r = n*m*p = 18, schoolbook works.
    # For r < 18, we need to find if a solution exists.
    
    # Use a numerical approach first to check feasibility, then rationalize.
    
    # Set up the system symbolically using a grid of values.
    # For simplicity, try to find solutions where a_r, b_r, c_r have values in {0, 1, -1, 1/2, -1/2, 2, -2}
    
    # Actually, let's try a different approach: use the known structure.
    # For 2x3x3, the rank is known to be 16 (or possibly 15).
    # Let me try to construct a rank-16 decomposition.
    
    # Known result: The rank of 2x3 matrix multiplication is 16.
    # Reference: Strassen, or subsequent work.
    
    # Let me try to find it by solving the polynomial system numerically.
    
    # Number of unknowns for rank r: r*(n+m+p) = 8r
    # Number of equations: 18
    # For r=16: 128 unknowns, 18 equations
    
    # We can fix the scaling: set a_r[0] = 1 for all r (since n=2, a_r has 2 components)
    # This removes r degrees of freedom, leaving 8r - r = 7r unknowns.
    # Still very underdetermined.
    
    # Let's try a randomized numerical approach:
    # 1. Randomly initialize a_r, b_r, c_r
    # 2. Use Newton's method or least squares to solve
    
    # Actually, since we have more unknowns than equations, we can use a simpler approach:
    # Fix all but 18 unknowns to simple values, then solve for the remaining 18.
    
    # Let me try: for each rank term r, set most of a_r, b_r, c_r to 0 or 1,
    # and leave a few free variables.
    
    # Simpler: try the "block" construction.
    # For 2x3x3, think of it as multiplying two 2x3 blocks.
    
    # Let me just try to verify if rank 16 is achievable with a specific construction.
    # I'll use a known algorithm structure.
    
    # For now, let's try a numerical search with rational reconstruction.
    
    if deadline is not None and time.time() > deadline:
        return None
    
    # Try rank from r down to 9
    # For each rank, try to find a solution
    
    # Use a simple iterative method:
    # Start with random initialization and minimize the residual
    
    best_result = None
    for attempt in range(50):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization with rational values
        np.random.seed(42 + attempt)
        
        # a: shape (r, n), b: shape (r, m), c: shape (r, p)
        a = np.random.randn(r, n)
        b = np.random.randn(r, m)
        c = np.random.randn(r, p)
        
        # Normalize to avoid overflow
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-10)
        
        # Use gradient descent on the residual
        lr = 0.01
        for it in range(200):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute residual
            residual = np.zeros((n, m, p))
            for rr in range(r):
                residual += a[rr, :, None, None] * b[rr, None, :, None] * c[rr, None, None, :]
            
            diff = residual - 1.0
            
            # Gradient with respect to a, b, c
            grad_a = np.zeros_like(a)
            grad_b = np.zeros_like(b)
            grad_c = np.zeros_like(c)
            
            for rr in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # d/d a[rr,i] of a[rr,i]*b[rr,j]*c[rr,k] = b[rr,j]*c[rr,k]
                            grad_a[rr, i] += diff[i, j, k] * b[rr, j] * c[rr, k]
                            grad_b[rr, j] += diff[i, j, k] * a[rr, i] * c[rr, k]
                            grad_c[rr, k] += diff[i, j, k] * a[rr, i] * b[rr, j]
            
            # Update
            a -= lr * grad_a
            b -= lr * grad_b
            c -= lr * grad_c
        
        # Check if we found a solution
        residual = np.zeros((n, m, p))
        for rr in range(r):
            residual += a[rr, :, None, None] * b[rr, None, :, None] * c[rr, None, None, :]
        
        if np.max(np.abs(residual - 1.0)) < 1e-6:
            # Found a numerical solution, try to rationalize
            # Scale to get nice fractions
            # For each rank term, we can scale a_r by t, b_r by 1/t
            # Let's try to make values close to rationals with small denominators
            
            a_rat = []
            b_rat = []
            c_rat = []
            success = True
            for rr in range(r):
                a_row = []
                b_row = []
                c_row = []
                for val in a[rr]:
                    frac = Fraction(val).limit_denominator(100)
                    a_row.append(frac)
                for val in b[rr]:
                    frac = Fraction(val).limit_denominator(100)
                    b_row.append(frac)
                for val in c[rr]:
                    frac = Fraction(val).limit_denominator(100)
                    c_row.append(frac)
                a_rat.append(a_row)
                b_rat.append(b_row)
                c_rat.append(c_row)
            
            # Verify the rational solution
            residual_rat = np.zeros((n, m, p))
            for rr in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual_rat[i, j, k] += float(a_rat[rr][i]) * float(b_rat[rr][j]) * float(c_rat[rr][k])
            
            if np.max(np.abs(residual_rat - 1.0)) < 1e-9:
                # Build U, V, W
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                for rr in range(r):
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j][rr] = float(a_rat[rr][i]) * float(b_rat[rr][j])
                            W[k * n + i][rr] = 0  # placeholder
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k][rr] = float(b_rat[rr][j]) * float(c_rat[rr][k])
                            W[k * n + 0][rr] = float(c_rat[rr][k]) * float(a_rat[rr][0])
                
                # Fix W properly
                W = [[0] * r for _ in range(p * n)]
                for rr in range(r):
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i][rr] = float(c_rat[rr][k]) * float(a_rat[rr][i])
                
                # Verify
                if _verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin
    
    # Try ranks from 16 down to 9
    for r in range(16, 8, -1):
        if time.time() > deadline:
            break
        result = _try_rank_exact(r, n, m, p, deadline)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
