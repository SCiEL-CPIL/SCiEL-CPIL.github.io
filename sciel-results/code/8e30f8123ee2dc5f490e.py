"""
Heat2D 10x10 ODE solver.

Strategy:
- The heat equation discretised on a 10x10 grid gives a linear, autonomous,
  stiff ODE: y' = A y (with possibly boundary handling, but linear).
- We estimate the matrix A using a small number of function evaluations.
- Since the system is linear and time-independent, we can use the matrix
  exponential expm(A*t) * y0 to get the exact solution.
- We estimate A by probing the structure of the linear operator.

Key insight: For a 5-point stencil Laplacian on a 10x10 grid, we can infer
the stencil from a single interior row and verify with one random vector.
This reduces f-calls from ~106 to just 2 in the common case.

If the structure doesn't match, we fall back to full matrix construction.
"""
import numpy as np
from scipy.linalg import expm


def _try_infer_stencil(f, y0, t0):
    """
    Try to infer the linear operator A from the structure of f.
    For a 5-point stencil Laplacian, we can determine A with minimal calls.
    
    Returns (A, success) where A is the 100x100 matrix if successful.
    """
    n = len(y0)
    if n != 100:
        return None, False
    
    # Call f once to get A*y0
    Ay0 = np.array(f(t0, y0), dtype=float)
    
    # For a 5-point stencil, the matrix is tridiagonal-like with specific structure.
    # Let's probe by creating test vectors with only one nonzero element.
    
    # First, let's try to determine if this is a standard Laplacian.
    # Create a vector with 1 in the center (index 55 for 10x10 grid, row 5, col 5)
    # and 0 elsewhere.
    
    # Actually, let's use a smarter approach:
    # Call f with a vector that has 1 at position i and 0 elsewhere,
    # for a few strategic positions to determine the stencil.
    
    # For a 10x10 grid, position (row, col) maps to index row*10 + col.
    # Interior point (5,5) -> index 55
    # Its neighbors: (4,5)=45, (6,5)=65, (5,4)=54, (5,6)=56, (5,5)=55
    
    # Let's probe with a single 1 at an interior point
    test_vec = np.zeros(n)
    test_vec[55] = 1.0  # center of 10x10 grid
    
    A_test = np.array(f(t0, test_vec), dtype=float)
    
    # For a standard 5-point Laplacian with Dirichlet BCs:
    # A[55,55] = -4, A[45,55] = 1, A[65,55] = 1, A[54,55] = 1, A[56,55] = 1
    # (or with some scaling factor)
    
    # Check if the pattern matches
    diag_val = A_test[55]
    up_val = A_test[45]
    down_val = A_test[65]
    left_val = A_test[54]
    right_val = A_test[56]
    
    # For a standard Laplacian, these should be:
    # diag = -4 (or -4*h^2 scaled), neighbors = 1 (or h^2 scaled)
    # Check if the ratio is consistent
    if abs(diag_val) > 1e-15:
        ratio_up = up_val / diag_val
        ratio_down = down_val / diag_val
        ratio_left = left_val / diag_val
        ratio_right = right_val / diag_val
        
        # For standard 5-point: ratios should all be -0.25
        if (abs(ratio_up + 0.25) < 1e-6 and 
            abs(ratio_down + 0.25) < 1e-6 and
            abs(ratio_left + 0.25) < 1e-6 and
            abs(ratio_right + 0.25) < 1e-6):
            
            # This looks like a standard 5-point Laplacian!
            # Now let's verify with another point to be sure.
            # Probe corner-adjacent point (1,1) -> index 11
            test_vec2 = np.zeros(n)
            test_vec2[11] = 1.0  # (row 1, col 1)
            
            A_test2 = np.array(f(t0, test_vec2), dtype=float)
            
            # For (1,1): neighbors are (0,1)=1, (2,1)=21, (1,0)=10, (1,2)=12
            # But (0,1) is a boundary, so for Dirichlet BC, the stencil might be different
            # or the boundary might be handled differently.
            
            # Let's just check if the diagonal is consistent
            diag2 = A_test2[11]
            if abs(diag2) > 1e-15 and abs(diag2 - diag_val) < 1e-6:
                # Consistent! Let's construct the full matrix.
                # For a 10x10 5-point Laplacian with Dirichlet BCs:
                A = np.zeros((n, n))
                for i in range(10):
                    for j in range(10):
                        idx = i * 10 + j
                        # Diagonal
                        A[idx, idx] = diag_val
                        # Neighbors (only if interior)
                        if i > 0:
                            A[idx, (i-1)*10 + j] = up_val
                        if i < 9:
                            A[idx, (i+1)*10 + j] = down_val
                        if j > 0:
                            A[idx, i*10 + (j-1)] = left_val
                        if j < 9:
                            A[idx, i*10 + (j+1)] = right_val
                
                # Verify with the original y0
                A_y0_check = A @ y0
                if np.allclose(A_y0_check, Ay0, rtol=1e-10, atol=1e-12):
                    return A, True
                else:
                    return None, False
            else:
                return None, False
        else:
            return None, False
    else:
        return None, False


def _construct_full_matrix(f, t0, y0):
    """
    Construct the full matrix A by probing each column.
    This requires n calls to f (one per column).
    """
    n = len(y0)
    A = np.zeros((n, n))
    
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float)
    
    return A


def _verify_linearity(f, t0, y0, A, num_checks=3):
    """
    Verify that f(t, y) = A @ y for several random vectors.
    """
    n = len(y0)
    for _ in range(num_checks):
        test_vec = np.random.randn(n)
        f_val = np.array(f(t0, test_vec), dtype=float)
        A_val = A @ test_vec
        if not np.allclose(f_val, A_val, rtol=1e-10, atol=1e-12):
            return False
    return True


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Try to infer the stencil structure first (uses 2-3 f calls)
    A, success = _try_infer_stencil(f, y0, t0)
    
    if not success:
        # Fall back to full matrix construction (n f calls)
        A = _construct_full_matrix(f, t0, y0)
    
    # Verify linearity with a random vector
    if not _verify_linearity(f, t0, y0, A):
        # Linearity check failed - this shouldn't happen for the heat equation,
        # but fall back to RK4 just in case
        return _rk4_fallback(f, t0, t1, y0)
    
    # Use matrix exponential for exact solution
    dt = t1 - t0
    expm_A_dt = expm(A * dt)
    y_final = expm_A_dt @ y0
    
    return y_final


def _rk4_fallback(f, t0, t1, y0, steps=10000):
    """
    Fallback RK4 integrator if linearity check fails.
    """
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
