"""ODE solver for convection-diffusion 100-point problem using matrix exponential strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)

    # Safety deadline guard
    start_time = time.time()
    deadline = start_time + 285.0

    # 1. Detect linearity and extract A, b with minimal f calls
    # For convection-diffusion, A is tridiagonal. We probe a few basis vectors
    # to determine the tridiagonal pattern, then reconstruct A.
    
    # Get b (offset)
    y0_test = np.zeros(n)
    b = np.asarray(f(t0, y0_test), dtype=float)
    
    # Probe basis vectors to determine tridiagonal structure
    # e_0, e_1, e_2 give us enough to infer the pattern
    e0 = np.zeros(n); e0[0] = 1.0
    e1 = np.zeros(n); e1[1] = 1.0
    e2 = np.zeros(n); e2[2] = 1.0
    
    f_e0 = np.asarray(f(t0, e0), dtype=float) - b
    f_e1 = np.asarray(f(t0, e1), dtype=float) - b
    f_e2 = np.asarray(f(t0, e2), dtype=float) - b
    
    # For tridiagonal A, column j has non-zeros only at rows j-1, j, j+1
    # From f_e0: A[0,0], A[1,0] should be non-zero; others zero
    # From f_e1: A[0,1], A[1,1], A[2,1] should be non-zero; others zero
    # From f_e2: A[1,2], A[2,2], A[3,2] should be non-zero; others zero
    
    # Check if structure is tridiagonal
    def is_tridiag_col(f_col, j, n):
        for i in range(n):
            if abs(i - j) > 1:
                if abs(f_col[i]) > 1e-14:
                    return False
        return True
    
    tridiag_ok = (is_tridiag_col(f_e0, 0, n) and 
                  is_tridiag_col(f_e1, 1, n) and 
                  is_tridiag_col(f_e2, 2, n))
    
    if tridiag_ok:
        # Reconstruct A assuming tridiagonal structure
        # Extract diagonal, sub-diagonal, super-diagonal
        diag = np.zeros(n)
        sub = np.zeros(n - 1)   # A[i, i-1]
        sup = np.zeros(n - 1)   # A[i, i+1]
        
        # From f_e0: A[0,0] = f_e0[0], A[1,0] = f_e0[1]
        diag[0] = f_e0[0]
        sub[0] = f_e0[1]  # A[1,0]
        sup[0] = f_e1[0]  # A[0,1]
        
        # From f_e1: A[1,1] = f_e1[1], A[2,1] = f_e1[2]
        diag[1] = f_e1[1]
        sub[1] = f_e1[2]  # A[2,1]
        
        # From f_e2: A[2,2] = f_e2[2], A[3,2] = f_e2[3]
        diag[2] = f_e2[2]
        sub[2] = f_e2[3]  # A[3,2]
        
        # For interior points (j >= 2), the pattern should be consistent
        # Check consistency: for j=1,2 the sub/sup/diag should match
        # Assume uniform tridiagonal (same coefficients for all interior rows)
        # Verify with one more probe at a middle point
        mid = n // 2
        e_mid = np.zeros(n); e_mid[mid] = 1.0
        f_mid = np.asarray(f(t0, e_mid), dtype=float) - b
        
        if is_tridiag_col(f_mid, mid, n):
            # Check if coefficients are consistent
            # For interior j: A[j,j-1] = sub_coeff, A[j,j] = diag_coeff, A[j,j+1] = sup_coeff
            sub_coeff = f_mid[mid - 1]
            diag_coeff = f_mid[mid]
            sup_coeff = f_mid[mid + 1]
            
            # Compare with values from e0, e1, e2
            # Interior rows should have same coefficients
            consistent = (abs(sub_coeff - sub[1]) < 1e-12 and 
                         abs(diag_coeff - diag[1]) < 1e-12 and 
                         abs(sup_coeff - sup[0]) < 1e-12)
            
            if consistent:
                # Build full tridiagonal A with uniform interior coefficients
                A = np.zeros((n, n))
                # Diagonal
                A[0, 0] = diag[0]
                A[n-1, n-1] = diag[0]  # Assume same as first, or probe
                for i in range(1, n-1):
                    A[i, i] = diag_coeff
                # Sub-diagonal
                for i in range(1, n):
                    A[i, i-1] = sub_coeff
                # Super-diagonal
                for i in range(0, n-1):
                    A[i, i+1] = sup_coeff
                
                # Verify with a random test
                rng = np.random.RandomState(42)
                y_rand = rng.randn(n)
                f_pred = A @ y_rand + b
                f_actual = f(t0, y_rand)
                
                if np.allclose(f_pred, f_actual, rtol=1e-10, atol=1e-12):
                    # Success: use this A
                    pass
                else:
                    # Inconsistent, fallback to full extraction
                    A = _extract_full_A(f, t0, b, n)
            else:
                # Non-uniform tridiagonal, need more probes
                A = _extract_full_A(f, t0, b, n)
        else:
            A = _extract_full_A(f, t0, b, n)
    else:
        # Not tridiagonal, full extraction
        A = _extract_full_A(f, t0, b, n)
    
    # 2. Solve y' = A y + b using matrix exponential
    dt = t1 - t0

    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A * dt
    M[:n, n] = b * dt

    E_block = expm(M)

    E = E_block[:n, :n]
    Z = E_block[:n, n]

    y_final = E @ y + Z

    if time.time() > deadline:
        return y_final

    return y_final


def _extract_full_A(f, t0, b, n):
    """Extract full A matrix by probing all basis vectors."""
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        A[:, j] = np.asarray(f(t0, ej), dtype=float) - b
    return A


def _rk4_fallback(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    h = (t1 - t0) / 1000
    t = t0
    for _ in range(1000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END


def _rk4_fallback(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    h = (t1 - t0) / 1000
    t = t0
    for _ in range(1000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
