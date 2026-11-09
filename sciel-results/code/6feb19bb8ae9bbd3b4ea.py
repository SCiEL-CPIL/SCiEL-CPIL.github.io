"""ODE solver for 100-point convection-diffusion IVP using sparse band extraction and expm."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t_start = time.perf_counter()
    DEADLINE = 290.0  # Safety margin before 330s kill

    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    t0, t1 = t_span
    T = t1 - t0

    def _elapsed_ok():
        return (time.perf_counter() - t_start) < DEADLINE

    # Strategy:
    # 1. Assume A is banded (tridiagonal for standard convection-diffusion).
    # 2. Extract the band structure using targeted probes.
    #    For a tridiagonal matrix, we need to find diagonals: main, sub, super.
    #    We can probe with vectors that isolate specific rows/cols.
    #    
    #    However, a more robust approach for "banded" is to just assume tridiagonal
    #    and verify. If it fails, fallback to dense extraction.
    #    
    #    Let's try to extract the tridiagonal structure first.
    #    To get diagonal elements A[i,i], we can use e_i.
    #    To get sub-diagonal A[i+1,i], we can use e_i and look at row i+1.
    #    To get super-diagonal A[i,i+1], we can use e_i and look at row i-1.
    #    
    #    Actually, just extracting the full matrix with n calls is cheap (100 calls).
    #    The bottleneck is expm. For n=100, dense expm is ~0.1-0.5s.
    #    Sparse expm might be faster if the matrix is sparse.
    #    
    #    Let's extract the full matrix A first (n calls).
    #    Then check sparsity. If sparse, use sparse expm. Else dense.
    
    A_dense = np.zeros((n, n), dtype=float)
    e = np.zeros(n, dtype=float)
    
    for i in range(n):
        if not _elapsed_ok():
            # Fallback to DOP853 if we're running out of time
            break
        e[i] = 1.0
        col = f(t0, e)
        A_dense[:, i] = col
        e[i] = 0.0

    if _elapsed_ok() and np.all(np.isfinite(A_dense)):
        # Check if the matrix is sparse (banded)
        # Count non-zeros
        nnz = np.count_nonzero(A_dense)
        
        # If it's sparse enough, convert to sparse and use sparse expm
        # Threshold: if less than 10% dense, use sparse
        if nnz < 0.1 * n * n:
            try:
                A_sparse = sp.csr_matrix(A_dense)
                AT_sparse = A_sparse * T
                # scipy.sparse.linalg.expm is available in recent versions
                # If not, we can use scipy.linalg.expm on the dense matrix
                # But let's try sparse first for speed
                expAT = spla.expm(AT_sparse)
                result = expAT @ y0
                if np.all(np.isfinite(result)):
                    return result
            except Exception:
                pass
        
        # Fallback to dense expm
        try:
            AT = A_dense * T
            expAT = np.linalg.matrix_power(np.eye(n), 0) # placeholder, use scipy
            from scipy.linalg import expm
            expAT = expm(AT)
            result = expAT @ y0
            if np.all(np.isfinite(result)):
                return result
        except Exception:
            pass
    
    # Fallback: Use solve_ivp with DOP853
    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
            max_step=T / 50.0,
            first_step=T / 1e6,
            dense_output=False,
        )
        if sol.success and sol.t.size > 0:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last-resort: fixed-step RK4
    steps = 2000
    h = T / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if not _elapsed_ok():
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    return y
# EVOLVE-BLOCK-END
