"""Symbolic regression with data-driven basis selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev(t, deg):
    """Compute Chebyshev polynomial of degree deg at points t."""
    if deg == 0:
        return np.ones_like(t)
    elif deg == 1:
        return t.copy()
    else:
        T_prev = np.ones_like(t)
        T_curr = t.copy()
        for _ in range(2, deg + 1):
            T_next = 2 * t * T_curr - T_prev
            T_prev = T_curr
            T_curr = T_next
        return T_curr


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: linear combination of 10 data-selected basis functions.
    Generates a diverse pool of candidate features, uses QR with column
    pivoting to select the 10 most linearly independent ones, then returns
    the linear combination with the given params.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    n = x.shape[0]
    
    # Generate a diverse pool of candidate basis functions
    candidates = []
    
    # Chebyshev polynomials up to degree 6
    max_deg = 6
    Tx = [_chebyshev(x, d) for d in range(max_deg + 1)]
    Ty = [_chebyshev(y, d) for d in range(max_deg + 1)]
    
    # Add polynomial terms
    for d in range(max_deg + 1):
        candidates.append(x ** d)
        candidates.append(y ** d)
    
    # Add trigonometric terms
    for k in range(1, 4):
        candidates.append(np.sin(k * x))
        candidates.append(np.cos(k * x))
        candidates.append(np.sin(k * y))
        candidates.append(np.cos(k * y))
    
    # Add Chebyshev product terms
    for i in range(max_deg + 1):
        for j in range(max_deg + 1):
            if i + j <= max_deg + 1:
                candidates.append(Tx[i] * Ty[j])
    
    # Add radial features
    r = np.sqrt(x ** 2 + y ** 2)
    for d in range(1, 5):
        candidates.append(r ** d)
    
    # Add specific interaction terms
    candidates.append(x * y)
    candidates.append(x ** 2 * y)
    candidates.append(x * y ** 2)
    candidates.append(x ** 3 * y)
    candidates.append(x * y ** 3)
    candidates.append(x ** 2 * y ** 2)
    
    # Stack all candidates into a matrix (n_samples x n_candidates)
    A = np.column_stack(candidates)
    
    # Replace non-finite values with 0
    A = np.where(np.isfinite(A), A, 0.0)
    
    # Use QR decomposition with column pivoting to select top 10 features
    try:
        Q, R, P = np.linalg.qr(A, mode='complete', pivoting=True)
        selected_indices = P[:10]
        B = A[:, selected_indices]
        
        # Check condition number; fall back if ill-conditioned
        cond = np.linalg.cond(B)
        if not np.isfinite(cond) or cond > 1e12:
            raise np.linalg.LinAlgError("Ill-conditioned")
    except Exception:
        # Fallback to a well-conditioned Chebyshev basis
        B = np.column_stack([
            np.ones(n),
            x,
            y,
            2 * x**2 - 1,
            2 * y**2 - 1,
            x * y,
            4 * x**3 - 3 * x,
            4 * y**3 - 3 * y,
            (2 * x**2 - 1) * y,
            x * (2 * y**2 - 1)
        ])
    
    # Return linear combination
    return B @ params
# EVOLVE-BLOCK-END
