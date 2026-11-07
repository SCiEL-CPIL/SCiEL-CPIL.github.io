"""
Strategy: Fourier Spectral Regression via Orthogonal Basis Projection.
We model the hidden function f(x, y) as a linear combination of a rich basis of
2D Fourier modes (cos/sin) and low-order polynomials.
Since the domain is [-1, 1]^2, we map inputs to [0, 2*pi]^2 to use standard Fourier bases.
We use np.linalg.lstsq to find the coefficients that minimize training MSE.
This is a global, parametric approach that contrasts with the local, non-parametric
kernel regression of the current program. It is robust to noise and captures
global structure well.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 115.0  # seconds of budget to spend inside equation()


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.asarray(x, float).ravel(),
                         np.asarray(y, float).ravel()])
    t = np.asarray(y, float).ravel()
    n = X.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Baseline: constant prediction (mean of t)
    mean_t = float(np.mean(t))
    best_pred = np.full(n, mean_t)
    best_score = float(np.mean((t - mean_t)**2))
    
    if best_score < 1e-12:
        return best_pred

    # Map inputs to [0, 2*pi] for Fourier basis
    # x, y in [-1, 1] -> u, v in [0, 2*pi]
    u = (X[:, 0] + 1.0) * np.pi
    v = (X[:, 1] + 1.0) * np.pi
    
    # Construct Basis Matrix
    # We include:
    # 1. Constant term (1)
    # 2. Low-order polynomials (u, v, u^2, v^2, uv) - helps with non-periodic trends
    # 3. Fourier modes up to order K
    #    cos(k*u), sin(k*u), cos(k*v), sin(k*v), cos(k*u)*cos(k*v), etc.
    # To keep the basis size manageable and orthogonal-ish, we focus on pure modes and some interactions.
    
    K_max = 12 # Max frequency
    
    basis_cols = [np.ones(n)]
    
    # Polynomials (optional, but helps if function has non-periodic components)
    # Actually, pure Fourier is better for periodic. But inputs are uniform on [-1,1].
    # Let's stick to Fourier + maybe a few low-order terms to capture DC offset and linear trends.
    # However, Fourier series already includes the mean (constant term).
    # Let's add a few polynomial terms to handle non-periodic behavior better.
    basis_cols.append(u)
    basis_cols.append(v)
    basis_cols.append(u * v)
    
    # Fourier terms
    for k in range(1, K_max + 1):
        if _time_left() < 10.0:
            break
            
        # Pure u modes
        basis_cols.append(np.cos(k * u))
        basis_cols.append(np.sin(k * u))
        
        # Pure v modes
        basis_cols.append(np.cos(k * v))
        basis_cols.append(np.sin(k * v))
        
        # Interaction terms (Cartesian product of low freqs)
        # To keep dimensionality low, we only add interactions for small k
        if k <= 4:
             basis_cols.append(np.cos(k * u) * np.cos(k * v))
             basis_cols.append(np.cos(k * u) * np.sin(k * v))
             basis_cols.append(np.sin(k * u) * np.cos(k * v))
             basis_cols.append(np.sin(k * u) * np.sin(k * v))
             
             # Mixed low freqs
             if k > 1:
                 basis_cols.append(np.cos(u) * np.cos(k * v))
                 basis_cols.append(np.cos(k * u) * np.cos(v))
                 basis_cols.append(np.sin(u) * np.sin(k * v))
                 basis_cols.append(np.sin(k * u) * np.sin(v))
    
    B = np.column_stack(basis_cols)
    
    # Check for finite values
    if not np.all(np.isfinite(B)):
        return best_pred
        
    # Solve Least Squares
    try:
        # Use lstsq for stability
        coeffs, residuals, rank, s = np.linalg.lstsq(B, t, rcond=None)
        pred = B @ coeffs
        
        if not np.all(np.isfinite(pred)):
            return best_pred
            
        mse = float(np.mean((pred - t)**2))
        
        if mse < best_score:
            best_score = mse
            best_pred = pred
            
    except Exception:
        pass

    # Fallback
    if not np.all(np.isfinite(best_pred)):
        return np.full(n, mean_t)

    return best_pred
# EVOLVE-BLOCK-END
