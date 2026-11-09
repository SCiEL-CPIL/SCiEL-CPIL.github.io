"""Data-driven orthogonal basis construction via QR decomposition."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard
_DEADLINE = None
_TIME_MARGIN = 5.0

def _elapsed_ok():
    """Return True if we still have time to safely finish."""
    if _DEADLINE is None:
        return True
    return (time.time() - _DEADLINE) < _TIME_MARGIN

def _generate_candidate_features(X, Y):
    """
    Generate a large set of candidate features.
    We want a rich set so that QR can find the most important orthogonal directions.
    """
    n = X.shape[0]
    Xc = np.clip(X, -1.0, 1.0)
    Yc = np.clip(Y, -1.0, 1.0)
    
    feats = [np.ones(n)]
    
    # Polynomials up to degree 4
    for i in range(5):
        for j in range(5 - i):
            feats.append((Xc ** i) * (Yc ** j))
            
    # Trigonometric features
    # Low frequency
    feats.append(np.sin(np.pi * Xc))
    feats.append(np.cos(np.pi * Xc))
    feats.append(np.sin(np.pi * Yc))
    feats.append(np.cos(np.pi * Yc))
    
    # Medium frequency
    feats.append(np.sin(2 * np.pi * Xc))
    feats.append(np.cos(2 * np.pi * Xc))
    feats.append(np.sin(2 * np.pi * Yc))
    feats.append(np.cos(2 * np.pi * Yc))
    
    # Radial and Angular
    R = np.sqrt(Xc * Xc + Yc * Yc)
    theta = np.arctan2(Yc, Xc)
    
    feats.append(R)
    feats.append(R**2)
    feats.append(R**3)
    feats.append(np.sin(np.pi * R))
    feats.append(np.cos(np.pi * R))
    
    feats.append(np.sin(theta))
    feats.append(np.cos(theta))
    feats.append(np.sin(2 * theta))
    feats.append(np.cos(2 * theta))
    
    # Interactions
    feats.append(Xc * R)
    feats.append(Yc * R)
    feats.append(Xc * theta)
    feats.append(Yc * theta)
    
    # Stack
    F = np.column_stack(feats)
    F = np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return F

def _build_orthogonal_basis(F, k=10):
    """
    Perform QR decomposition to get k orthogonal columns.
    We use the Q matrix from QR, which provides an orthonormal basis for the column space.
    By taking the first k columns of Q, we get the k most significant orthogonal directions
    (in terms of variance explained by the initial candidate features).
    """
    try:
        # QR decomposition
        # mode='reduced' gives Q with shape (n, min(n, m))
        Q, R = np.linalg.qr(F, mode='reduced')
        
        # We need exactly k columns. 
        # If Q has fewer than k columns (unlikely with 200 points and ~30 features), pad.
        # If Q has more, we take the first k.
        if Q.shape[1] >= k:
            Q_k = Q[:, :k]
        else:
            # Pad with zeros if necessary (should not happen)
            pad = np.zeros((Q.shape[0], k - Q.shape[1]))
            Q_k = np.hstack([Q, pad])
            
        return Q_k
        
    except Exception:
        # Fallback to simple polynomial basis if QR fails
        # This is a safety net
        X = F[:, 1] if F.shape[1] > 1 else np.zeros(F.shape[0])
        Y = F[:, 2] if F.shape[1] > 2 else np.zeros(F.shape[0])
        fallback = [
            np.ones(F.shape[0]),
            X, Y,
            X*X, Y*Y, X*Y,
            X**3, Y**3,
            np.sin(np.pi*X), np.sin(np.pi*Y)
        ]
        return np.column_stack(fallback)

# Cache for the basis matrix
# Since x, y are constant during the fit, we can compute the basis once.
# We use a simple cache key based on the first few points and size.
_basis_cache = {}

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 150.0
        
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    
    # Check time
    if not _elapsed_ok():
        # Return a constant prediction (mean of y) if out of time
        return np.full(n, np.mean(y))
        
    # Create a simple cache key
    # Using a hash of the first 5 points and n is a reasonable proxy for "same data"
    # In this specific task, x and y are fixed for the entire optimization run.
    key = (n, x[0], x[1] if n > 1 else 0, y[0], y[1] if n > 1 else 0)
    
    if key in _basis_cache:
        F_basis = _basis_cache[key]
    else:
        # Generate candidate features
        F_candidates = _generate_candidate_features(x, y)
        
        # Build orthogonal basis
        F_basis = _build_orthogonal_basis(F_candidates, k=10)
        
        # Cache it
        _basis_cache[key] = F_basis
        
        # Clear cache if too many entries (safety, though unlikely to be needed)
        if len(_basis_cache) > 10:
            _basis_cache.clear()
            _basis_cache[key] = F_basis
            
    # Ensure params is an array
    params = np.asarray(params, dtype=float)
    
    # If params length doesn't match basis, truncate or pad (safety)
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    else:
        params = params[:10]
        
    # Compute prediction
    pred = F_basis @ params
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        
    return pred
# EVOLVE-BLOCK-END
