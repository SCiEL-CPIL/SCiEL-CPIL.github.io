"""Data-driven adaptive basis via iterative orthogonalization on candidate features."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = None
_TIME_MARGIN = 5.0

def _elapsed_ok():
    if _DEADLINE is None:
        return True
    return (time.time() - _DEADLINE) < _TIME_MARGIN

def _generate_candidate_pool(X, Y):
    """
    Generate a large pool of candidate univariate and bivariate features.
    All features are bounded since X, Y in [-1, 1].
    """
    n = X.shape[0]
    Xc = np.clip(X, -1.0, 1.0)
    Yc = np.clip(Y, -1.0, 1.0)
    
    candidates = []
    
    # Univariate in X: Chebyshev-like polynomials up to degree 8
    for d in range(1, 9):
        candidates.append(Xc ** d)
    
    # Univariate in Y: Chebyshev-like polynomials up to degree 8
    for d in range(1, 9):
        candidates.append(Yc ** d)
    
    # Trigonometric in X
    for k in [1, 2, 3]:
        candidates.append(np.sin(k * np.pi * Xc))
        candidates.append(np.cos(k * np.pi * Xc))
    
    # Trigonometric in Y
    for k in [1, 2, 3]:
        candidates.append(np.sin(k * np.pi * Yc))
        candidates.append(np.cos(k * np.pi * Yc))
    
    # Radial and angular
    R = np.sqrt(Xc * Xc + Yc * Yc)
    theta = np.arctan2(Yc, Xc)
    candidates.append(R)
    candidates.append(R ** 2)
    candidates.append(R ** 3)
    candidates.append(np.sin(np.pi * R))
    candidates.append(np.cos(np.pi * R))
    candidates.append(np.sin(theta))
    candidates.append(np.cos(theta))
    candidates.append(np.sin(2 * theta))
    candidates.append(np.cos(2 * theta))
    
    # Bivariate products: low-degree cross terms
    for i in range(1, 5):
        for j in range(1, 5):
            if i + j <= 5:  # Keep total degree <= 5 to avoid too many
                candidates.append((Xc ** i) * (Yc ** j))
    
    # Additional smooth nonlinearities
    candidates.append(np.exp(-Xc ** 2))
    candidates.append(np.exp(-Yc ** 2))
    candidates.append(np.exp(-R ** 2))
    candidates.append(np.tanh(Xc))
    candidates.append(np.tanh(Yc))
    candidates.append(np.tanh(Xc + Yc))
    candidates.append(np.tanh(Xc - Yc))
    candidates.append(np.abs(Xc) ** 1.5)
    candidates.append(np.abs(Yc) ** 1.5)
    candidates.append(np.sign(Xc) * np.sqrt(np.abs(Xc)))
    candidates.append(np.sign(Yc) * np.sqrt(np.abs(Yc)))
    
    # Stack into matrix: n x m
    pool = np.column_stack(candidates)
    pool = np.nan_to_num(pool, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Standardize each column
    means = pool.mean(axis=0)
    stds = pool.std(axis=0)
    stds[stds < 1e-12] = 1.0
    pool = (pool - means) / stds
    
    return pool

def _select_orthogonal_basis(pool, y, k_select=9):
    """
    Select k_select orthogonal directions from the pool that best explain y.
    Uses iterative Gram-Schmidt: at each step, pick the candidate with highest
    correlation with the residual, orthogonalize it against already-selected
    features, and add it to the basis.
    """
    n, m = pool.shape
    residual = y.copy()
    selected = []
    used_indices = set()
    
    # Pre-compute norms for efficiency
    for _ in range(k_select):
        # Find candidate with max absolute correlation with residual
        # corr = |<residual, col>| / (||residual|| * ||col||)
        # Since columns are standardized, ||col|| ~ sqrt(n)
        # So we just need max |<residual, col>|
        if len(selected) == 0:
            projections = pool.T @ residual
        else:
            # Project residual onto each candidate, but also need to account
            # for orthogonality to already-selected features
            # Simpler: just use correlation with current residual
            projections = pool.T @ residual
        
        # Exclude already used
        for idx in used_indices:
            projections[idx] = -np.inf
        
        best_idx = np.argmax(np.abs(projections))
        used_indices.add(best_idx)
        
        # Orthogonalize this candidate against all previously selected
        v = pool[:, best_idx].copy()
        for u in selected:
            coeff = np.dot(v, u) / (np.dot(u, u) + 1e-12)
            v = v - coeff * u
        
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            # Nearly linearly dependent; skip and try next best
            continue
        
        v = v / norm_v
        selected.append(v)
        
        # Update residual
        coeff = np.dot(residual, v)
        residual = residual - coeff * v
    
    # Build design matrix with intercept + selected orthogonal features
    F = np.column_stack([np.ones(n)] + selected)
    return F

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 150.0
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    
    # Generate candidate pool
    pool = _generate_candidate_pool(x, y)
    
    # Select 9 orthogonal directions + intercept = 10 features total
    # This matches the 10 params the evaluator provides
    F = _select_orthogonal_basis(pool, y, k_select=9)
    
    # F is n x 10: [intercept, ortho_1, ..., ortho_9]
    # params is length 10
    pred = F @ params
    
    # Ensure finite output
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
