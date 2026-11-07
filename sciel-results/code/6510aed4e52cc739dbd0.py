"""Symbolic regression for E. coli growth rate - GAM with B-spline-like basis."""
import numpy as np


def _legendre_basis(x, n=5):
    """Compute first n Legendre polynomials evaluated at x (vectorized)."""
    # x is expected in [-1, 1] or we normalize it
    P0 = np.ones_like(x, dtype=float)
    if n <= 1:
        return P0[:, np.newaxis]
    P1 = x.copy()
    polys = [P0, P1]
    for k in range(2, n):
        Pk = ((2 * k - 1) * x * P1 - (k - 1) * P0) / k
        polys.append(Pk)
        P0, P1 = P1, Pk
    return np.stack(polys, axis=1)


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Generalized Additive Model (GAM) using Legendre polynomial basis functions.
    
    The model is linear in the 10 parameters, which makes it amenable to BFGS
    optimization without local minima issues.
    
    Inputs are normalized to [-1, 1] range.
    Each input gets 2 Legendre basis terms (P0, P1, P2, P3) -> 4*4 = 16 terms total.
    But we only have 10 params, so we use a subset:
    - b: P0, P1 (2)
    - s: P0, P1 (2)  
    - temp: P0, P1, P2 (3)
    - pH: P0, P1 (2)
    - interaction b*s: P0 (1)
    Total: 10
    """
    p = params
    
    # Normalize inputs to [-1, 1]
    # b in [1, 4.998] -> [-1, 1]
    b_norm = 2.0 * (b - 1.0) / (4.998 - 1.0) - 1.0
    b_norm = np.clip(b_norm, -1.0, 1.0)
    
    # s in [1, 39.98] -> [-1, 1]
    s_norm = 2.0 * (s - 1.0) / (39.98 - 1.0) - 1.0
    s_norm = np.clip(s_norm, -1.0, 1.0)
    
    # temp in [1, 12] -> [-1, 1]
    temp_norm = 2.0 * (temp - 1.0) / (12.0 - 1.0) - 1.0
    temp_norm = np.clip(temp_norm, -1.0, 1.0)
    
    # pH in [1, 4.999] -> [-1, 1]
    ph_norm = 2.0 * (pH - 1.0) / (4.999 - 1.0) - 1.0
    ph_norm = np.clip(ph_norm, -1.0, 1.0)
    
    # Compute Legendre basis for each variable
    # b: P0, P1
    b_basis = np.stack([np.ones_like(b_norm), b_norm], axis=1)  # shape (n, 2)
    
    # s: P0, P1
    s_basis = np.stack([np.ones_like(s_norm), s_norm], axis=1)  # shape (n, 2)
    
    # temp: P0, P1, P2
    P0_t = np.ones_like(temp_norm)
    P1_t = temp_norm.copy()
    P2_t = (3.0 * temp_norm ** 2 - 1.0) / 2.0
    temp_basis = np.stack([P0_t, P1_t, P2_t], axis=1)  # shape (n, 3)
    
    # pH: P0, P1
    ph_basis = np.stack([np.ones_like(ph_norm), ph_norm], axis=1)  # shape (n, 2)
    
    # interaction: b_norm * s_norm (one term)
    interact = b_norm * s_norm  # shape (n,)
    
    # Build design matrix: 10 columns
    # [b_P0, b_P1, s_P0, s_P1, temp_P0, temp_P1, temp_P2, ph_P0, ph_P1, interact]
    design = np.column_stack([
        b_basis[:, 0],
        b_basis[:, 1],
        s_basis[:, 0],
        s_basis[:, 1],
        temp_basis[:, 0],
        temp_basis[:, 1],
        temp_basis[:, 2],
        ph_basis[:, 0],
        ph_basis[:, 1],
        interact
    ])
    
    # Linear model: y = design @ params
    mu = design @ p
    
    return mu
