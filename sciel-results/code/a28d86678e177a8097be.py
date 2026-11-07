"""Symbolic regression with 1 input - flexible basis function approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Extract parameters
    p = params  # 10 parameters
    
    # Build a rich but stable basis of 10 functions
    # Mix polynomial, trigonometric, and rational-like terms
    x0 = np.ones_like(x)
    x1 = x
    x2 = x**2
    x3 = x**3
    x4 = x**4
    
    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    
    # Bounded rational-like terms (safe)
    # tanh(x) is smooth and bounded
    tanh_x = np.tanh(x)
    
    # Interaction term that captures asymmetric oscillatory behavior
    x_sin = x * np.sin(np.pi * x)
    
    # Combine: use all 10 parameters to weight these basis functions
    # Basis: [1, x, x^2, x^3, x^4, sin(x), cos(x), tanh(x), x*sin(pi*x), sin(2x)]
    basis = np.column_stack([
        x0,
        x1,
        x2,
        x3,
        x4,
        sin_x,
        cos_x,
        tanh_x,
        x_sin,
        sin_2x
    ])
    
    # Linear combination
    return basis @ p
# EVOLVE-BLOCK-END
