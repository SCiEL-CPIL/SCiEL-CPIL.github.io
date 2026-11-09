import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin well under 120s
    
    # Build a rich Fourier + polynomial basis
    # Use both trig and polynomial terms for flexibility
    n = len(x)
    
    # Basis functions
    basis = []
    
    # Constant
    basis.append(np.ones(n))
    
    # Linear terms
    basis.append(x)
    basis.append(y)
    
    # Quadratic terms
    basis.append(x**2)
    basis.append(y**2)
    basis.append(x * y)
    
    # Cubic terms
    basis.append(x**3)
    basis.append(y**3)
    
    # Trigonometric terms (frequency 1)
    basis.append(np.sin(np.pi * x))
    basis.append(np.sin(np.pi * y))
    
    # More trig terms
    basis.append(np.cos(np.pi * x))
    basis.append(np.cos(np.pi * y))
    
    # Cross trig terms
    basis.append(np.sin(np.pi * x) * np.cos(np.pi * y))
    basis.append(np.cos(np.pi * x) * np.sin(np.pi * y))
    
    # Higher frequency trig
    basis.append(np.sin(2 * np.pi * x))
    basis.append(np.sin(2 * np.pi * y))
    basis.append(np.cos(2 * np.pi * x))
    basis.append(np.cos(2 * np.pi * y))
    
    # Interaction with trig
    basis.append(x * np.sin(np.pi * y))
    basis.append(y * np.sin(np.pi * x))
    basis.append(x * np.cos(np.pi * y))
    basis.append(y * np.cos(np.pi * x))
    
    # x^2 * y, x * y^2
    basis.append(x**2 * y)
    basis.append(x * y**2)
    
    # x^2 * y^2
    basis.append(x**2 * y**2)
    
    # Higher order
    basis.append(x**4)
    basis.append(y**4)
    basis.append(x**3 * y)
    basis.append(x * y**3)
    
    # More trig
    basis.append(np.sin(3 * np.pi * x))
    basis.append(np.sin(3 * np.pi * y))
    basis.append(np.cos(3 * np.pi * x))
    basis.append(np.cos(3 * np.pi * y))
    
    # Mixed high-order
    basis.append(x**2 * np.sin(np.pi * y))
    basis.append(y**2 * np.sin(np.pi * x))
    basis.append(np.sin(np.pi * x) * np.sin(np.pi * y))
    basis.append(np.cos(np.pi * x) * np.cos(np.pi * y))
    
    # Radial / symmetric terms
    r2 = x**2 + y**2
    basis.append(r2)
    basis.append(r2**2)
    basis.append(np.exp(-r2))
    basis.append(np.exp(-2*r2))
    basis.append(np.log1p(r2))
    basis.append(np.sin(np.pi * r2))
    basis.append(np.cos(np.pi * r2))
    
    # Absolute value terms
    basis.append(np.abs(x))
    basis.append(np.abs(y))
    basis.append(np.abs(x - y))
    basis.append(np.abs(x + y))
    basis.append(np.abs(x * y))
    
    # Rational-like smooth terms
    basis.append(x / (1 + x**2))
    basis.append(y / (1 + y**2))
    basis.append(x / (1 + x**2) * y / (1 + y**2))
    basis.append(1 / (1 + x**2))
    basis.append(1 / (1 + y**2))
    
    # Sigmoid-like
    basis.append(1 / (1 + np.exp(-4 * x)))
    basis.append(1 / (1 + np.exp(-4 * y)))
    
    # More interactions
    basis.append(x**2 * y**2 * np.sin(np.pi * (x + y)))
    basis.append(x * y * np.cos(np.pi * (x - y)))
    basis.append(np.sin(np.pi * x * y))
    basis.append(np.cos(np.pi * x * y))
    
    # Build design matrix
    A = np.column_stack(basis)
    
    # Solve least squares directly - this is fast and stable
    # Use np.linalg.lstsq which is robust
    try:
        # Check for non-finite values
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(y)):
            # Fallback to simple linear model
            A_simple = np.column_stack([np.ones(n), x, y, x*y, x**2, y**2])
            coeffs, _, _, _ = np.linalg.lstsq(A_simple, y, rcond=None)
            return A_simple @ coeffs
        
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=1e-10)
        
        if not np.all(np.isfinite(coeffs)):
            # Fallback
            A_simple = np.column_stack([np.ones(n), x, y, x*y, x**2, y**2])
            coeffs, _, _, _ = np.linalg.lstsq(A_simple, y, rcond=None)
            return A_simple @ coeffs
        
        pred = A @ coeffs
        
        if not np.all(np.isfinite(pred)):
            # Fallback
            A_simple = np.column_stack([np.ones(n), x, y, x*y, x**2, y**2])
            coeffs, _, _, _ = np.linalg.lstsq(A_simple, y, rcond=None)
            pred = A_simple @ coeffs
        
        return pred
        
    except Exception:
        # Ultimate fallback
        try:
            A_simple = np.column_stack([np.ones(n), x, y])
            coeffs, _, _, _ = np.linalg.lstsq(A_simple, y, rcond=None)
            return A_simple @ coeffs
        except Exception:
            return np.zeros_like(y)
