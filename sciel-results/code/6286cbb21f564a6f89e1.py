import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recover an unknown function using a mixed polynomial and trigonometric basis.
    This basis allows the model to capture both polynomial trends and periodic
    components, which is advantageous for symbolic regression tasks where the
    target function may involve trigonometric operations.
    
    Input x is scaled to [-1, 1] for the trigonometric parts to maintain
    consistent frequency scaling, but we can also use the raw scaled u.
    Basis:
    0: 1
    1: u
    2: u^2
    3: u^3
    4: u^4
    5: sin(pi * u)
    6: cos(pi * u)
    7: sin(2 * pi * u)
    8: cos(2 * pi * u)
    9: sin(3 * pi * u)
    """
    x = np.asarray(x, dtype=np.float64)
    u = x / 3.0
    
    n_points = x.size
    
    B = np.zeros((n_points, 10), dtype=np.float64)
    
    # Polynomial terms
    B[:, 0] = 1.0
    B[:, 1] = u
    B[:, 2] = u**2
    B[:, 3] = u**3
    B[:, 4] = u**4
    
    # Trigonometric terms (scaled to [-1, 1] so pi*u covers [-pi, pi])
    B[:, 5] = np.sin(np.pi * u)
    B[:, 6] = np.cos(np.pi * u)
    B[:, 7] = np.sin(2.0 * np.pi * u)
    B[:, 8] = np.cos(2.0 * np.pi * u)
    B[:, 9] = np.sin(3.0 * np.pi * u)
    
    return B @ params
