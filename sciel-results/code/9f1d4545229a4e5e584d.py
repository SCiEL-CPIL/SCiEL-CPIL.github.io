import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Basis functions:
    # 1, x, y, x^2, y^2 (5 global polynomial terms)
    # 5 RBFs at corners and center (5 local terms)
    # Total 10 parameters.
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Polynomial terms
    term_1 = np.ones_like(x)
    term_2 = x
    term_3 = y
    term_4 = x**2
    term_5 = y**2
    
    # RBF terms
    # Centers: (-1,-1), (1,1), (-1,1), (1,-1), (0,0)
    # Width sigma = 0.5
    sigma = 0.5
    sigma2 = 2 * sigma**2
    
    # Center 1: (-1, -1)
    dist1 = (x - (-1))**2 + (y - (-1))**2
    term_6 = np.exp(-dist1 / sigma2)
    
    # Center 2: (1, 1)
    dist2 = (x - 1)**2 + (y - 1)**2
    term_7 = np.exp(-dist2 / sigma2)
    
    # Center 3: (-1, 1)
    dist3 = (x - (-1))**2 + (y - 1)**2
    term_8 = np.exp(-dist3 / sigma2)
    
    # Center 4: (1, -1)
    dist4 = (x - 1)**2 + (y - (-1))**2
    term_9 = np.exp(-dist4 / sigma2)
    
    # Center 5: (0, 0)
    dist5 = (x - 0)**2 + (y - 0)**2
    term_10 = np.exp(-dist5 / sigma2)
    
    return (params[0] * term_1 +
            params[1] * term_2 +
            params[2] * term_3 +
            params[3] * term_4 +
            params[4] * term_5 +
            params[5] * term_6 +
            params[6] * term_7 +
            params[7] * term_8 +
            params[8] * term_9 +
            params[9] * term_10)
