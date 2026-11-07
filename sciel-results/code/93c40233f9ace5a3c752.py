import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params.astype(float)
    
    # Product-of-sums decomposition: f(x,y) = g(x) * h(y)
    # g(x) = sum of 5 basis functions of x (params 0-4)
    # h(y) = sum of 5 basis functions of y (params 5-9)
    
    # Basis for x: 1, x, x^2, x^3, x^4
    x_basis = np.stack([
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4
    ], axis=-1)
    
    # Basis for y: 1, y, y^2, y^3, y^4
    y_basis = np.stack([
        np.ones_like(y),
        y,
        y**2,
        y**3,
        y**4
    ], axis=-1)
    
    # g(x) = x_basis @ p[0:5]
    g_x = x_basis @ p[0:5]
    
    # h(y) = y_basis @ p[5:10]
    h_y = y_basis @ p[5:10]
    
    return g_x * h_y
