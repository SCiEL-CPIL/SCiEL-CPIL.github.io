import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Trigonometric basis model to capture periodic or oscillatory behaviors.
    Basis functions:
    1. 1
    2. sin(x)
    3. cos(x)
    4. sin(y)
    5. cos(y)
    6. sin(x)*sin(y)
    7. cos(x)*cos(y)
    8. sin(x)*cos(y)
    9. cos(x)*sin(y)
    10. cos(2*x)*cos(2*y)  (Higher frequency interaction)
    """
    # Precompute trig functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)
    
    # Higher frequency term
    cos_2x = np.cos(2 * x)
    cos_2y = np.cos(2 * y)
    
    p = params
    
    # Linear combination
    # Index 0: Constant
    # Index 1: sin(x)
    # Index 2: cos(x)
    # Index 3: sin(y)
    # Index 4: cos(y)
    # Index 5: sin(x)*sin(y)
    # Index 6: cos(x)*cos(y)
    # Index 7: sin(x)*cos(y)
    # Index 8: cos(x)*sin(y)
    # Index 9: cos(2x)*cos(2y)
    
    pred = (p[0] * 1.0 + 
            p[1] * sin_x + 
            p[2] * cos_x + 
            p[3] * sin_y + 
            p[4] * cos_y + 
            p[5] * sin_x * sin_y + 
            p[6] * cos_x * cos_y + 
            p[7] * sin_x * cos_y + 
            p[8] * cos_x * sin_y + 
            p[9] * cos_2x * cos_2y)
            
    return pred
