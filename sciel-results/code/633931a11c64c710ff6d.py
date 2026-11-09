import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Basis functions:
    # 1. Constant
    # 2. Linear x
    # 3. Linear y
    # 4. Quadratic x^2
    # 5. Quadratic y^2
    # 6. Interaction xy
    # 7. Sine of x (scaled to pi to cover period 2)
    # 8. Sine of y
    # 9. Cosine of x
    # 10. Cosine of y
    
    term_1 = np.ones_like(x)
    term_2 = x
    term_3 = y
    term_4 = x**2
    term_5 = y**2
    term_6 = x * y
    term_7 = np.sin(np.pi * x)
    term_8 = np.sin(np.pi * y)
    term_9 = np.cos(np.pi * x)
    term_10 = np.cos(np.pi * y)
    
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
