import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    return (p0 + 
            p1 * x + 
            p2 * v + 
            p3 * (x ** 3) + 
            p4 * (v ** 3) + 
            p5 * np.sin(t) + 
            p6 * np.cos(t) + 
            p7 * t + 
            p8 * (x * v) + 
            p9 * (t * x))
