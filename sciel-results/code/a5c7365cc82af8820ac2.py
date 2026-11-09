import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Extract parameters
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
        
        # Compute powers
        x2 = x * x
        y2 = y * y
        x3 = x2 * x
        y3 = y2 * y
        
        # Compute prediction
        # Basis:
        # 0: 1
        # 1: x
        # 2: y
        # 3: x^2
        # 4: y^2
        # 5: xy
        # 6: x^3
        # 7: y^3
        # 8: x^2 y
        # 9: x y^2
        
        pred = (p0 + 
                p1 * x + 
                p2 * y + 
                p3 * x2 + 
                p4 * y2 + 
                p5 * x * y + 
                p6 * x3 + 
                p7 * y3 + 
                p8 * x2 * y + 
                p9 * x * y2)
                
        return pred
