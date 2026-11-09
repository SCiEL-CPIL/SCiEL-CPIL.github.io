import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Normalize to [-1, 1]
        # x, y, z in [0.5, 2.0]
        # Center 1.25, Range 1.5 -> Half-range 0.75
        u = (x - 1.25) / 0.75
        v = (y - 1.25) / 0.75
        w = (z - 1.25) / 0.75
        
        p = params
        
        # Basis functions
        # p0: 1
        # p1: u
        # p2: v
        # p3: w
        # p4: u^2
        # p5: v^2
        # p6: w^2
        # p7: uv
        # p8: uw
        # p9: vw
        
        pred = (p[0] 
                + p[1] * u 
                + p[2] * v 
                + p[3] * w 
                + p[4] * u * u 
                + p[5] * v * v 
                + p[6] * w * w 
                + p[7] * u * v 
                + p[8] * u * w 
                + p[9] * v * w)
                
        return pred
