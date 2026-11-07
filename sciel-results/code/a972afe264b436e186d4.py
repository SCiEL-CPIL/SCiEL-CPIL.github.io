import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

        # Normalization with centering
        # x range approx [-0.2056, 0.1749] -> center -0.015, scale 0.19
        # v range approx [-0.3391, 0.3624] -> center 0.011, scale 0.35
        xn = (x + 0.015) / 0.19
        vn = (v - 0.011) / 0.35

        # Precompute
        xn2 = xn * xn
        xn3 = xn2 * xn

        # Trig
        sin_t = np.sin(t)
        cos_t = np.cos(t)

        # Basis
        b0 = 1.0
        b1 = xn
        b2 = xn3
        b3 = vn
        b4 = xn2 * vn
        b5 = xn * vn
        b6 = cos_t
        b7 = sin_t
        b8 = xn * cos_t
        b9 = xn * sin_t

        return (p0 * b0 + 
                p1 * b1 + 
                p2 * b2 + 
                p3 * b3 + 
                p4 * b4 + 
                p5 * b5 + 
                p6 * b6 + 
                p7 * b7 + 
                p8 * b8 + 
                p9 * b9)
