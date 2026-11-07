import numpy as np

def equation(x, y, params):
    # Full cubic polynomial in x and y
    # Basis: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    return (p0 + p1 * x + p2 * y + p3 * x**2 + p4 * x * y + p5 * y**2 + 
            p6 * x**3 + p7 * x**2 * y + p8 * x * y**2 + p9 * y**3)
