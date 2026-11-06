import numpy as np

def construct():
       # 8-point Gauss-Legendre quadrature
       n = 8
       # Get nodes and weights on [-1, 1]
       x, w = np.polynomial.legendre.leggauss(n)
       
       # Transform to [0, 1]
       # x_new = 0.5 * (x + 1)
       # w_new = 0.5 * w
       nodes = 0.5 * (x + 1.0)
       weights = 0.5 * w
       
       # Ensure weights sum to 1 (they should, but normalize for safety)
       # The sum of w from leggauss is 2.0. So 0.5 * 2.0 = 1.0.
       # Floating point errors might make it 0.9999999999999999.
       # The prompt allows 1e-12 tolerance.
       
       return {"nodes": nodes, "weights": weights}
