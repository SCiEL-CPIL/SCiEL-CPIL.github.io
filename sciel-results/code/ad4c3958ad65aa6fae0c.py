# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate: A hybrid model using 4 fixed-center Gaussian Radial Basis Functions (RBFs)
    with learnable widths and weights, plus a linear term in x and a bias.
    
    This approach differs from the polynomial basis by using localized, non-polynomial basis functions.
    The centers are fixed to cover the domain [-1, 1]^2:
      C1 = (0, 0)
      C2 = (0.5, 0.5)
      C3 = (-0.5, -0.5)
      C4 = (0.5, -0.5)
    
    Parameters (10 total):
      p0: weight for RBF 1
      p1: weight for RBF 2
      p2: weight for RBF 3
      p3: weight for RBF 4
      p4: width (inverse variance) for RBF 1
      p5: width (inverse variance) for RBF 2
      p6: width (inverse variance) for RBF 3
      p7: width (inverse variance) for RBF 4
      p8: coefficient for linear term x
      p9: bias term
    
    Model:
      pred = p9 + p8 * x + p0 * exp(-p4 * ((x-0)^2 + (y-0)^2)) 
                           + p1 * exp(-p5 * ((x-0.5)^2 + (y-0.5)^2))
                           + p2 * exp(-p6 * ((x+0.5)^2 + (y+0.5)^2))
                           + p3 * exp(-p7 * ((x-0.5)^2 + (y+0.5)^2))
    
    This allows fitting localized structures and global trends. The fixed centers avoid the symmetry
    breaking issues associated with learning centers from a symmetric initialization.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure widths are positive to avoid overflow in exp
    # We use softplus or just clip? BFGS might try negative widths.
    # exp(-negative * r^2) = exp(positive * r^2) -> overflow.
    # We must constrain widths to be positive.
    # A common trick is to use exp(params) or abs(params) or clip.
    # Using exp(params) ensures positivity.
    g1 = np.exp(p4)
    g2 = np.exp(p5)
    g3 = np.exp(p6)
    g4 = np.exp(p7)
    
    # Fixed centers
    # C1: (0, 0)
    r1_sq = x*x + y*y
    
    # C2: (0.5, 0.5)
    dx2 = x - 0.5
    dy2 = y - 0.5
    r2_sq = dx2*dx2 + dy2*dy2
    
    # C3: (-0.5, -0.5)
    dx3 = x + 0.5
    dy3 = y + 0.5
    r3_sq = dx3*dx3 + dy3*dy3
    
    # C4: (0.5, -0.5)
    dx4 = x - 0.5
    dy4 = y + 0.5
    r4_sq = dx4*dx4 + dy4*dy4
    
    # Compute RBFs
    rbf1 = np.exp(-g1 * r1_sq)
    rbf2 = np.exp(-g2 * r2_sq)
    rbf3 = np.exp(-g3 * r3_sq)
    rbf4 = np.exp(-g4 * r4_sq)
    
    # Linear combination
    pred = p9 + p8 * x + p0 * rbf1 + p1 * rbf2 + p2 * rbf3 + p3 * rbf4
    
    return pred
# EVOLVE-BLOCK-END
