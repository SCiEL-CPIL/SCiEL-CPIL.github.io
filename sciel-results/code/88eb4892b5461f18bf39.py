import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Global Orthogonal Basis using Shifted Legendre Polynomials.
    
    Instead of using local Gaussian RBFs (which require careful center placement and 
    suffer from potential local minima if centers were optimized, or limited expressivity 
    if fixed) or raw monomials (which suffer from high condition numbers and Runge 
    phenomenon), we use a basis of shifted Legendre polynomials up to degree 3.
    
    The inputs x, y, z are in [0.5, 2.0]. We normalize them to [-1, 1] to match the 
    standard Legendre polynomial domain.
    Let u = (2 * x - 2.5) / 1.5. Since x in [0.5, 2.0], 2x in [1, 4], 2x-2.5 in [-1.5, 1.5], 
    u in [-1, 1].
    
    Shifted Legendre Polynomials P_n(u) on [-1, 1]:
    P_0(u) = 1
    P_1(u) = u
    P_2(u) = 0.5 * (3*u^2 - 1)
    P_3(u) = 0.5 * (5*u^3 - 3*u)
    
    We construct a feature vector for each point. We have 10 parameters.
    We allocate them as follows:
    1. Constant: P_0(x) = 1 (Note: P_0 is same for x,y,z, so we use one)
    2. Linear x: P_1(x)
    3. Linear y: P_1(y)
    4. Linear z: P_1(z)
    5. Quadratic x: P_2(x)
    6. Quadratic y: P_2(y)
    7. Quadratic z: P_2(z)
    8. Cubic x: P_3(x)
    9. Cubic y: P_3(y)
    10. Interaction: P_1(x) * P_1(y)  (A single cross term to capture basic non-additivity)
    
    This basis is orthogonal with respect to the uniform measure on [-1, 1] for each 
    dimension. While the full tensor product would be orthogonal, we are selecting a 
    sparse subset. This subset is well-conditioned and captures smooth variations 
    effectively. The optimization is linear in params, so BFGS will converge reliably 
    to the global minimum of the least squares problem.
    """
    
    start_time = time.time()
    
    # Normalize inputs to [-1, 1]
    # Domain is [0.5, 2.0]
    # u = (2 * val - 2.5) / 1.5
    ux = (2.0 * x - 2.5) / 1.5
    uy = (2.0 * y - 2.5) / 1.5
    uz = (2.0 * z - 2.5) / 1.5
    
    # Compute Legendre Polynomials up to degree 3
    # P0 = 1
    # P1 = u
    # P2 = 0.5 * (3*u^2 - 1)
    # P3 = 0.5 * (5*u^3 - 3*u)
    
    P1_x = ux
    P1_y = uy
    P1_z = uz
    
    P2_x = 0.5 * (3.0 * ux**2 - 1.0)
    P2_y = 0.5 * (3.0 * uy**2 - 1.0)
    P2_z = 0.5 * (3.0 * uz**2 - 1.0)
    
    P3_x = 0.5 * (5.0 * ux**3 - 3.0 * ux)
    P3_y = 0.5 * (5.0 * uy**3 - 3.0 * uy)
    
    # Construct features
    # Feature 0: Constant (P0)
    f0 = np.ones_like(x)
    
    # Feature 1: P1(x)
    f1 = P1_x
    
    # Feature 2: P1(y)
    f2 = P1_y
    
    # Feature 3: P1(z)
    f3 = P1_z
    
    # Feature 4: P2(x)
    f4 = P2_x
    
    # Feature 5: P2(y)
    f5 = P2_y
    
    # Feature 6: P2(z)
    f6 = P2_z
    
    # Feature 7: P3(x)
    f7 = P3_x
    
    # Feature 8: P3(y)
    f8 = P3_y
    
    # Feature 9: Interaction P1(x)*P1(y)
    f9 = P1_x * P1_y
    
    # Stack into a matrix of shape (N, 10)
    # Using column_stack is efficient for small number of columns
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))
    
    # Prediction is features @ params
    pred = features @ params
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:
        pass # Just return, computation is already done
        
    return pred
