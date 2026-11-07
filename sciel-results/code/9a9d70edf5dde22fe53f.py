import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Rich Radial Basis Function (RBF) Network with Fixed Centers.
    
    Instead of a small parametric model (like a 2-neuron MLP) where the optimizer
    struggles with non-convexity in both weights and structure, we use a large
    library of pre-defined RBFs. The centers and widths are fixed deterministically
    based on the input domain [0.5, 2]. The only learnable parameters are the
    linear coefficients (weights) for these basis functions.
    
    This makes the optimization problem convex (linear least squares) with respect
    to the parameters, allowing BFGS to find the optimal combination of these
    rich basis functions reliably. The richness of the basis set (50+ functions)
    allows approximation of complex non-linear targets.
    
    Parameter mapping (10 params):
    The 10 parameters are NOT used directly as weights for 50+ basis functions 
    because we only have 10 slots. Instead, we use the 10 params to define 
    a small set of "active" RBFs or to scale a fixed set. 
    However, a better approach given the constraint of exactly 10 params is to 
    use the 10 params as coefficients for 10 carefully chosen, orthogonalized 
    or diverse basis functions. 
    BUT, the previous inspiration showed that a fixed basis of 10 diverse terms 
    (exp, log, poly) scored 1.3. The MLP scored 10.7. 
    The key insight from inspiration 3 is that "RBF approach" was mentioned. 
    Let's try a hybrid: Use the 10 params as coefficients for 10 specific RBFs 
    centered at different points in the domain with different widths. 
    This is structurally different from the MLP because the non-linearity is 
    in the fixed basis, not in the learned weights of a small network.
    
    Actually, to maximize flexibility with only 10 parameters, we can use the 
    parameters to define the centers of 5 RBFs (2 params each: center_x, center_y 
    for a 2D projection? No, inputs are 3D). 
    Let's stick to the "Rich Fixed Basis" idea but ensure the basis is rich enough.
    If we fix the basis, we need more than 10 basis functions to be rich. 
    But we only have 10 parameters. This implies we can only have 10 basis functions 
    with learnable weights. 
    The previous "fixed basis" attempt (inspiration 1) scored 7.3. 
    The "hybrid basis" (inspiration 3) scored 1.3. 
    The MLP (current) scored 10.7.
    
    Let's try a different structural change: **Kernel Regression with Learned Bandwidths**.
    We define 10 fixed centers (e.g., grid points in [0.5, 2]^3).
    We use the 10 parameters to define the weights of these 10 kernels.
    The widths of the kernels are fixed to a value that covers the domain well.
    This is effectively an RBF network with fixed centers and widths, learnable weights.
    This is convex in the parameters.
    
    Centers: 10 points sampled from the domain.
    Width: sigma = 0.8 (covers most of the domain [0.5, 2] with radius ~1.5).
    
    Wait, if the problem is non-linear, 10 RBFs with fixed centers might not be enough 
    to capture high-frequency features. 
    However, the "Hybrid Basis" in inspiration 3 (score 1.3) likely used a mix of 
    global functions (polys, exps, logs). 
    Let's combine the strengths: Use a set of 10 basis functions that are a mix of 
    global polynomials and local RBFs.
    
    Basis Functions:
    1. 1 (Constant)
    2. x
    3. y
    4. z
    5. x*y
    6. y*z
    7. z*x
    8. x^2 + y^2 + z^2 (Norm squared)
    9. exp(-0.5 * ((x-1.25)^2 + (y-1.25)^2 + (z-1.25)^2)) (Central RBF)
    10. exp(-0.5 * ((x-0.75)^2 + (y-0.75)^2 + (z-0.75)^2)) (Corner RBF)
    
    This set is diverse: linear, interaction, quadratic, and two localized bumps.
    This is structurally different from the MLP (which has arbitrary non-linearities 
    via tanh) and different from a pure polynomial. It is a "hand-crafted" sparse 
    basis designed to cover both global trends and local variations.
    """
    start_time = time.time()
    
    try:
        n = len(x)
        
        # Ensure inputs are 1D arrays
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        z = np.asarray(z).flatten()
        params = np.asarray(params).flatten()
        
        if params.shape[0] < 10:
            # Pad if necessary, though evaluator provides 10
            params = np.pad(params, (0, 10 - params.shape[0]))
        
        # Define the 10 basis functions
        # 1. Constant
        b0 = np.ones(n)
        
        # 2-4. Linear
        b1 = x
        b2 = y
        b3 = z
        
        # 5-7. Interactions
        b4 = x * y
        b5 = y * z
        b6 = z * x
        
        # 8. Quadratic Sum (Isotropic quadratic)
        b7 = x**2 + y**2 + z**2
        
        # 9. Central RBF (Centered at 1.25, width 0.8)
        # Distance squared from center (1.25, 1.25, 1.25)
        dist2_center = (x - 1.25)**2 + (y - 1.25)**2 + (z - 1.25)**2
        sigma2 = 0.8**2
        b8 = np.exp(-dist2_center / (2 * sigma2))
        
        # 10. Corner RBF (Centered at 0.75, 0.75, 0.75, width 0.6)
        dist2_corner = (x - 0.75)**2 + (y - 0.75)**2 + (z - 0.75)**2
        sigma2_corner = 0.6**2
        b9 = np.exp(-dist2_corner / (2 * sigma2_corner))
        
        # Stack into matrix (n, 10)
        # Note: Order must match param indices
        basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
        
        # Linear combination
        # output = basis @ params
        output = basis @ params[:10]
        
        # Safety check
        if not np.all(np.isfinite(output)):
            return np.zeros(n)
            
        # Time guard
        elapsed = time.time() - start_time
        if elapsed > 110:
             # This shouldn't happen as it's O(N), but just in case
             return output
             
        return output

    except Exception:
        try:
            return np.zeros(len(x))
        except:
            return np.array([0.0])
