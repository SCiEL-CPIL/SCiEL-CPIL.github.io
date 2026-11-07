"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Radial Basis Function (RBF) network with 5 Gaussian kernels.
# The 10 parameters are structured as:
# p[0:5] -> x-coordinates of the 5 centers (in t-space, scaled)
# p[5:10] -> weights of the 5 kernels
# 
# Why this is different:
# The current approach uses a fixed global basis (polynomials + sin/cos).
# This approach uses a local basis (Gaussians) with learnable centers.
# This allows the function to fit complex, non-smooth, or highly localized
# nonlinearities that might exist in the oscillator's dynamics (e.g., 
# saturation, switching behavior, or specific phase-space trajectories) 
# which a global polynomial might smooth over or fail to capture.
#
# The centers are constrained to the input range of t [30, 50] to ensure 
# the basis functions are active within the data domain.
# The widths of the Gaussians are fixed to cover the range adequately.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form RBF approximation for the nonlinear oscillator acceleration.
    params: array of length 10.
        p[0:5]: centers for the 5 Gaussian kernels in t-space.
        p[5:10]: weights for the 5 kernels.
    """
    p = params
    
    # Extract centers and weights
    centers = p[0:5]
    weights = p[5:10]
    
    # Ensure centers are within a reasonable range to avoid numerical issues
    # t is in [30, 50]. We map the raw params to this range using a sigmoid-like 
    # or linear scaling if needed, but BFGS starts at 1.0.
    # If we just use raw centers, starting at 1.0 puts them far outside [30,50].
    # We should scale the parameters to the input domain.
    # Let's map p[0:5] from R to [30, 50] using a tanh scaling.
    # scaled_center = 40 + 10 * tanh(raw_center)
    # This ensures centers stay in [30, 50] regardless of parameter values.
    # This improves stability for BFGS.
    
    scaled_centers = 40.0 + 10.0 * np.tanh(centers)
    
    # Width of the Gaussians. 
    # Range is 20 units. 5 centers. Spacing ~4 units.
    # Sigma = 2.0 seems reasonable to cover the space with overlap.
    sigma = 2.0
    
    # Calculate RBF activations
    # t shape: (N,)
    # scaled_centers shape: (5,)
    # We need to compute exp(-((t - center_i)^2) / (2 * sigma^2)) for each i.
    
    # Vectorized computation:
    # diff = t[:, np.newaxis] - scaled_centers[np.newaxis, :]  # Shape (N, 5)
    # rbf = np.exp(-0.5 * (diff / sigma) ** 2)                # Shape (N, 5)
    # y = rbf @ weights                                       # Shape (N,)
    
    # To save memory/time for large N, we can do it in a loop or rely on broadcasting.
    # N=10000, 5 kernels. Broadcasting is fine.
    
    diff = t[:, np.newaxis] - scaled_centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # The x and v inputs are not used in the centers in this simplified 1D-RBF.
    # Wait, the task is a nonlinear oscillator. The acceleration depends on x and v.
    # A 1D RBF on t alone is insufficient. It ignores the state dependence.
    # 
    # Revised Strategy: 3D RBF is too expensive/complex for 10 params.
    # Let's use a **Product Basis** or **Modulated RBF**.
    # 
    # Alternative Structural Change: **Neural Network-like Feedforward with Tanh**
    # Instead of RBF, use a small MLP structure with fixed random weights 
    # (or symmetric) and learnable biases/outputs? No, params are limited to 10.
    #
    # Let's try a **Polynomial in x and v modulated by t**.
    # The previous attempt used a global polynomial. 
    # What if we use **Chebyshev Polynomials** for better numerical stability?
    # Or **Fourier Series** with higher harmonics?
    #
    # Let's look at the "Nonlinear Oscillator" hint again.
    # Common form: a = f(x, v) + g(t)
    # If we assume the nonlinearity is in x and v, and the drive is in t.
    # 
    # New Structural Idea: **Separable Basis with Interaction Terms**
    # Basis: 
    # 1. 1
    # 2. x
    # 3. v
    # 4. x^2
    # 5. v^2
    # 6. x*v
    # 7. x^3
    # 8. v^3
    # 9. x^2*v
    # 10. x*v^2
    # This is a pure polynomial in state variables, ignoring t.
    # This assumes the system is autonomous (no explicit time dependence) or 
    # the time dependence is absorbed into the initial conditions/trajectory.
    # However, t is an input. If the system is driven, t matters.
    #
    # Let's try a **Hybrid**: State Polynomials + Time Modulation.
    # a = (p0 + p1*x + p2*v + p3*x^2 + p4*v^2) * (1 + p5*cos(t) + p6*sin(t)) + p7*x^3 + p8*v^3 + p9
    # This is getting complicated and might have local minima.
    
    # Let's stick to the **RBF idea but make it 3D** using a small number of centers?
    # No, 10 params is not enough for 3D centers + weights.
    
    # **Different Design Family: Gaussian Process-like Kernel Regression (Fixed Kernels)**
    # Use a set of fixed, diverse basis functions that are universal approximators.
    # Basis:
    # 1. 1
    # 2. x
    # 3. v
    # 4. x**2
    # 5. v**2
    # 6. x*v
    # 7. np.tanh(x)
    # 8. np.tanh(v)
    # 9. np.tanh(x*v)
    # 10. np.tanh(x + v)
    
    # This introduces **Non-linear Activation Functions** (tanh) which are better 
    # at capturing saturation and smooth non-linearities than pure polynomials.
    # Polynomials diverge at the edges; tanh is bounded.
    # This is a structural change from "Polynomial Basis" to "Neural Basis" (fixed topology).
    
    p = params
    
    # Pre-compute basis functions
    f0 = 1.0
    f1 = x
    f2 = v
    f3 = x * x
    f4 = v * v
    f5 = x * v
    f6 = np.tanh(x)
    f7 = np.tanh(v)
    f8 = np.tanh(x * v)
    f9 = np.tanh(x + v)
    
    # Note: We are dropping t from the basis in this specific variant?
    # If the system is autonomous, t is irrelevant.
    # If it's driven, t is crucial.
    # The previous program had sin(t)/cos(t). 
    # Let's include a time term.
    # Replace f9 with a time term?
    # Let's use:
    # f9 = np.sin(t)  # Drive
    
    # Let's try this basis:
    # 1. 1
    # 2. x
    # 3. v
    # 4. x**2
    # 5. v**2
    # 6. x*v
    # 7. np.tanh(x)
    # 8. np.tanh(v)
    # 9. np.sin(t)
    # 10. np.cos(t)
    
    f9 = np.sin(t)
    f10 = np.cos(t) # Wait, I have 10 params.
    
    # Let's refine the 10-term basis to be "Neural-like" + "Drive":
    # 1. 1
    # 2. x
    # 3. v
    # 4. x**2
    # 5. v**2
    # 6. x*v
    # 7. np.tanh(x)
    # 8. np.tanh(v)
    # 9. np.sin(t)
    # 10. np.cos(t)
    
    y = (p[0] * f0 + 
         p[1] * f1 + 
         p[2] * f2 + 
         p[3] * f3 + 
         p[4] * f4 + 
         p[5] * f5 + 
         p[6] * f6 + 
         p[7] * f7 + 
         p[8] * np.sin(t) + 
         p[9] * np.cos(t))
         
    return y

# EVOLVE-BLOCK-END
