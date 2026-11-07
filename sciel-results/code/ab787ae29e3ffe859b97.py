# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Symbolic regression using a linear combination of 10 diverse basis functions.
        
        The model is:
        f(x) = w0*1 + w1*x + w2*x^2 + w3*x^3 + w4*x^4 
               + w5*exp(x/3) + w6*exp(-x/3) + w7*tanh(x) + w8*sin(x) + w9*cos(x)
        
        This model is linear in the parameters (weights), making the optimization
        problem convex. BFGS will converge to the global least-squares solution
        efficiently.
        
        The basis includes:
        - Polynomials up to degree 4 to capture smooth polynomial trends.
        - Exponential terms to capture growth/decay behavior.
        - Tanh to capture saturation/step-like behavior.
        - Sine and cosine to capture periodic behavior.
        """
        x = np.asarray(x, dtype=np.float64)
        
        # Guard against non-finite inputs
        x = np.where(np.isfinite(x), x, 0.0)
        
        # Extract parameters (weights)
        w = params[:10]
        
        # Build the basis matrix Phi of shape (N, 10)
        
        # 1. Constant
        phi0 = np.ones_like(x)
        
        # 2-5. Polynomials
        phi1 = x
        phi2 = x**2
        phi3 = x**3
        phi4 = x**4
        
        # 6-7. Exponentials (scaled by 1/3 to keep values O(1) on [-3,3])
        phi5 = np.exp(x / 3.0)
        phi6 = np.exp(-x / 3.0)
        
        # 8. Tanh (saturation)
        phi7 = np.tanh(x)
        
        # 9-10. Trigonometric (low frequency)
        phi8 = np.sin(x)
        phi9 = np.cos(x)
        
        # Stack into matrix
        Phi = np.column_stack((phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9))
        
        # Compute prediction: y = Phi @ w
        # Ensure w is finite
        w = np.where(np.isfinite(w), w, 0.0)
        
        result = Phi @ w
        
        # Ensure finite output
        result = np.where(np.isfinite(result), result, 0.0)
        
        return result
    # EVOLVE-BLOCK-END
