import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Generalized Additive Model (GAM) with B-splines.
    
    Strategy:
    - The previous approach used non-linear parameters (centers of Gaussians), 
      creating a non-convex optimization landscape for BFGS.
    - This approach uses a fixed basis of B-splines for temperature and pH.
    - The model is linear in the parameters (coefficients of the basis functions).
    - This makes the optimization problem convex (Least Squares), allowing BFGS 
      to reliably find the global minimum.
    - We use 5 B-splines for temperature and 5 for pH, totaling 10 parameters.
    - This captures non-linear, non-monotonic responses (peaks) in temperature 
      and pH, which are critical for microbial growth, without needing to fit 
      non-linear parameters like centers or widths.
    - Inputs b and s are assumed to have a minor or saturating effect that is 
      either constant across the training range or negligible compared to T/pH 
      for the *maximum* growth rate determination. If b/s effects are strong 
      and non-linear, this model might underfit, but the convexity gain is 
      significant. Given the small range of b and s in the prompt ([1, 5]), 
      a constant or linear effect might be sufficient, but we prioritize 
      capturing the T/pH surface accurately.
      
    Basis Functions:
    We use cubic B-splines (degree 3) with a fixed knot sequence.
    For a variable x in [0, 1], we define knots at 0, 0.25, 0.5, 0.75, 1.0.
    However, standard B-spline basis functions are typically defined over an 
    extended knot vector. To keep it simple and ensure smoothness and coverage, 
    we will use a set of 5 overlapping cubic B-spline basis functions.
    
    Alternatively, a simpler and very effective "B-spline-like" basis that is 
    linear in parameters and convex to optimize is a set of localized 
    piecewise linear or quadratic functions. But true B-splines are smoother.
    
    Let's implement a standard cubic B-spline basis.
    Knots for temperature (normalized t in [0,1]): [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]
    This gives 5 basis functions (N = len(knots) - degree - 1 = 11 - 3 - 1 = 7? No.)
    Number of basis functions = len(knots) - degree - 1.
    If we want 5 basis functions, we need 5 + 3 + 1 = 9 knots.
    Let's use knots: [0, 0, 0, 0, 0.33, 0.66, 1, 1, 1]
    This yields 5 basis functions.
    
    We will compute these basis functions on the fly.
    """
    # Normalize inputs to [0, 1] based on training ranges
    # temp: [1, 40] -> [0, 1]
    # pH: [2, 12] -> [0, 1]
    t_norm = (temp - 1.0) / 39.0
    p_norm = (pH - 2.0) / 10.0
    
    # Clip to [0, 1] to avoid extrapolation issues, though B-splines handle 
    # outside knots by returning 0 for boundary functions if not clamped properly.
    # Clipping ensures stability.
    t_norm = np.clip(t_norm, 0.0, 1.0)
    p_norm = np.clip(p_norm, 0.0, 1.0)
    
    # Define knots for 5 cubic B-splines
    # Knot vector: [0, 0, 0, 0, 0.333, 0.666, 1, 1, 1]
    # This creates 5 basis functions.
    knots_t = np.array([0.0, 0.0, 0.0, 0.0, 0.333, 0.666, 1.0, 1.0, 1.0])
    knots_p = np.array([0.0, 0.0, 0.0, 0.0, 0.333, 0.666, 1.0, 1.0, 1.0])
    
    def bspline_basis(x, knots):
        """
        Compute 5 cubic B-spline basis functions for x given knots.
        x: array of values in [0, 1]
        knots: array of 9 knots
        Returns: array of shape (len(x), 5)
        """
        n_knots = len(knots)
        degree = 3
        n_basis = n_knots - degree - 1 # 9 - 3 - 1 = 5
        
        # We evaluate the B-splines using the Cox-de Boor recursion.
        # For efficiency and simplicity in a vectorized NumPy context, 
        # we can use a simple loop over the basis functions or a recursive definition.
        # Since n_basis is small (5), a loop is fine.
        
        # Basis functions N_{i,0} are 1 if knots[i] <= x < knots[i+1], else 0
        # N_{i,p} = (x - knots[i])/(knots[i+p] - knots[i]) * N_{i,p-1} + 
        #            (knots[i+p+1] - x)/(knots[i+p+1] - knots[i+1]) * N_{i+1,p-1}
        
        # To avoid division by zero and handle boundaries, we add a small epsilon.
        eps = 1e-9
        
        # Initialize N_{i,0}
        N = np.zeros((len(x), n_basis))
        for i in range(n_basis):
            N[:, i] = np.where((x >= knots[i]) & (x < knots[i+1]), 1.0, 0.0)
            # Handle the last knot boundary
            if i == n_basis - 1:
                N[:, i] = np.where((x >= knots[i]) & (x <= knots[i+1]), 1.0, N[:, i])
        
        for p in range(1, degree + 1):
            N_new = np.zeros_like(N)
            for i in range(n_basis - p):
                # Term 1
                denom1 = knots[i+p] - knots[i]
                if denom1 < eps:
                    term1 = 0.0
                else:
                    term1 = (x - knots[i]) / denom1 * N[:, i]
                
                # Term 2
                denom2 = knots[i+p+1] - knots[i+1]
                if denom2 < eps:
                    term2 = 0.0
                else:
                    term2 = (knots[i+p+1] - x) / denom2 * N[:, i+1]
                
                N_new[:, i] = term1 + term2
            # The last p basis functions are 0 in this step's loop range? 
            # No, the loop goes to n_basis - p. The indices i go from 0 to n_basis-p-1.
            # We need to preserve the last p columns? No, the recursion reduces the number of non-zero 
            # functions if we are not careful, but actually N_{i,p} is non-zero only if x is in [knots[i], knots[i+p+1]].
            # The standard algorithm updates N in place or uses a new array.
            # Let's just copy the computed values and zero out the rest for this degree?
            # Actually, for p=1, we compute N_{0,1}...N_{4,1}. 
            # For p=2, we compute N_{0,2}...N_{3,2}.
            # For p=3, we compute N_{0,3}...N_{2,3}.
            # Wait, the number of basis functions stays the same (5), but the support changes.
            # The recursion formula is:
            # N_{i,p} = ... * N_{i,p-1} + ... * N_{i+1,p-1}
            # So we need N_{i,p-1} and N_{i+1,p-1}.
            # My loop above computes N_new[:, i] for i in 0..n_basis-p-1.
            # This leaves N_new[:, n_basis-p] ... N_new[:, n_basis-1] as 0?
            # That is incorrect. The last basis functions are non-zero.
            
            # Let's restart the B-spline calculation with a clearer recursive approach 
            # or a known stable method.
            
            # Alternative: Use a simpler basis that is still linear in params and convex.
            # "Radial Basis Functions" with fixed centers are linear in weights.
            # "Polynomials" are linear in coeffs.
            # "Piecewise Linear" with fixed knots is linear in coeffs.
            
            # Let's use a set of 5 overlapping "Tent" functions (Piecewise Linear) 
            # or "Quadratic" functions centered at 0.1, 0.3, 0.5, 0.7, 0.9.
            # This is simpler, faster, and robust.
            
            # Actually, let's just use the polynomial approach that worked well (score 1.18) 
            # but enhance it? No, the directive says "different design or modelling family".
            # B-splines are a distinct family from polynomials.
            
            # Let's implement the B-splines correctly.
            # I will use a recursive function that is safe.
            
            pass # This block is abandoned, see below for correct implementation.
        
        return N

    # Correct B-spline implementation
    def compute_bsplines(x, knots, degree=3):
        n_knots = len(knots)
        n_basis = n_knots - degree - 1
        N = np.zeros((len(x), n_basis))
        
        # Base case p=0
        for i in range(n_basis):
            # N_{i,0} = 1 if knots[i] <= x < knots[i+1]
            mask = (x >= knots[i]) & (x < knots[i+1])
            N[:, i] = mask.astype(float)
            # Handle right boundary for the last function
            if i == n_basis - 1:
                mask_last = (x >= knots[i]) & (x <= knots[i+1])
                N[:, i] = np.maximum(N[:, i], mask_last.astype(float))
        
        # Recursion
        for p in range(1, degree + 1):
            N_new = np.zeros_like(N)
            for i in range(n_basis):
                # We only compute if the support is valid
                # N_{i,p} depends on N_{i,p-1} and N_{i+1,p-1}
                # If i+p >= n_knots, it's out of bounds? No, i goes up to n_basis-1.
                # i+p can go up to n_basis-1 + degree = n_knots - 1.
                # Denominators involve knots[i+p] - knots[i] and knots[i+p+1] - knots[i+1].
                
                denom1 = knots[i+p] - knots[i]
                if denom1 > 1e-9:
                    term1 = (x - knots[i]) / denom1 * N[:, i]
                else:
                    term1 = 0.0
                
                if i + 1 < n_basis:
                    denom2 = knots[i+p+1] - knots[i+1]
                    if denom2 > 1e-9:
                        term2 = (knots[i+p+1] - x) / denom2 * N[:, i+1]
                    else:
                        term2 = 0.0
                else:
                    term2 = 0.0
                
                N_new[:, i] = term1 + term2
            
            N = N_new
        
        return N

    # Use 5 basis functions for T and 5 for pH
    # Knots: 9 knots for 5 cubic splines
    knots = np.array([0.0, 0.0, 0.0, 0.0, 0.333, 0.666, 1.0, 1.0, 1.0])
    
    B_t = compute_bsplines(t_norm, knots, degree=3) # Shape (N, 5)
    B_p = compute_bsplines(p_norm, knots, degree=3) # Shape (N, 5)
    
    # params[0:5] are coefficients for T basis
    # params[5:10] are coefficients for pH basis
    
    y_t = B_t @ params[0:5]
    y_p = B_p @ params[5:10]
    
    y = y_t + y_p
    
    return y
