import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible quadratic model for E. coli growth rate.
    
    Replaces rigid physical kernels (Monod, Gaussian) with a sparse quadratic polynomial 
    in scaled inputs. This allows BFGS to fit arbitrary parabolic responses and interactions 
    without being constrained by specific functional forms.
    
    Input ranges assumed from task description:
    b: [1, 4.998]
    s: [1, 4.999]
    temp: [1.002, 39.98]
    pH: [2, 12]
    """
    # Unpack 10 parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to approximate [-1, 1] range for numerical stability
    # b: center ~3, scale ~2
    b_scaled = (b - 3.0) / 2.0
    
    # s: center ~3, scale ~2
    s_scaled = (s - 3.0) / 2.0
    
    # temp: center ~20.5, scale ~10
    t_scaled = (temp - 20.5) / 10.0
    
    # pH: center ~7, scale ~5
    ph_scaled = (pH - 7.0) / 5.0
    
    # Construct sparse quadratic basis
    # Terms selected to cover main effects, key interactions, and curvature for T, pH, s
    term_const = 1.0
    term_b = b_scaled
    term_s = s_scaled
    term_t = t_scaled
    term_ph = ph_scaled
    term_t2 = t_scaled * t_scaled
    term_ph2 = ph_scaled * ph_scaled
    term_b_s = b_scaled * s_scaled
    term_s2 = s_scaled * s_scaled
    term_t_ph = t_scaled * ph_scaled
    
    # Linear combination
    # p0: intercept
    # p1: linear b
    # p2: linear s
    # p3: linear T
    # p4: linear pH
    # p5: quadratic T (curvature)
    # p6: quadratic pH (curvature)
    # p7: interaction b*s
    # p8: quadratic s (saturation approximation)
    # p9: interaction T*pH
    
    growth = (p0 * term_const + 
              p1 * term_b + 
              p2 * term_s + 
              p3 * term_t + 
              p4 * term_ph + 
              p5 * term_t2 + 
              p6 * term_ph2 + 
              p7 * term_b_s + 
              p8 * term_s2 + 
              p9 * term_t_ph)
              
    return growth
