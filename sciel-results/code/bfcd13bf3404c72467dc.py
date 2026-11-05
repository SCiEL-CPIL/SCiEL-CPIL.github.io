"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def construct():
    """
    Returns six coefficients for the growth-index ansatz.
    
    The ansatz is:
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    
    where:
    q = 1 - Om(a)
    u = 1 + w0 + wa*(1-a)
    
    f = Om(a)^gamma
    
    We aim to minimize the worst relative error of f and D(a)/D(1) over validation cosmologies.
    
    Physically-motivated initial condition:
    - c0 = 0.55: standard constant growth index for EdS
    - c3 = -0.5: tuned to cancel leading-order w0 dependence
    - Remaining coefficients model subdominant corrections
    
    Based on perturbation theory and numerical fitting to CPL dark energy models:
    - For w0 = -1, wa = 0 (LambdaCDM), gamma ≈ 0.55
    - For varying w0, wa, we need corrections
    """
    # Physically-motivated coefficients
    # c0: baseline growth index for EdS-like
    c0 = 0.55
    
    # c1: linear correction in q (matter density deficit)
    # For small deviations from EdS, gamma increases slightly with q
    c1 = 0.15
    
    # c2: quadratic correction in q
    c2 = -0.05
    
    # c3: correction from u = 1 + w0 + wa*(1-a)
    # Set to -0.5 to cancel leading w0 term
    c3 = -0.5
    
    # c4: correction from wa*a term
    c4 = 0.3
    
    # c5: cross term u*q
    c5 = 0.1
    
    return [c0, c1, c2, c3, c4, c5]
# EVOLVE-BLOCK-END
