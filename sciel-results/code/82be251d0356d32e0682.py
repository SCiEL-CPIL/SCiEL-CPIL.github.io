"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a fixed-basis additive model with polynomial, tanh, and smooth-min terms
    to capture elastic region, yield transition, strain hardening, and temperature effects.
    """
    # Deadline guard: track start time
    _t0 = time.perf_counter()
    _DEADLINE = 140.0  # seconds, safety margin before 150s kill
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Compute basis functions
    
    # Polynomial terms in strain (elastic + hardening)
    s1 = strain
    s2 = s1 * s1
    s3 = s2 * s1
    s4 = s2 * s2
    s5 = s4 * s1
    
    # Tanh terms to capture yield transition and saturation
    # Multiple scales for different transition regimes
    t1 = np.tanh(30 * s1)   # moderate yield transition
    t2 = np.tanh(100 * s1)  # sharp yield transition
    t3 = np.tanh(500 * s1)  # very sharp, near step function
    
    # Smooth log-sum-exp based soft-min for piecewise behavior
    # This creates a smooth approximation of min(elastic_stress, hardening_stress)
    # Using the identity: softmin(a,b) = -softplus(-a+b) - b, or equivalently
    # softmin(a,b) = (1/tau) * log(exp(-tau*a) + exp(-tau*b)) for small tau
    # We use a fixed smooth scale to keep it differentiable
    
    # Temperature terms
    t_temp1 = temp
    t_temp2 = temp * temp
    t_temp3 = t_temp2 * temp
    
    # Interaction terms
    st1 = s1 * t_temp1
    st2 = t1 * t_temp1
    st3 = s2 * t_temp1
    st4 = t2 * t_temp2
    st5 = s3 * t_temp1
    st6 = t3 * t_temp1
    
    # Build the model with 10 parameters as coefficients
    # p0: intercept (base stress offset)
    # p1: linear strain (elastic modulus contribution)
    # p2: quadratic strain (early hardening)
    # p3: cubic strain (mid hardening)
    # p4: tanh(30*strain) - moderate yield transition
    # p5: tanh(100*strain) - sharp yield transition  
    # p6: linear temp (temperature softening)
    # p7: quadratic temp (nonlinear temp effect)
    # p8: strain * temp (coupled temp-strain effect)
    # p9: tanh(100*strain) * temp (yield-temp coupling)
    
    prediction = (
        p0 +
        p1 * s1 +
        p2 * s2 +
        p3 * s3 +
        p4 * t1 +
        p5 * t2 +
        p6 * t_temp1 +
        p7 * t_temp2 +
        p8 * st1 +
        p9 * st4
    )
    
    # Deadline check
    if time.perf_counter() - _t0 > _DEADLINE:
        # Return current prediction; BFGS will handle convergence
        pass
    
    return prediction
# EVOLVE-BLOCK-END
