"""Piecewise-linear ramp model for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: A linear elastic term plus a sum of clipped linear (ramp) functions
    to capture yield point and plastic hardening. Temperature dependence is
    incorporated by scaling the yield thresholds and hardening slopes.
    
    stress = p0 * strain * (1 - p1 * temp) 
           + sum_{i=0}^{3} p_{i+2} * max(0, strain - t_i) * (1 - p_{i+6} * temp)
    
    where t_i = p_{i+5} for i in 0..3 are the yield strain thresholds.
    
    Parameters:
        p0: Elastic modulus (scaled by (1 - p1*temp))
        p1: Temperature coefficient for elastic modulus
        p1..p4 (indices 2-5): Hardening slopes
        t0..t3 (indices 5-8): Yield strain thresholds
        p5..p8 (indices 6-9): Temperature coefficients for hardening slopes
    """
    # Extract parameters
    p0, p1 = params[0], params[1]
    slopes = params[2:6]      # p2, p3, p4, p5
    thresholds = params[5:9]  # p6, p7, p8, p9
    temp_coeffs = params[9]   # single temp coefficient for hardening
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus
    E_eff = p0 * (1.0 - p1 * temp)
    
    # Elastic term
    elastic = E_eff * strain
    
    # Temperature-dependent hardening slopes
    hardening_factor = 1.0 - temp_coeffs * temp
    effective_slopes = slopes * hardening_factor
    
    # Sum of ramp functions: sum_i slope_i * max(0, strain - threshold_i)
    plastic = np.zeros_like(strain)
    for i in range(4):
        ramp = np.maximum(0.0, strain - thresholds[i])
        plastic += effective_slopes[i] * ramp
    
    prediction = elastic + plastic
    
    return prediction
# EVOLVE-BLOCK-END
