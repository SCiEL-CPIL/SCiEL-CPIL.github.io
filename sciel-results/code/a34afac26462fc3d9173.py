"""Piecewise-linear ramp model for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Model: a sum of clipped linear (ramp) functions to capture yield point
    and hardening, plus a linear elastic term.
    
    stress = p0*strain + sum_i p_i * max(0, strain - t_i)
    where t_i = p_{5+i} for i in 0..3 (yield strain thresholds)
    and p_1..p_4 are the slopes of the ramp functions.
    
    Temperature dependence is folded into the parameters via the linear
    elastic term: we use p0*(1 - p1*temp) for the elastic slope.
    
    Actually, let's use all 10 params more effectively:
    - p0: base elastic modulus
    - p1: temperature coefficient for elastic modulus
    - p2: slope of first ramp (yield plateau)
    - p3: threshold for first ramp
    - p4: slope of second ramp (hardening)
    - p5: threshold for second ramp
    - p6: slope of third ramp
    - p7: threshold for third ramp
    - p8: slope of fourth ramp
    - p9: threshold for fourth ramp
    
    stress = p0*(1 - p1*temp)*strain + p2*max(0, strain - p3) 
           + p4*max(0, strain - p5) + p6*max(0, strain - p7) + p8*max(0, strain - p9)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Elastic term with temperature dependence
    elastic = p0 * (1.0 - p1 * temp) * strain
    
    # Ramp functions (clipped linear terms)
    ramp1 = np.maximum(0.0, strain - p3)
    ramp2 = np.maximum(0.0, strain - p5)
    ramp3 = np.maximum(0.0, strain - p7)
    ramp4 = np.maximum(0.0, strain - p9)
    
    prediction = elastic + p2 * ramp1 + p4 * ramp2 + p6 * ramp3 + p8 * ramp4
    
    return prediction
# EVOLVE-BLOCK-END
