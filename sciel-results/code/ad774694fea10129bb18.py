"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametric limit-cycle oscillator with bounded tanh nonlinearities.
    Structure: a = -k(x)*x - c(v)*v + F(t)
    where k(x) and c(v) are smooth bounded functions using tanh.
    """
    # Normalize inputs to help BFGS converge from initial guess of 1s
    # t: [30, 50] -> center 40, scale 10
    # x: [-0.2056, 0.1749] -> center -0.015, scale 0.2
    # v: [-0.3391, 0.3624] -> center 0.01, scale 0.35
    
    t_n = (t - 40.0) / 10.0
    x_n = (x + 0.015) / 0.2
    v_n = (v - 0.01) / 0.35
    
    # Bounded nonlinear restoring force: k(x) = 1 + p3*tanh(x_n)
    # This gives a softening/hardening spring depending on sign of p3
    k_x = 1.0 + params[3] * np.tanh(x_n)
    
    # Bounded nonlinear damping: c(v) = 1 + p4*tanh(v_n)
    # This gives nonlinear damping that saturates
    c_v = 1.0 + params[4] * np.tanh(v_n)
    
    # Driving force: combination of sin and cos with bounded amplitude modulation
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Modulated drive: F(t) = p5*sin(t) + p6*cos(t) + p7*sin(t)*tanh(x_n)
    # The last term couples drive to position in a bounded way
    drive = params[5] * sin_t + params[6] * cos_t + params[7] * sin_t * np.tanh(x_n)
    
    # Additional bounded cross-coupling: p8 * tanh(x_n * v_n)
    # This captures nonlinear interaction between position and velocity
    cross = params[8] * np.tanh(x_n * v_n)
    
    # Second harmonic drive for richer dynamics: p9 * sin(2*t)
    drive_2h = params[9] * np.sin(2.0 * t)
    
    # Assemble: a = p0 - p1*k_x*x_n - p2*c_v*v_n + drive + cross + drive_2h
    result = (params[0] 
              - params[1] * k_x * x_n 
              - params[2] * c_v * v_n 
              + drive 
              + cross 
              + drive_2h)
    
    return result
# EVOLVE-BLOCK-END
