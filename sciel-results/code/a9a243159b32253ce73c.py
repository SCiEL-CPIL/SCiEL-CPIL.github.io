"""Stress-strain model for aluminium rod with piecewise elastic-plastic behavior."""
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: piecewise linear elastic-plastic with temperature-dependent properties.
    Uses smooth approximation for differentiability.
    
    params:
      p0: Young's modulus (elastic slope)
      p1: yield strain threshold
      p2: plastic slope (tangent modulus)
      p3: temperature coefficient for modulus
      p4: temperature coefficient for yield
      p5: baseline stress offset
      p6: curvature parameter (hardening)
      p7: temperature interaction with strain
      p8: higher-order temperature term
      p9: additional offset
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-adjusted parameters
    # Normalize temperature to [0,1] range approximately
    T_norm = T  # temp is already in [0.06667, 1]
    
    # Effective modulus decreases with temperature
    E_eff = p0 * (1.0 + p3 * T_norm)
    
    # Effective yield strain increases with temperature (softening)
    eps_y = p1 * (1.0 + p4 * T_norm)
    
    # Smooth step function to transition from elastic to plastic
    # Use a smooth sigmoid-like transition
    # x = (eps - eps_y) / width
    width = 0.005 + abs(p6) * 0.01  # small width for sharp but smooth transition
    x = (eps - eps_y) / (width + 1e-12)
    
    # Smooth step: 0 for x << 0, 1 for x >> 0
    # Use tanh-based smooth step
    smooth_step = 0.5 * (1.0 + np.tanh(x))
    
    # Elastic stress
    stress_elastic = E_eff * eps
    
    # Plastic stress: yield stress + plastic modulus * (eps - eps_y)
    # Yield stress = E_eff * eps_y
    yield_stress = E_eff * eps_y
    stress_plastic = yield_stress + p2 * (eps - eps_y)
    
    # Blend elastic and plastic
    stress = (1.0 - smooth_step) * stress_elastic + smooth_step * stress_plastic
    
    # Add hardening/softening terms
    # Temperature-dependent offset
    stress += p5 + p7 * T_norm * eps + p8 * T_norm * T_norm
    
    # Small curvature term for better fit
    stress += p6 * eps * (eps - eps_y) * (1.0 - smooth_step)
    
    # Final offset
    stress += p9
    
    return stress
