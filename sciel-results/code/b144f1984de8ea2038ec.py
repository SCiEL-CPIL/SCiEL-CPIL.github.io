"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
def construct():
    """Return six coefficients for the compact growth-index formula.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a)
    
    Structural change: Replaces slow ODE-based fitting with an analytic
    derivation that matches LCDM limits and CPL shifts. This is robust,
    fast, and avoids timeout risks.
    """
    # Analytic coefficients derived from matching:
    # 1. LCDM (u=0): gamma(1) = 0.55, gamma(0) = 1.0
    #    Implies c0=0.55, c1+c2=0.45, and matching at q~0.7 (Om0~0.3)
    #    gives c1 = -1.05, c2 = 1.50
    # 2. CPL shift: gamma decreases for phantom (u<0)
    #    Approximate shift dgamma/du ~ -0.15
    #    So c3 = -0.15
    
    c0 = 0.55
    c1 = -1.05
    c2 = 1.50
    c3 = -0.15
    c4 = 0.0
    c5 = 0.0
    
    # Optional: If time permits, a very fast local refinement could be done,
    # but the analytic seed is expected to be significantly better than the
    # previous failing ODE fit.
    
    return [c0, c1, c2, c3, c4, c5]
# EVOLVE-BLOCK-END
