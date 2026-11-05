"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    # Based on CPL growth index approximations:
    # gamma ≈ 0.55 + 0.05*(1+w0) - 0.02*wa
    # In our ansatz: gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # where u = 1+w0+wa*(1-a), q = 1-Om(a)
    # Set c0=0.55, c3=0.05, c4=-0.02, rest=0
    return [0.55, 0.0, 0.0, 0.05, -0.02, 0.0]
# EVOLVE-BLOCK-END
